#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  ECA.py
# Direct-path Disturbance Cancellation block (Block Lattice Filter)
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALGORITHM: Block Lattice Filter (Table 5.4 in ref. [4])           │
# │             — Order-recursive Wiener filter, O(N·M) per CPI        │
# │                                                                     │
# │  In passive radar the surveillance channel contains:               │
# │    surv[n] = target echoes + direct path + multipath clutter       │
# │                                                                     │
# │  The batch ECA formulation solves                                  │
# │      w = (X^H X + μI)^{-1} X^H surv                                │
# │  which costs O(N·K² + K³) per CPI (Toeplitz build + matrix solve). │
# │                                                                     │
# │  The block lattice filter achieves the same suppression with only │
# │  O(N·M) ops per CPI by factoring the Wiener solution into M        │
# │  order-recursive lattice stages that orthogonalise the reference   │
# │  on the fly — no matrix is ever built or inverted.                 │
# │                                                                     │
# │  STAGE RECURSION  (i = 0 … M−1):                                   │
# │                                                                     │
# │    Initialisation:                                                  │
# │      b_0(n) = x_r(n)         (backward prediction error)           │
# │      f_0(n) = x_r(n)         (forward  prediction error)           │
# │      e_0(n) = x_e(n)         (surveillance to be cleaned)          │
# │                                                                     │
# │    Reflection coefficient:                                          │
# │                 2 Σ_n  b_i(n−1) f_i*(n)                            │
# │      κ_{i+1} = ─────────────────────────────                       │
# │                Σ_n ( |f_i(n)|² + |b_i(n−1)|² )                     │
# │                                                                     │
# │    Prediction-error update:                                         │
# │      b_{i+1}(n) = b_i(n−1) − κ_{i+1}  f_i(n)                       │
# │      f_{i+1}(n) = f_i(n)   − κ*_{i+1} b_i(n−1)                     │
# │                                                                     │
# │    Filter (joint-process) branch:                                   │
# │               Σ_n e_i(n) b_i*(n)                                   │
# │      h_i = ─────────────────────                                   │
# │               Σ_n |b_i(n)|²                                        │
# │      e_{i+1}(n) = e_i(n) − h_i  b_i(n)                             │
# │                                                                     │
# │  After M stages, e_M(n) is the cleaned surveillance.               │
# │                                                                     │
# │  μ (reg_mu) is added to the reflection / filter denominators as a  │
# │  small loading term to avoid division by zero if a stage fully     │
# │  cancels the remaining reference energy.                           │
# │                                                                     │
# │  WHAT THIS REMOVES:                                                 │
# │    - Direct-path signal (zero-Doppler, zero-delay)                 │
# │    - Static / slow multipath clutter within M delay bins           │
# │                                                                     │
# │  WHAT SURVIVES:                                                     │
# │    - Target echoes with non-zero Doppler (moving targets)          │
# │    - Echoes beyond M delay bins                                    │
# │                                                                     │
# │  After ECA, the cleaned surveillance goes to the CAF block.        │
# │  You should see the zero-Doppler ridge suppressed in the RD map,   │
# │  making moving targets visible.                                     │
# └─────────────────────────────────────────────────────────────────────┘
#
# GNU Radio block type: sync_block (vector in → vector out, 1:1 CPI)
#
# Inputs
# ------
#   Port 0  ref       — complex64 vector, length = cpi_samp
#   Port 1  surv      — complex64 vector, length = cpi_samp
#
# Output
# ------
#   Port 0  surv_clean — complex64 vector, length = cpi_samp
#                        (cleaned surveillance, same shape as input)
#

import numpy as np
import gnuradio.gr as gr


class ECA(gr.sync_block):
    """
    Direct-path Disturbance Cancellation (ECA) — gr-PR

    Removes the direct-path signal and short-delay multipath clutter
    from the surveillance channel using an order-recursive block
    lattice (Gram–Schmidt / joint-process) Wiener filter.

    Compared to the batch ECA matrix solve, the block lattice filter
    produces an equivalent clutter-suppression result but scales as
    O(N·M) instead of O(N·M² + M³), so doubling the number of taps
    only doubles the cost.

    Place this block BETWEEN stream_to_vector and the CAF block:
        stream_to_vector → ECA → CAF

    The reference input passes through unchanged on the output;
    only the surveillance channel is cleaned.

    Parameters
    ----------
    cpi_samp  : int   — CPI length in samples (must match stream_to_vector)
    num_taps  : int   — lattice order M (number of delay bins to cancel).
                        Larger = cancels more multipath but costs more CPU.
                        Start with 100–300 for DVB-T at 2.4 MHz.
    reg_mu    : float — loading added to the lattice denominators for
                        numerical safety.  A small positive value
                        (e.g. 1e-3 to 1.0) prevents division by zero if
                        a stage fully nulls its input energy.
    """

    def __init__(self,
                 cpi_samp=24000,
                 num_taps=128,
                 reg_mu=0.01):

        if num_taps > cpi_samp:
            raise ValueError("num_taps must be <= cpi_samp")
        if num_taps < 1:
            raise ValueError("num_taps must be >= 1")

        gr.sync_block.__init__(
            self,
            name="PR ECA",
            in_sig=[
                np.dtype((np.complex64, cpi_samp)),  # port 0: reference
                np.dtype((np.complex64, cpi_samp)),  # port 1: surveillance
            ],
            out_sig=[
                np.dtype((np.complex64, cpi_samp)),  # port 0: ref (pass-through)
                np.dtype((np.complex64, cpi_samp)),  # port 1: surv_clean
            ]
        )

        self._N = cpi_samp
        self._K = num_taps
        self._mu = float(reg_mu)

    # ────────────────────────────────────────────────────────────────────
    def work(self, input_items, output_items):
        ref_vecs  = input_items[0]
        surv_vecs = input_items[1]
        out_ref   = output_items[0]
        out_surv  = output_items[1]
        n_cpis    = len(ref_vecs)

        M  = self._K
        mu = self._mu

        for i in range(n_cpis):
            ref  = ref_vecs[i]
            surv = surv_vecs[i]

            # ── Block Lattice Filter (Table 5.4) ───────────────────────
            # Initialisation:  b_0 = f_0 = x_r,   e_0 = x_e
            b = ref.astype(np.complex64, copy=True)   # b_i
            f = ref.astype(np.complex64, copy=True)   # f_i
            e = surv.astype(np.complex64, copy=True)  # e_i

            for _ in range(M):
                # Delayed backward prediction error  b_i(n−1)
                # (zero at n = 0 by convention)
                b_delayed = np.empty_like(b)
                b_delayed[0]  = 0
                b_delayed[1:] = b[:-1]

                # Reflection coefficient
                #   κ_{i+1} = 2 Σ b_i(n−1) f_i*(n)
                #             ───────────────────────────────
                #             Σ (|f_i(n)|² + |b_i(n−1)|²)
                # np.vdot(f, b_delayed) = Σ conj(f) · b_delayed
                num = 2.0 * np.vdot(f, b_delayed)
                den = (np.vdot(f, f).real
                       + np.vdot(b_delayed, b_delayed).real
                       + mu)
                kappa = num / den

                # Joint-process (filter) tap, computed on b_i BEFORE it
                # is overwritten by the prediction-error update.
                #   h_i = Σ e_i(n) b_i*(n)  /  Σ |b_i(n)|²
                b_norm = np.vdot(b, b).real + mu
                h = np.vdot(b, e) / b_norm

                # Filter-branch update:  e_{i+1} = e_i − h_i b_i
                e -= h * b

                # Prediction-error update:
                #   b_{i+1}(n) = b_i(n−1) − κ_{i+1}  f_i(n)
                #   f_{i+1}(n) = f_i(n)   − κ*_{i+1} b_i(n−1)
                b_new = b_delayed - kappa * f
                f     = f - np.conj(kappa) * b_delayed
                b     = b_new

            # e is now e_M = cleaned surveillance
            out_ref[i]  = ref
            out_surv[i] = e.astype(np.complex64)

        return n_cpis

    # ────────────────────────────────────────────────────────────────────
    # Runtime-adjustable parameters
    def set_num_taps(self, val):
        val = int(val)
        if 1 <= val <= self._N:
            self._K = val

    def set_reg_mu(self, val):
        self._mu = float(val)

    def get_cpi_samp(self):  return self._N
    def get_num_taps(self):  return self._K
    def get_reg_mu(self):    return self._mu
