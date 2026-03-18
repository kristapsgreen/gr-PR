#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  DDC.py
# Direct-path Disturbance Cancellation block (Wiener / ECA filter)
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALGORITHM: Extensive Cancellation Algorithm (ECA)                 │
# │             — Batch Least-Squares Wiener Filter                    │
# │                                                                     │
# │  In passive radar the surveillance channel contains:               │
# │    surv[n] = target echoes + direct path + multipath clutter       │
# │                                                                     │
# │  The direct path and short-delay multipath are modelled as a       │
# │  linear FIR filter applied to the reference signal:                │
# │                                                                     │
# │    d[n] = Σ_{k=0}^{K-1}  w[k] · ref[n − k]                       │
# │                                                                     │
# │  where K = num_taps (number of filter taps / delay bins to cancel) │
# │                                                                     │
# │  MATRIX FORMULATION (per CPI):                                     │
# │                                                                     │
# │    Build X  (N × K) Toeplitz-like matrix of delayed reference:     │
# │      X[n, k] = ref[n − k]   (zero-padded for n < k)               │
# │                                                                     │
# │    Solve for weights w via least squares:                          │
# │      w = (X^H X + μI)^{-1}  X^H  surv                            │
# │                                                                     │
# │    μI is Tikhonov regularisation (diagonal loading) to stabilise   │
# │    the inversion when X^H X is ill-conditioned.                    │
# │                                                                     │
# │    Clean surveillance:                                              │
# │      surv_clean = surv − X w                                       │
# │                                                                     │
# │  WHAT THIS REMOVES:                                                 │
# │    - Direct-path signal (zero-Doppler, zero-delay)                 │
# │    - Static / slow multipath clutter within K delay bins           │
# │    - Any zero-Doppler component up to delay K                      │
# │                                                                     │
# │  WHAT SURVIVES:                                                     │
# │    - Target echoes with non-zero Doppler (moving targets)          │
# │    - Echoes beyond K delay bins                                    │
# │                                                                     │
# │  After DDC, the cleaned surveillance goes to the CAF block.        │
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


class DDC(gr.sync_block):
    """
    Direct-path Disturbance Cancellation (DDC) — gr-PR

    Removes the direct-path signal and short-delay multipath clutter
    from the surveillance channel using the ECA batch Wiener filter.

    Place this block BETWEEN stream_to_vector and the CAF block:
        stream_to_vector → DDC → CAF

    The reference input passes through unchanged on the output;
    only the surveillance channel is cleaned.

    Parameters
    ----------
    cpi_samp  : int   — CPI length in samples (must match stream_to_vector)
    num_taps  : int   — filter length (number of delay bins to cancel)
                        Larger = cancels more multipath but costs more CPU.
                        Start with 100–300 for DVB-T at 2.4 MHz.
    reg_mu    : float — Tikhonov regularisation parameter (diagonal loading)
                        Small positive value (e.g. 1e-3 to 1.0) stabilises
                        the matrix inversion.  Increase if you see artefacts.
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
            name="PR DDC",
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

        for i in range(n_cpis):
            ref  = ref_vecs[i]
            surv = surv_vecs[i]

            # ── Build Toeplitz reference matrix X (N × K) ──────────────
            #   X[n, k] = ref[n − k],  zero for n < k
            #
            # Using stride tricks for efficiency: we prepend K-1 zeros to
            # the reference and take K consecutive shifts via as_strided.
            padded = np.zeros(self._N + self._K - 1, dtype=np.complex64)
            padded[self._K - 1:] = ref

            strides = (padded.strides[0], padded.strides[0])
            X = np.lib.stride_tricks.as_strided(
                padded, shape=(self._N, self._K), strides=strides
            ).copy()  # copy to ensure contiguous memory

            # ── Solve for filter weights ───────────────────────────────
            #   w = (X^H X + μI)^{-1}  X^H  surv
            XhX = X.conj().T @ X                          # (K × K)
            XhX += self._mu * np.eye(self._K, dtype=np.complex64)
            Xhs = X.conj().T @ surv                       # (K,)
            w = np.linalg.solve(XhX, Xhs)                 # (K,)

            # ── Subtract estimated interference ────────────────────────
            surv_clean = surv - X @ w

            # ── Write outputs ──────────────────────────────────────────
            out_ref[i]  = ref
            out_surv[i] = surv_clean.astype(np.complex64)

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
