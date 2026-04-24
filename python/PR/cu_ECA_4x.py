#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  cu_ECA_4x.py
# Direct-path Disturbance Cancellation — CuPy GPU, 4-surveillance-channel variant
#
# Same Block Lattice Filter algorithm as cu_ECA.py (Table 5.4 in ref. [4]),
# extended to process four surveillance channels per CPI while running the
# expensive lattice prediction work only once.
#
# KEY COMPUTE SAVING
# ──────────────────
#   In the lattice inner loop the prediction state (b, f, b_delayed, b_new)
#   and the reflection coefficient kappa depend ONLY on the reference signal.
#   The four surveillance channels differ only in h (the joint-process tap)
#   and e (the running cleaned estimate).
#
#   This block therefore runs the b/f/kappa update ONCE per stage and reuses
#   those results across all four surv channels in a (4, N) matrix operation.
#   Compared with four independent cu_ECA blocks the saving per CPI is
#   approximately 3 × M lattice-prediction sweeps (b_delayed shift, two vdot
#   reductions, the kappa division, and the f/b update — all O(N) work) where
#   M is the lattice order.
#
# GPU MEMORY NOTES
# ─────────────────
#   Steady-state VRAM: 5·N·8 bytes for the ref-side buffers  +  4·N·8 bytes
#   for the (4, N) surv matrix = 9·N·8 bytes ≈ 1.7 MB at N=24000.
#   No FFTs → no cuFFT plan cache growth.
#
#   release_mem_each_cpi and mem_pool_limit_mb behave identically to cu_ECA.
#
# GNU Radio block type: sync_block (vector in → vector out, 1:1 CPI)
#
# Inputs
# ------
#   Port 0  ref   — complex64 vector, length = cpi_samp  (reference channel)
#   Port 1  surv1 — complex64 vector, length = cpi_samp  (surveillance channel 1)
#   Port 2  surv2 — complex64 vector, length = cpi_samp  (surveillance channel 2)
#   Port 3  surv3 — complex64 vector, length = cpi_samp  (surveillance channel 3)
#   Port 4  surv4 — complex64 vector, length = cpi_samp  (surveillance channel 4)
#
# Outputs
# -------
#   Port 0  ref         — complex64 vector, length = cpi_samp  (pass-through)
#   Port 1  surv_clean1 — complex64 vector, length = cpi_samp
#   Port 2  surv_clean2 — complex64 vector, length = cpi_samp
#   Port 3  surv_clean3 — complex64 vector, length = cpi_samp
#   Port 4  surv_clean4 — complex64 vector, length = cpi_samp
#
#   The pass-through ref keeps all five CPIs aligned so the downstream
#   cu_caf_4x block receives the ref and all four cleaned surv from the
#   same CPI without any explicit FIFO buffering.
#

import numpy as np
import cupy as cp
import gnuradio.gr as gr


class cu_ECA_4x(gr.sync_block):
    """
    Direct-path Disturbance Cancellation — gr-PR  [CuPy GPU, 4-channel]

    Four-surveillance-channel variant of cu_ECA. Runs the Block Lattice
    Filter prediction (b, f, kappa) ONCE per CPI stage — shared across
    all four surveillance channels — and applies the joint-process tap h
    and residual update e separately for each channel via a (4, N) matrix.

    Parameters
    ----------
    cpi_samp  : int    — CPI length in samples (must match stream_to_vector)
    num_taps  : int    — lattice order M (delay bins to cancel)
    reg_mu    : float  — loading added to the lattice denominators
    release_mem_each_cpi : bool — free CuPy pool blocks after every CPI
                                  to prevent slow VRAM creep when running
                                  alongside cu_caf_4x.  Default True.
    mem_pool_limit_mb : int  — hard cap on CuPy's default memory pool in MB.
                                0 = no limit (default).
                                Process-wide setting shared with cu_caf_4x;
                                set it on whichever GPU block instantiates first.
    """

    _NUM_SURV = 4  # fixed number of surveillance channels

    def __init__(self,
                 cpi_samp=24000,
                 num_taps=128,
                 reg_mu=0.01,
                 release_mem_each_cpi=True,
                 mem_pool_limit_mb=0):

        if num_taps > cpi_samp:
            raise ValueError("num_taps must be <= cpi_samp")
        if num_taps < 1:
            raise ValueError("num_taps must be >= 1")

        vec = np.dtype((np.complex64, cpi_samp))

        gr.sync_block.__init__(
            self,
            name="PR CuPy ECA 4x",
            in_sig=[vec] * (1 + self._NUM_SURV),   # port 0: ref  +  ports 1-4: surv
            out_sig=[vec] * (1 + self._NUM_SURV),   # port 0: ref pass-through  +  ports 1-4: clean
        )

        self._N  = cpi_samp
        self._K  = int(num_taps)
        self._mu = float(reg_mu)
        self._release_mem = bool(release_mem_each_cpi)

        # ── Optional VRAM cap on CuPy's default pool ─────────────────────
        if mem_pool_limit_mb and mem_pool_limit_mb > 0:
            cp.get_default_memory_pool().set_limit(
                size=int(mem_pool_limit_mb) * 1024 * 1024
            )

        # ── Pre-allocate GPU working buffers ─────────────────────────────
        # Reference-side prediction buffers — shared across all channels.
        # Reused every CPI; zero per-CPI allocations in the hot loop.
        self._b         = cp.empty(cpi_samp, dtype=cp.complex64)
        self._f         = cp.empty(cpi_samp, dtype=cp.complex64)
        self._b_delayed = cp.empty(cpi_samp, dtype=cp.complex64)
        self._b_new     = cp.empty(cpi_samp, dtype=cp.complex64)

        # Surveillance-side residual matrix: shape (4, N).
        # Each row ch holds e_ch — the cleaned estimate for channel ch.
        # Using a contiguous (4, N) array instead of a list of 4 arrays
        # lets CuPy vectorise the h-tap and residual update across all
        # four channels in a single kernel call.
        self._e = cp.empty((self._NUM_SURV, cpi_samp), dtype=cp.complex64)

    # ────────────────────────────────────────────────────────────────────
    def work(self, input_items, output_items):
        ref_vecs = input_items[0]
        out_ref  = output_items[0]
        n_cpis   = len(ref_vecs)

        M  = self._K
        mu = self._mu

        # Local aliases — avoids repeated attribute lookup in the inner loop
        b, f              = self._b, self._f
        b_delayed, b_new  = self._b_delayed, self._b_new
        e                 = self._e          # (4, N)

        for i in range(n_cpis):
            # ── Host → device transfer ──────────────────────────────────
            ref_gpu = cp.asarray(ref_vecs[i])

            # Load all four surveillance vectors into rows of e simultaneously.
            # cp.stack produces a fresh (4, N) array — assign into pre-allocated
            # buffer to avoid a transient allocation.
            e[:] = cp.stack([
                cp.asarray(input_items[ch + 1][i])
                for ch in range(self._NUM_SURV)
            ])

            # ── Initialise lattice state:  b_0 = f_0 = x_r ───────────────
            cp.copyto(b, ref_gpu)
            cp.copyto(f, ref_gpu)

            for _ in range(M):
                # Delayed backward prediction error  b_i(n−1)
                # (zero at n = 0 by convention).
                b_delayed[0]  = 0
                b_delayed[1:] = b[:-1]

                # Reflection coefficient — depends ONLY on the reference.
                #   κ_{i+1} = 2 Σ b_i(n−1) f_i*(n)
                #             ───────────────────────────────
                #             Σ (|f_i(n)|² + |b_i(n−1)|²) + μ
                num   = 2.0 * cp.vdot(f, b_delayed)
                den   = (cp.vdot(f, f).real
                         + cp.vdot(b_delayed, b_delayed).real
                         + mu)
                kappa = num / den

                # Denominator for all joint-process taps at this stage.
                # Computed once; reused by all four channels below.
                b_norm = cp.vdot(b, b).real + mu

                # ── Per-channel joint-process tap + residual update ───────
                # h_ch = Σ e_ch · b* / (Σ|b|² + μ)  [one reduction per channel]
                # e_ch -= h_ch · b                    [rank-1 update]
                #
                # With e as a (4, N) matrix and b as (N,):
                #   e @ b.conj()  gives a (4,) vector of dot products — all
                #   four channels in a single matmul kernel, far cheaper than
                #   four separate cp.vdot calls.
                h_vec = e @ cp.conj(b) / b_norm    # (4,) — one h per channel
                # Outer product h_vec[:, newaxis] * b[newaxis, :] subtracts
                # the rank-1 estimate from every channel simultaneously.
                e -= h_vec[:, cp.newaxis] * b[cp.newaxis, :]

                # ── Prediction-error update (reference only, done ONCE) ───
                #   b_{i+1}(n) = b_i(n−1) − κ_{i+1}  f_i(n)
                #   f_{i+1}(n) = f_i(n)   − κ*_{i+1} b_i(n−1)
                cp.subtract(b_delayed, kappa * f, out=b_new)
                f -= cp.conj(kappa) * b_delayed
                b, b_new = b_new, b  # pointer swap — no data copy

            # After M stages, each row of e is the cleaned surveillance.
            # Keep local aliases consistent with the swapped pointer.
            self._b, self._b_new = b, b_new

            # ── Device → host transfer ──────────────────────────────────
            out_ref[i] = ref_vecs[i]   # pass-through: no GPU roundtrip
            for ch in range(self._NUM_SURV):
                output_items[ch + 1][i] = cp.asnumpy(e[ch])

        # Release transient blocks held by CuPy's pool (temporary reductions,
        # cuFFT plan cache from cu_caf_4x, etc.) so VRAM doesn't creep.
        if self._release_mem:
            cp.get_default_memory_pool().free_all_blocks()

        return n_cpis

    # ────────────────────────────────────────────────────────────────────
    # Runtime-adjustable parameters
    def set_num_taps(self, val):
        val = int(val)
        if 1 <= val <= self._N:
            self._K = val

    def set_reg_mu(self, val):
        self._mu = float(val)

    def set_release_mem_each_cpi(self, val):
        self._release_mem = bool(val)

    def get_cpi_samp(self):  return self._N
    def get_num_taps(self):  return self._K
    def get_reg_mu(self):    return self._mu
