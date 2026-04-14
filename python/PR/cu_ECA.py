#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  cu_ECA.py
# Direct-path Disturbance Cancellation — CuPy GPU variant of ECA.py
#
# Same Block Lattice Filter algorithm as ECA.py (Table 5.4 in ref. [4]),
# ported to CuPy.  Public API and I/O signature match ECA exactly so it
# is a drop-in replacement in a flowgraph.
#
# GPU MEMORY NOTES
# ─────────────────
#   ECA is purely element-wise vector work plus dot-product reductions.
#   There are no FFTs, so no cuFFT plan cache — the steady-state VRAM
#   footprint is just five N-long complex64 buffers (≈ 5·N·8 bytes).
#
#   The main GPU memory risk on a flowgraph that ALSO runs cu_caf is
#   that CuPy's default memory pool grows and doesn't shrink on its
#   own.  Two mitigations are exposed as constructor args:
#
#     release_mem_each_cpi : call mempool.free_all_blocks() at the end
#                            of every work() to release transient
#                            allocations held by the pool.  Small cost,
#                            prevents slow memory creep.
#
#     mem_pool_limit_mb    : hard cap on CuPy's default memory pool
#                            (0 = no limit).  Set to e.g. 2048 and the
#                            pool will refuse to grow past 2 GB,
#                            forcing CuPy to reuse blocks rather than
#                            allocate new ones.  This is a PROCESS-wide
#                            setting shared with cu_caf; set it on
#                            whichever GPU block instantiates first.
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
#   Port 0  ref        — complex64 vector, length = cpi_samp  (pass-through)
#   Port 1  surv_clean — complex64 vector, length = cpi_samp
#

import numpy as np
import cupy as cp
import gnuradio.gr as gr


class cu_ECA(gr.sync_block):
    """
    Direct-path Disturbance Cancellation — gr-PR  [CuPy GPU]

    Drop-in replacement for ECA. See ECA.py for the algorithm
    description — this class performs the same Block Lattice Filter
    on an NVIDIA GPU via CuPy.

    Parameters
    ----------
    cpi_samp  : int    — CPI length in samples (must match stream_to_vector)
    num_taps  : int    — lattice order M (delay bins to cancel)
    reg_mu    : float  — loading added to the lattice denominators
    release_mem_each_cpi : bool — free CuPy pool blocks after every CPI
                                  to prevent slow VRAM creep when
                                  running alongside cu_caf.  Default True.
    mem_pool_limit_mb : int  — hard cap on CuPy's default memory pool
                                in MB.  0 = no limit (default).
                                Affects the whole process, not just this block.
    """

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

        gr.sync_block.__init__(
            self,
            name="PR CuPy ECA",
            in_sig=[
                np.dtype((np.complex64, cpi_samp)),  # port 0: reference
                np.dtype((np.complex64, cpi_samp)),  # port 1: surveillance
            ],
            out_sig=[
                np.dtype((np.complex64, cpi_samp)),  # port 0: ref (pass-through)
                np.dtype((np.complex64, cpi_samp)),  # port 1: surv_clean
            ]
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
        # Reused across every CPI and every stage — zero allocations in
        # the hot loop.  Total footprint: 5 · N · 8 bytes.
        self._b         = cp.empty(cpi_samp, dtype=cp.complex64)
        self._f         = cp.empty(cpi_samp, dtype=cp.complex64)
        self._e         = cp.empty(cpi_samp, dtype=cp.complex64)
        self._b_delayed = cp.empty(cpi_samp, dtype=cp.complex64)
        self._b_new     = cp.empty(cpi_samp, dtype=cp.complex64)

    # ────────────────────────────────────────────────────────────────────
    def work(self, input_items, output_items):
        ref_vecs  = input_items[0]
        surv_vecs = input_items[1]
        out_ref   = output_items[0]
        out_surv  = output_items[1]
        n_cpis    = len(ref_vecs)

        M  = self._K
        mu = self._mu

        # Local aliases — saves attribute lookups inside the inner loop
        b, f, e               = self._b, self._f, self._e
        b_delayed, b_new      = self._b_delayed, self._b_new

        for i in range(n_cpis):
            # ── Host → device transfer ──────────────────────────────────
            ref_gpu  = cp.asarray(ref_vecs[i])
            surv_gpu = cp.asarray(surv_vecs[i])

            # ── Initialise lattice state:  b_0 = f_0 = x_r,  e_0 = x_e ──
            cp.copyto(b, ref_gpu)
            cp.copyto(f, ref_gpu)
            cp.copyto(e, surv_gpu)

            for _ in range(M):
                # Delayed backward prediction error  b_i(n−1)
                # (zero at n = 0 by convention).  In-place into preallocated buffer.
                b_delayed[0]  = 0
                b_delayed[1:] = b[:-1]

                # Reflection coefficient
                #   κ_{i+1} = 2 Σ b_i(n−1) f_i*(n)
                #             ───────────────────────────────
                #             Σ (|f_i(n)|² + |b_i(n−1)|²) + μ
                num   = 2.0 * cp.vdot(f, b_delayed)
                den   = (cp.vdot(f, f).real
                         + cp.vdot(b_delayed, b_delayed).real
                         + mu)
                kappa = num / den

                # Joint-process filter tap  h_i = Σ e_i b_i* / Σ|b_i|²
                # Computed BEFORE b is overwritten by the prediction-error update.
                b_norm = cp.vdot(b, b).real + mu
                h = cp.vdot(b, e) / b_norm

                # Filter-branch update:  e_{i+1} = e_i − h_i b_i
                e -= h * b

                # Prediction-error update:
                #   b_{i+1}(n) = b_i(n−1) − κ_{i+1}  f_i(n)
                #   f_{i+1}(n) = f_i(n)   − κ*_{i+1} b_i(n−1)
                cp.subtract(b_delayed, kappa * f, out=b_new)
                f -= cp.conj(kappa) * b_delayed
                b, b_new = b_new, b  # swap — b_new becomes the next b

            # After M stages, e = cleaned surveillance
            # Keep local aliases pointing at the right buffers for next CPI
            self._b, self._b_new = b, b_new

            # ── Device → host transfer ──────────────────────────────────
            out_ref[i]  = ref_vecs[i]          # pass-through, no GPU roundtrip
            out_surv[i] = cp.asnumpy(e)

        # Release transient blocks held by CuPy's pool (FFT plan cache
        # from cu_caf, temporary reductions, etc.) so VRAM doesn't creep.
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
