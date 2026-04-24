#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  -  cu_caf_4x.py
# Cross Ambiguity Function block, 4-surveillance-channel variant (CuPy GPU-accelerated)
#
# GPU-accelerated 4-channel CAF. Accepts one reference and four surveillance
# vectors per CPI and emits four CAF matrices simultaneously.
#
# Key compute saving vs four separate cu_caf blocks:
#   The Doppler steering tones are applied to the REFERENCE signal only
#   (same passive-radar textbook formulation as cu_caf). Therefore
#   FFT(ref_shifted) and its conjugate are IDENTICAL for all four
#   surveillance channels. This block computes those (nd, N) complex arrays
#   ONCE per CPI and reuses them across all four surv FFT correlations,
#   saving (num_doppler_bins × cpi_samp) complex multiplications and
#   (num_doppler_bins) forward FFTs per CPI relative to four independent blocks.
#
# GNU Radio block type: sync_block (vector in -> vector out, 1:1 CPI)
#
# Inputs
# ------
#   Port 0  ref    - complex64 vector, length = cpi_samp  (reference channel)
#   Port 1  surv1  - complex64 vector, length = cpi_samp  (surveillance channel 1)
#   Port 2  surv2  - complex64 vector, length = cpi_samp  (surveillance channel 2)
#   Port 3  surv3  - complex64 vector, length = cpi_samp  (surveillance channel 3)
#   Port 4  surv4  - complex64 vector, length = cpi_samp  (surveillance channel 4)
#
# Outputs
# -------
#   Port 0  caf_out1 - complex64 vector, length = num_doppler_bins * num_delay_bins
#   Port 1  caf_out2 - complex64 vector, length = num_doppler_bins * num_delay_bins
#   Port 2  caf_out3 - complex64 vector, length = num_doppler_bins * num_delay_bins
#   Port 3  caf_out4 - complex64 vector, length = num_doppler_bins * num_delay_bins
#
#   Each output is a row-major flattened CAF matrix:
#     row  = Doppler bin  (centred, zero-Doppler at centre row)
#     col  = delay (range) bin
#   Phase is preserved for downstream DOA processing across the 4 channels.
#

import numpy as np
import cupy as cp
import gnuradio.gr as gr


class cu_caf_4x(gr.sync_block):
    """
    Cross Ambiguity Function, 4-channel (CAF 4x) -- gr-PR  [CuPy GPU]

    Computes CAF between one reference signal and four surveillance signals
    for one CPI, emitting four CAF matrices. All heavy computation runs on
    an NVIDIA GPU via CuPy / cuFFT.

    The Doppler steering tones are applied to the REFERENCE signal (x_ref),
    and the Doppler-shifted reference FFTs are reused across all four
    surveillance correlations. Only the per-channel surveillance FFTs and
    the final multiply+IFFT differ between channels.

    Parameters
    ----------
    cpi_samp         : int   - CPI length in samples (must match stream_to_vector)
    num_doppler_bins : int   - number of Doppler frequency bins
    num_delay_bins   : int   - number of delay (range) bins  <= cpi_samp
    cpi_skip         : int   - process 1 in every N CPIs (1 = process all)
    sample_rate      : float - receiver sample rate in Hz
    max_doppler      : float - maximum Doppler shift in Hz
    """

    _NUM_SURV = 4  # fixed number of surveillance channels

    def __init__(self,
                 cpi_samp=24000,
                 num_doppler_bins=128,
                 num_delay_bins=128,
                 cpi_skip=1,
                 sample_rate=2.4e6,
                 max_doppler=1000.0):

        # Validate
        if num_delay_bins > cpi_samp:
            raise ValueError("num_delay_bins must be <= cpi_samp")

        out_len = num_doppler_bins * num_delay_bins
        vec_in  = np.dtype((np.complex64, cpi_samp))
        vec_out = np.dtype((np.complex64, out_len))

        gr.sync_block.__init__(
            self,
            name="PR CuPy CAF 4x",
            in_sig=[vec_in] * (1 + self._NUM_SURV),   # ref + 4 surv
            out_sig=[vec_out] * self._NUM_SURV,        # 4 CAF outputs
        )

        self._N   = cpi_samp
        self._nd  = num_doppler_bins
        self._nl  = num_delay_bins
        self._fs  = sample_rate
        self._max_doppler = float(max_doppler)
        self._skip = max(1, int(cpi_skip))
        self._skip_ctr = 0

        # -- Pre-compute Hann window (GPU) -----------------------------------
        # Applied to all channels before cross-correlation.
        # Reduces range sidelobes caused by signal discontinuities at
        # CPI boundaries (Gibbs phenomenon).
        self._window = cp.hanning(cpi_samp).astype(cp.float32)

        # -- Pre-compute Doppler steering matrix (GPU) -----------------------
        # Shape: (num_doppler_bins, cpi_samp)
        # Each row d: exp(j*2*pi*v_d*t)
        # Applied to the REFERENCE signal per textbook formulation.
        doppler_freqs = cp.linspace(-max_doppler, max_doppler, num_doppler_bins)
        t = cp.arange(cpi_samp, dtype=cp.float32) / sample_rate
        self._doppler_tones = cp.exp(
            1j * 2.0 * cp.pi
            * doppler_freqs[:, cp.newaxis]   # (nd, 1)
            * t[cp.newaxis, :]               # (1,  N)
        ).astype(cp.complex64)               # -> (nd, N)

        # -- Pre-allocate per-channel output matrices (GPU) ------------------
        # One (nd, nl) buffer per surveillance channel, reused each CPI.
        # Keeping them separate avoids an extra GPU copy when writing outputs.
        self._caf_mats = [
            cp.zeros((num_doppler_bins, num_delay_bins), dtype=cp.complex64)
            for _ in range(self._NUM_SURV)
        ]

    # -------------------------------------------------------------------------
    def work(self, input_items, output_items):
        """
        GNU Radio work() -- called by the scheduler with one or more CPI vectors.

        input_items[0]   shape: (n_cpis, cpi_samp) -- reference  (CPU numpy)
        input_items[1-4] shape: (n_cpis, cpi_samp) -- surv 1..4  (CPU numpy)
        output_items[0-3] shape: (n_cpis, nd * nl) -- CAF 1..4   (CPU numpy)
        """
        ref_vecs = input_items[0]
        n_cpis   = len(ref_vecs)
        produced = 0

        for i in range(n_cpis):
            # -- CPI decimation ----------------------------------------------
            # If cpi_skip > 1, silently drop CPIs to reduce GPU load.
            # The scheduler still consumes all input vectors.
            self._skip_ctr += 1
            if self._skip_ctr < self._skip:
                continue
            self._skip_ctr = 0

            # -- Transfer reference to GPU and apply window ------------------
            ref_gpu = cp.asarray(ref_vecs[i]) * self._window

            # -- Compute shared Doppler-shifted reference FFTs ---------------
            # This is the expensive part that is REUSED across all 4 channels:
            # conj(FFT(ref * tone_d)) for every Doppler bin d simultaneously.
            # Shape: (nd, N) -- lives on GPU, never copied to host.
            CONJ_REF_ALL = cp.conj(
                cp.fft.fft(
                    ref_gpu[cp.newaxis, :] * self._doppler_tones,  # (nd, N)
                    axis=1,
                )
            )  # (nd, N) complex64

            # -- Per-channel: surv FFT, multiply, IFFT, write output ---------
            for ch in range(self._NUM_SURV):
                # Transfer this channel's surveillance vector host->device once
                surv_gpu = cp.asarray(input_items[ch + 1][i]) * self._window

                # SURV FFT computed once per channel; broadcast across all
                # Doppler bins via the [newaxis, :] indexing below
                SURV = cp.fft.fft(surv_gpu)  # (N,) complex64

                # Batched cross-correlation over all Doppler bins:
                #   CAF[d, :] = IFFT( FFT(surv) * conj(FFT(ref_shifted[d])) )
                R_all = cp.fft.ifft(
                    SURV[cp.newaxis, :] * CONJ_REF_ALL,  # (nd, N)
                    axis=1,
                )  # (nd, N) complex64

                # Truncate to num_delay_bins and store into pre-allocated buffer
                self._caf_mats[ch][:] = R_all[:, :self._nl].astype(cp.complex64)

                # Transfer to host and write into GNU Radio output buffer
                output_items[ch][produced] = cp.asnumpy(self._caf_mats[ch].ravel())

            produced += 1

        return produced

    # -------------------------------------------------------------------------
    # Runtime-adjustable parameters (wirable from GRC variable blocks)
    def set_cpi_skip(self, val):
        self._skip = max(1, int(val))

    def get_cpi_samp(self):         return self._N
    def get_num_doppler_bins(self): return self._nd
    def get_num_delay_bins(self):   return self._nl
    def get_sample_rate(self):      return self._fs
    def get_max_doppler(self):      return self._max_doppler
