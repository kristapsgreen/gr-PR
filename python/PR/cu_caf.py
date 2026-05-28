#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  -  cu_caf.py
# Cross Ambiguity Function block (CuPy GPU-accelerated)
#
# GPU-accelerated version of caf.py using CuPy.
# Same algorithm, same inputs/outputs -- runs on NVIDIA GPU via cuFFT.
#
# Changes vs CPU caf.py:
#   - numpy -> cupy for all array operations
#   - Pre-computed arrays (window, doppler_tones, caf_mat) live in GPU VRAM
#   - Input vectors transferred host->device with cp.asarray() each CPI
#   - Output transferred device->host with cp.asnumpy() for GNU Radio
#   - FFT uses cuFFT under the hood (cp.fft.fft / cp.fft.ifft)
#
# Algorithm fix (both caf.py and cu_caf.py):
#   Doppler steering tones are applied to the REFERENCE signal (x_ref),
#   NOT the surveillance signal. The surveillance (echo) goes straight to
#   the correlator. This matches the passive radar textbook formulation
#   (Fig. 4.2): COR(x_surv, x_ref_shifted).
#
# GNU Radio block type: sync_block (vector in -> vector out, 1:1 CPI)
#
# Inputs
# ------
#   Port 0  ref   - complex64 vector, length = cpi_samp
#   Port 1  surv  - complex64 vector, length = cpi_samp
#
# Output
# ------
#   Port 0  caf_out - complex64 vector, length = num_doppler_bins * num_delay_bins
#                     Row-major: row = Doppler bin, col = delay bin
#                     Doppler axis is centred (zero-Doppler at centre row)
#                     Phase preserved for downstream DOA processing
#

import numpy as np
import cupy as cp
import gnuradio.gr as gr


class cu_caf(gr.sync_block):
    """
    Cross Ambiguity Function (CAF) -- gr-PR  [CuPy GPU]

    GPU-accelerated CAF between a reference and surveillance signal for one CPI.
    Identical algorithm to the CPU caf block, but all heavy computation runs
    on an NVIDIA GPU via CuPy / cuFFT.

    The Doppler steering tones are applied to the REFERENCE signal (x_ref),
    and the result is cross-correlated with the surveillance/echo signal (x_surv).
    This matches the standard passive radar CAF formulation (see Fig. 4.2).

    Parameters
    ----------
    cpi_samp         : int   - CPI length in samples (must match stream_to_vector)
    num_doppler_bins : int   - number of Doppler frequency bins
    num_delay_bins   : int   - number of delay (range) bins  <= cpi_samp
    cpi_skip         : int   - process 1 in every N CPIs (1 = process all)
    sample_rate      : float - receiver sample rate in Hz
    max_doppler      : float - maximum Doppler shift in Hz
    """

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

        gr.sync_block.__init__(
            self,
            name="PR CuPy CAF",
            in_sig=[
                np.dtype((np.complex64, cpi_samp)),  # port 0: reference
                np.dtype((np.complex64, cpi_samp)),  # port 1: surveillance
            ],
            out_sig=[
                np.dtype((np.complex64, num_doppler_bins * num_delay_bins)),
            ]
        )

        self._N   = cpi_samp
        self._nd  = num_doppler_bins
        self._nl  = num_delay_bins
        self._fs  = sample_rate
        self._max_doppler = float(max_doppler)
        self._skip = max(1, int(cpi_skip))
        self._skip_ctr = 0

        # -- Pre-compute Hann window (GPU) -----------------------------------
        # Applied to both channels before cross-correlation.
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

        # -- Pre-allocate output matrix (GPU) --------------------------------
        # Complex output preserves phase for downstream DOA processing.
        self._caf_mat = cp.zeros((num_doppler_bins, num_delay_bins),
                                  dtype=cp.complex64)

    # -------------------------------------------------------------------------
    def work(self, input_items, output_items):
        """
        GNU Radio work() -- called by the scheduler with one or more CPI vectors.

        input_items[0] shape: (n_cpis, cpi_samp)  -- reference  (CPU numpy)
        input_items[1] shape: (n_cpis, cpi_samp)  -- surveillance (CPU numpy)
        output_items[0] shape: (n_cpis, nd * nl)  -- flattened CAF (CPU numpy)
        """
        ref_vecs  = input_items[0]
        surv_vecs = input_items[1]
        out       = output_items[0]
        n_cpis    = len(ref_vecs)
        produced  = 0

        for i in range(n_cpis):
            # -- CPI decimation ----------------------------------------------
            # If cpi_skip > 1, silently drop CPIs to reduce GPU load.
            # The scheduler still consumes all input vectors.
            self._skip_ctr += 1
            if self._skip_ctr < self._skip:
                continue
            self._skip_ctr = 0

            # -- Transfer to GPU (host -> device) ----------------------------
            ref_gpu  = cp.asarray(ref_vecs[i])
            surv_gpu = cp.asarray(surv_vecs[i])

            # -- Apply window (GPU) ------------------------------------------
            ref_w  = ref_gpu  * self._window
            surv_w = surv_gpu * self._window

            # -- Compute CAF (GPU) -------------------------------------------
            self._compute_caf(ref_w, surv_w)

            # -- Transfer back to CPU (device -> host) and write output ------
            out[produced] = cp.asnumpy(self._caf_mat.ravel())
            produced += 1

        return produced

    # -------------------------------------------------------------------------
    def _compute_caf(self, ref_w, surv_w):
        """
        Core CAF computation (vectorised on GPU). Writes result into self._caf_mat.

        Doppler tones are applied to the REFERENCE signal (x_ref), matching
        the passive radar textbook formulation (Fig. 4.2):

          R_shifted = ref_w[newaxis, :] * doppler_tones        (nd, N)  GPU
          SURV      = FFT(surv_w)                               (N,)     GPU (cuFFT)
          CAF[d,:]  = IFFT( SURV * conj(FFT(R_shifted[d,:])) ) truncated to nl
        """
        # Vectorised Doppler shift on REFERENCE: (1, N) * (nd, N) -> (nd, N)
        R_shifted = ref_w[cp.newaxis, :] * self._doppler_tones

        # FFT of surveillance (echo) signal - computed once for all Doppler bins
        SURV = cp.fft.fft(surv_w)                               # (N,) complex64

        # Batched FFT cross-correlation: IFFT( FFT(surv) * conj(FFT(ref_shifted)) )
        R_all = cp.fft.ifft(
            SURV[cp.newaxis, :] * cp.conj(cp.fft.fft(R_shifted, axis=1)),
            axis=1,
        )                                                        # (nd, N) complex

        # Keep first num_delay_bins lags (positive delays only), preserve complex
        self._caf_mat[:] = R_all[:, :self._nl].astype(cp.complex64)

    # -------------------------------------------------------------------------
    # Runtime-adjustable parameters (wirable from GRC variable blocks)
    def set_cpi_skip(self, val):
        self._skip = max(1, int(val))

    def get_cpi_samp(self):        return self._N
    def get_num_doppler_bins(self): return self._nd
    def get_num_delay_bins(self):   return self._nl
    def get_sample_rate(self):      return self._fs
    def get_max_doppler(self):      return self._max_doppler
