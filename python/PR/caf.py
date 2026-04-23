#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  -  caf.py
# Cross Ambiguity Function block
#
# +---------------------------------------------------------------------+
# |  ALGORITHM: FFT-based Cross Ambiguity Function (CAF)               |
# |                                                                     |
# |  The CAF measures the similarity between two signals as a function  |
# |  of both time delay (t) and Doppler frequency shift (v):           |
# |                                                                     |
# |    CAF(t, v) = integral x_ref(t) * x_surv*(t - T) * e^(j2pvt) dt  |
# |                                                                     |
# |  In passive radar:                                                  |
# |    - x_ref  = reference channel  (direct path from transmitter)    |
# |    - x_surv = surveillance channel (reflections off targets)       |
# |    - A peak at (T0, v0) means a target at range r = c*T0/2         |
# |      moving with radial velocity vel = c*v0 / (2*f_c)              |
# |                                                                     |
# |  DISCRETE IMPLEMENTATION (Batch FFT method):                       |
# |                                                                     |
# |  For each Doppler bin d with frequency shift v_d:                  |
# |                                                                     |
# |    1. DOPPLER SHIFT (applied to REFERENCE - matches textbook Fig.4.2)|
# |       Multiply reference by a complex tone at v_d:                 |
# |         r_shifted[n] = x_ref[n] * exp(j*2*pi*v_d*n/fs)            |
# |       This frequency-shifts the reference to each Doppler bin      |
# |                                                                     |
# |    2. WINDOWING                                                     |
# |       Apply Hann window to both ref and surv before shifting:      |
# |         x[n] = x[n] * w[n]   where w = 0.5(1 - cos(2*pi*n/(N-1)))|
# |       Reduces spectral leakage in the delay dimension              |
# |                                                                     |
# |    3. CROSS-CORRELATION VIA FFT                                    |
# |       R[T] = IFFT( FFT(x_surv) * conj(FFT(r_shifted)) )           |
# |       Surveillance is the echo signal; reference is Doppler-shifted.|
# |       Complexity: O(N log N) vs O(N^2) for direct computation      |
# |                                                                     |
# |    4. COMPLEX OUTPUT                                                |
# |       CAF[d, :] = R[T]  for T = 0 ... num_delay_bins - 1          |
# |       Complex values preserve phase for downstream DOA processing  |
# |                                                                     |
# |  OUTPUT SHAPE:                                                      |
# |    Matrix:  (num_doppler_bins, num_delay_bins)  - complex64        |
# |    Flattened to 1D vector for GNU Radio stream transport           |
# |    Row d = Doppler bin d, Col l = delay bin l                      |
# |                                                                     |
# |  DOPPLER AXIS:                                                      |
# |    Bins are linearly spaced from -max_doppler to +max_doppler Hz   |
# |    Zero-Doppler is at the centre row - no fftshift needed          |
# |                                                                     |
# |  DELAY AXIS:                                                        |
# |    Bin l corresponds to delay T = l / fs seconds                   |
# |    Max unambiguous range = num_delay_bins * c / (2 * fs)           |
# +---------------------------------------------------------------------+
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
import gnuradio.gr as gr


class caf(gr.sync_block):
    """
    Cross Ambiguity Function (CAF) -- gr-PR

    Computes the CAF between a reference and surveillance signal for one CPI.
    Both inputs are complex64 vectors of length cpi_samp, produced by
    upstream stream_to_vector blocks.

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
            name="PR CAF",
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

        # -- Pre-compute Hann window -----------------------------------------
        # Applied to both channels before cross-correlation.
        # Reduces range sidelobes caused by signal discontinuities at
        # CPI boundaries (Gibbs phenomenon).
        self._window = np.hanning(cpi_samp).astype(np.float32)

        # -- Pre-compute Doppler steering matrix -----------------------------
        # Shape: (num_doppler_bins, cpi_samp)
        # Each row d: exp(j*2*pi*v_d*t)
        # Applied to the REFERENCE signal per textbook formulation.
        doppler_freqs = np.linspace(-max_doppler, max_doppler, num_doppler_bins)
        t = np.arange(cpi_samp, dtype=np.float32) / sample_rate
        self._doppler_tones = np.exp(
            1j * 2.0 * np.pi
            * doppler_freqs[:, np.newaxis]   # (nd, 1)
            * t[np.newaxis, :]               # (1,  N)
        ).astype(np.complex64)              # -> (nd, N)

        # -- Pre-allocate output matrix --------------------------------------
        # Complex output preserves phase for downstream DOA processing.
        self._caf_mat = np.zeros((num_doppler_bins, num_delay_bins),
                                 dtype=np.complex64)

    # -------------------------------------------------------------------------
    def work(self, input_items, output_items):
        """
        GNU Radio work() -- called by the scheduler with one or more CPI vectors.

        input_items[0] shape: (n_cpis, cpi_samp)  -- reference
        input_items[1] shape: (n_cpis, cpi_samp)  -- surveillance
        output_items[0] shape: (n_cpis, nd * nl)  -- flattened CAF (complex)
        """
        ref_vecs  = input_items[0]
        surv_vecs = input_items[1]
        out       = output_items[0]
        n_cpis    = len(ref_vecs)
        produced  = 0

        for i in range(n_cpis):
            # -- CPI decimation ----------------------------------------------
            # If cpi_skip > 1, silently drop CPIs to reduce CPU load.
            # The scheduler still consumes all input vectors.
            self._skip_ctr += 1
            if self._skip_ctr < self._skip:
                continue
            self._skip_ctr = 0

            # -- Apply window ------------------------------------------------
            ref_w  = ref_vecs[i]  * self._window
            surv_w = surv_vecs[i] * self._window

            # -- Compute CAF -------------------------------------------------
            self._compute_caf(ref_w, surv_w)

            # -- Flatten and write to output port ----------------------------
            # Doppler axis is already centred (linspace from -max to +max),
            # so no fftshift is needed.
            out[produced] = self._caf_mat.ravel()
            produced += 1

        return produced

    # -------------------------------------------------------------------------
    def _compute_caf(self, ref_w, surv_w):
        """
        Core CAF computation (vectorised). Writes result into self._caf_mat.

        Doppler tones are applied to the REFERENCE signal (x_ref), matching
        the passive radar textbook formulation (Fig. 4.2):

          R_shifted = ref_w[newaxis, :] * doppler_tones        (nd, N)
          SURV      = FFT(surv_w)                               (N,)
          CAF[d,:]  = IFFT( SURV * conj(FFT(R_shifted[d,:])) ) truncated to nl
        """
        # Vectorised Doppler shift on REFERENCE: (1, N) * (nd, N) -> (nd, N)
        R_shifted = ref_w[np.newaxis, :] * self._doppler_tones

        # FFT of surveillance (echo) signal - computed once for all Doppler bins
        SURV = np.fft.fft(surv_w)                               # (N,) complex64

        # Batched FFT cross-correlation: IFFT( FFT(surv) * conj(FFT(ref_shifted)) )
        R_all = np.fft.ifft(
            SURV[np.newaxis, :] * np.conj(np.fft.fft(R_shifted, axis=1)),
            axis=1,
        )                                                        # (nd, N) complex

        # Keep first num_delay_bins lags (positive delays only), preserve complex
        self._caf_mat[:] = R_all[:, :self._nl].astype(np.complex64)

    # -------------------------------------------------------------------------
    # Runtime-adjustable parameters (wirable from GRC variable blocks)
    def set_cpi_skip(self, val):
        self._skip = max(1, int(val))

    def get_cpi_samp(self):        return self._N
    def get_num_doppler_bins(self): return self._nd
    def get_num_delay_bins(self):   return self._nl
    def get_sample_rate(self):      return self._fs
    def get_max_doppler(self):      return self._max_doppler
