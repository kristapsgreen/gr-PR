#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  caf.py
# Cross Ambiguity Function block
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALGORITHM: FFT-based Cross Ambiguity Function (CAF)               │
# │                                                                     │
# │  The CAF measures the similarity between two signals as a function  │
# │  of both time delay (τ) and Doppler frequency shift (ν):           │
# │                                                                     │
# │    CAF(τ, ν) = ∫ x_ref(t) · x_surv*(t − τ) · e^(j2πνt) dt       │
# │                                                                     │
# │  In passive radar:                                                  │
# │    - x_ref  = reference channel  (direct path from transmitter)    │
# │    - x_surv = surveillance channel (reflections off targets)       │
# │    - A peak at (τ₀, ν₀) means a target at range r = c·τ₀/2        │
# │      moving with radial velocity v = c·ν₀ / (2·f_c)               │
# │                                                                     │
# │  DISCRETE IMPLEMENTATION (Batch FFT method):                       │
# │                                                                     │
# │  For each Doppler bin d with frequency shift ν_d:                  │
# │                                                                     │
# │    1. DOPPLER SHIFT                                                 │
# │       Multiply surveillance by a complex tone at ν_d:              │
# │         s_shifted[n] = x_surv[n] · exp(j·2π·ν_d·n/fs)            │
# │       This "de-rotates" the signal to baseband for that Doppler    │
# │                                                                     │
# │    2. WINDOWING                                                     │
# │       Apply Hann window to both ref and shifted surv:              │
# │         x[n] = x[n] · w[n]   where w = 0.5(1 − cos(2πn/(N−1)))   │
# │       Reduces spectral leakage in the delay dimension              │
# │                                                                     │
# │    3. CROSS-CORRELATION VIA FFT                                    │
# │       R[τ] = IFFT( FFT(x_ref) · conj(FFT(s_shifted)) )           │
# │       This is the standard Wiener–Khinchin cross-correlation.      │
# │       Complexity: O(N log N) vs O(N²) for direct computation       │
# │                                                                     │
# │    4. MAGNITUDE SQUARED                                             │
# │       CAF[d, :] = |R[τ]|²  for τ = 0 … num_delay_bins − 1        │
# │                                                                     │
# │  OUTPUT SHAPE:                                                      │
# │    Matrix:  (num_doppler_bins, num_delay_bins)  — float32          │
# │    Flattened to 1D vector for GNU Radio stream transport           │
# │    Row d = Doppler bin d, Col l = delay bin l                      │
# │                                                                     │
# │  DOPPLER AXIS:                                                      │
# │    Bins are spaced at fs / num_doppler_bins Hz                     │
# │    Arranged via fftfreq: 0 … +fs/2, −fs/2 … 0                    │
# │    fftshift is applied before output so zero-Doppler is centred    │
# │                                                                     │
# │  DELAY AXIS:                                                        │
# │    Bin l corresponds to delay τ = l / fs seconds                   │
# │    Max unambiguous range = num_delay_bins · c / (2 · fs)           │
# └─────────────────────────────────────────────────────────────────────┘
#
# GNU Radio block type: sync_block (vector in → vector out, 1:1 CPI)
#
# Inputs
# ------
#   Port 0  ref   — complex64 vector, length = cpi_samp
#   Port 1  surv  — complex64 vector, length = cpi_samp
#
# Output
# ------
#   Port 0  caf_out — float32 vector, length = num_doppler_bins * num_delay_bins
#                     Row-major: row = Doppler bin, col = delay bin
#                     Doppler axis is fftshifted (zero-Doppler at centre row)
#

import numpy as np
import gnuradio.gr as gr


class caf(gr.sync_block):
    """
    Cross Ambiguity Function (CAF) — gr-PR

    Computes the CAF between a reference and surveillance signal for one CPI.
    Both inputs are complex64 vectors of length cpi_samp, produced by
    upstream stream_to_vector blocks.

    Parameters
    ----------
    cpi_samp         : int   — CPI length in samples (must match stream_to_vector)
    num_doppler_bins : int   — number of Doppler frequency bins
    num_delay_bins   : int   — number of delay (range) bins  ≤ cpi_samp
    cpi_skip         : int   — process 1 in every N CPIs (1 = process all)
    sample_rate      : float — receiver sample rate in Hz
    """

    def __init__(self,
                 cpi_samp=24000,
                 num_doppler_bins=128,
                 num_delay_bins=128,
                 cpi_skip=1,
                 sample_rate=2.4e6):

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
                np.dtype((np.float32, num_doppler_bins * num_delay_bins)),
            ]
        )

        self._N   = cpi_samp
        self._nd  = num_doppler_bins
        self._nl  = num_delay_bins
        self._fs  = sample_rate
        self._skip = max(1, int(cpi_skip))
        self._skip_ctr = 0

        # ── Pre-compute Hann window ──────────────────────────────────────
        # Applied to both channels before cross-correlation.
        # Reduces range sidelobes caused by signal discontinuities at
        # CPI boundaries (Gibbs phenomenon).
        self._window = np.hanning(cpi_samp).astype(np.float32)

        # ── Pre-compute Doppler steering matrix ─────────────────────────
        # Shape: (num_doppler_bins, cpi_samp)
        # Each row d: exp(j·2π·ν_d·t)  where ν_d = fftfreq[d] · fs
        #
        # Using fftfreq ensures the Doppler axis is symmetric around zero
        # after fftshift (applied later on the output matrix).
        max_doppler_hz = 1000.0  # ±1000 Hz covers ~300 m/s at 510 MHz
        doppler_freqs = np.linspace(-max_doppler_hz, max_doppler_hz, num_doppler_bins)
        t = np.arange(cpi_samp, dtype=np.float32) / sample_rate
        self._doppler_tones = np.exp(
            1j * 2.0 * np.pi
            * doppler_freqs[:, np.newaxis]   # (nd, 1)
            * t[np.newaxis, :]               # (1,  N)
        ).astype(np.complex64)              # → (nd, N)

        # ── Pre-allocate output matrix ───────────────────────────────────
        # Avoids heap allocation on every CPI.
        self._caf_mat = np.zeros((num_doppler_bins, num_delay_bins),
                                 dtype=np.float32)

    # ────────────────────────────────────────────────────────────────────
    def work(self, input_items, output_items):
        """
        GNU Radio work() — called by the scheduler with one or more CPI vectors.

        input_items[0] shape: (n_cpis, cpi_samp)  — reference
        input_items[1] shape: (n_cpis, cpi_samp)  — surveillance
        output_items[0] shape: (n_cpis, nd * nl)  — flattened CAF magnitude
        """
        ref_vecs  = input_items[0]
        surv_vecs = input_items[1]
        out       = output_items[0]
        n_cpis    = len(ref_vecs)
        produced  = 0

        for i in range(n_cpis):
            # ── CPI decimation ───────────────────────────────────────────
            # If cpi_skip > 1, silently drop CPIs to reduce CPU load.
            # The scheduler still consumes all input vectors.
            self._skip_ctr += 1
            if self._skip_ctr < self._skip:
                continue
            self._skip_ctr = 0

            # ── Apply window ─────────────────────────────────────────────
            ref_w  = ref_vecs[i]  * self._window
            surv_w = surv_vecs[i] * self._window

            # ── Compute CAF ──────────────────────────────────────────────
            self._compute_caf(ref_w, surv_w)

            # ── fftshift Doppler axis ────────────────────────────────────
            # Rearranges rows so zero-Doppler is at the centre row,
            # negative Doppler above, positive below — natural for display.
            shifted = np.fft.fftshift(self._caf_mat, axes=0)

            # ── Flatten and write to output port ─────────────────────────
            out[produced] = shifted.ravel()
            produced += 1

        return produced

    # ────────────────────────────────────────────────────────────────────
    def _compute_caf(self, ref_w, surv_w):
        """
        Core CAF computation. Writes result into self._caf_mat.

        Steps per Doppler bin d
        -----------------------
        1.  s_d = surv_w * tone_d          — Doppler shift surveillance
        2.  R   = IFFT(FFT(ref) * conj(FFT(s_d)))  — cross-correlation
        3.  caf_mat[d, :] = |R[0:nl]|²    — magnitude squared, take nl lags
        """
        # Pre-compute FFT of windowed reference (shared across all Doppler bins)
        REF = np.fft.fft(ref_w)            # (N,) complex64

        for d in range(self._nd):
            # Step 1: Doppler shift
            s_d = surv_w * self._doppler_tones[d]

            # Step 2: FFT cross-correlation
            #   was swapped  corrected - IFFT( FFT(ref) · FFT*(surv_shifted) )
            #   = circular cross-correlation in time domain
            R = np.fft.ifft(np.fft.fft(s_d) * np.conj(REF))

            # Step 3: |·|², keep first num_delay_bins lags (positive delays)
            self._caf_mat[d, :] = (np.abs(R[:self._nl]) ** 2).astype(np.float32)

    # ────────────────────────────────────────────────────────────────────
    # Runtime-adjustable parameters (wirable from GRC variable blocks)
    def set_cpi_skip(self, val):
        self._skip = max(1, int(val))

    def get_cpi_samp(self):        return self._N
    def get_num_doppler_bins(self): return self._nd
    def get_num_delay_bins(self):   return self._nl
    def get_sample_rate(self):      return self._fs