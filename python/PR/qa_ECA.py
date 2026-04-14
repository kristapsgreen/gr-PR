#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 none.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio.PR import ECA


class qa_ECA(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        """Block instantiates with default parameters."""
        instance = ECA()
        self.assertEqual(instance.get_cpi_samp(), 24000)
        self.assertEqual(instance.get_num_taps(), 128)

    def test_direct_path_cancellation(self):
        """
        Synthetic test: surveillance = scaled + delayed copy of reference + weak target.
        After ECA the direct-path energy should be heavily suppressed while
        the target component partially survives.
        """
        N = 2048
        K = 64
        rng = np.random.default_rng(42)

        # Reference: random complex noise (simulates DVB-T-like signal)
        ref = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)

        # Direct path: ref scaled by 1.0 (strong) + delayed copy at tap 10 scaled 0.5
        direct = 1.0 * ref + 0.5 * np.roll(ref, 10)

        # Target echo: weak, Doppler-shifted (should partially survive cancellation)
        fd = 50.0  # Hz Doppler
        fs = 2.4e6
        t = np.arange(N, dtype=np.float32) / fs
        target = 0.01 * np.roll(ref, 200) * np.exp(1j * 2 * np.pi * fd * t)

        surv = (direct + target).astype(np.complex64)

        # ── Run the block ──────────────────────────────────────────────
        src_ref  = blocks.vector_source_c(ref.tolist(),  vlen=N)
        src_surv = blocks.vector_source_c(surv.tolist(), vlen=N)
        eca      = ECA(cpi_samp=N, num_taps=K, reg_mu=0.01)
        snk_ref  = blocks.vector_sink_c(vlen=N)
        snk_surv = blocks.vector_sink_c(vlen=N)

        self.tb.connect(src_ref,  (eca, 0))
        self.tb.connect(src_surv, (eca, 1))
        self.tb.connect((eca, 0), snk_ref)
        self.tb.connect((eca, 1), snk_surv)
        self.tb.run()

        # ── Check results ──────────────────────────────────────────────
        out_ref  = np.array(snk_ref.data(),  dtype=np.complex64)
        out_surv = np.array(snk_surv.data(), dtype=np.complex64)

        # Reference should pass through unchanged
        np.testing.assert_array_almost_equal(out_ref, ref, decimal=5)

        # Cleaned surveillance should have much less power than input
        power_in    = np.mean(np.abs(surv) ** 2)
        power_clean = np.mean(np.abs(out_surv) ** 2)
        suppression_dB = 10 * np.log10(power_clean / power_in)

        # We expect at least 20 dB suppression of the direct path
        self.assertLess(suppression_dB, -20.0,
                        f"Direct-path suppression only {suppression_dB:.1f} dB, expected < -20 dB")


if __name__ == '__main__':
    gr_unittest.run(qa_ECA)
