#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 none.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest

# CuPy / GPU may not be available in the CI environment.
# Skip the whole module cleanly rather than failing with an ImportError.
try:
    from gnuradio.PR import cu_caf_4x as _cu_caf_4x_mod
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


class qa_cu_caf_4x(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        """Instantiate cu_caf_4x and verify it constructs without error."""
        if not _CUPY_AVAILABLE:
            self.skipTest("CuPy not installed — skipping GPU block test")

        try:
            instance = _cu_caf_4x_mod(
                cpi_samp=1024,
                num_doppler_bins=32,
                num_delay_bins=32,
                cpi_skip=1,
                sample_rate=2.4e6,
                max_doppler=500.0,
            )
        except RuntimeError as exc:
            # No NVIDIA GPU present in this environment
            self.skipTest(f"GPU not available: {exc}")

        self.assertIsNotNone(instance)

    def test_getters(self):
        """Verify runtime getters return the values passed to the constructor."""
        if not _CUPY_AVAILABLE:
            self.skipTest("CuPy not installed — skipping GPU block test")

        try:
            instance = _cu_caf_4x_mod(
                cpi_samp=2048,
                num_doppler_bins=64,
                num_delay_bins=48,
                cpi_skip=2,
                sample_rate=1.0e6,
                max_doppler=750.0,
            )
        except RuntimeError as exc:
            self.skipTest(f"GPU not available: {exc}")

        self.assertEqual(instance.get_cpi_samp(),         2048)
        self.assertEqual(instance.get_num_doppler_bins(),   64)
        self.assertEqual(instance.get_num_delay_bins(),     48)
        self.assertAlmostEqual(instance.get_sample_rate(), 1.0e6)
        self.assertAlmostEqual(instance.get_max_doppler(), 750.0)


if __name__ == '__main__':
    gr_unittest.run(qa_cu_caf_4x)
