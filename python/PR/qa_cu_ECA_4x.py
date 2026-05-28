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
    from gnuradio.PR import cu_ECA_4x as _cu_ECA_4x_mod
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


class qa_cu_ECA_4x(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        """Instantiate cu_ECA_4x and verify it constructs without error."""
        if not _CUPY_AVAILABLE:
            self.skipTest("CuPy not installed — skipping GPU block test")

        try:
            instance = _cu_ECA_4x_mod(
                cpi_samp=1024,
                num_taps=32,
                reg_mu=0.01,
                release_mem_each_cpi=True,
                mem_pool_limit_mb=0,
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
            instance = _cu_ECA_4x_mod(
                cpi_samp=2048,
                num_taps=64,
                reg_mu=0.05,
                release_mem_each_cpi=False,
                mem_pool_limit_mb=0,
            )
        except RuntimeError as exc:
            self.skipTest(f"GPU not available: {exc}")

        self.assertEqual(instance.get_cpi_samp(), 2048)
        self.assertEqual(instance.get_num_taps(),   64)
        self.assertAlmostEqual(instance.get_reg_mu(), 0.05)

    def test_setters(self):
        """Verify runtime setters update internal state correctly."""
        if not _CUPY_AVAILABLE:
            self.skipTest("CuPy not installed — skipping GPU block test")

        try:
            instance = _cu_ECA_4x_mod(
                cpi_samp=1024,
                num_taps=32,
                reg_mu=0.01,
            )
        except RuntimeError as exc:
            self.skipTest(f"GPU not available: {exc}")

        instance.set_num_taps(128)
        self.assertEqual(instance.get_num_taps(), 128)

        instance.set_reg_mu(0.5)
        self.assertAlmostEqual(instance.get_reg_mu(), 0.5)

        # set_num_taps should clamp silently — value beyond cpi_samp is ignored
        instance.set_num_taps(99999)
        self.assertEqual(instance.get_num_taps(), 128)  # unchanged


if __name__ == '__main__':
    gr_unittest.run(qa_cu_ECA_4x)
