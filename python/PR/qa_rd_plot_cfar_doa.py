#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  qa_rd_plot_cfar_doa.py
# Minimal instantiation test for rd_plot_cfar_doa.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest

try:
    from PyQt5 import QtWidgets
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

try:
    from gnuradio.PR import rd_plot_cfar_doa
    _HAS_BLOCK = True
except ImportError:
    _HAS_BLOCK = False


class qa_rd_plot_cfar_doa(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        """Verify the block can be instantiated with default parameters."""
        if not _HAS_QT:
            self.skipTest('PyQt5 not importable in this test environment')
        if not _HAS_BLOCK:
            self.skipTest('gnuradio.PR.rd_plot_cfar_doa not importable')

        # Ensure a QApplication exists (required by any Qt widget at construction)
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        instance = rd_plot_cfar_doa(
            num_doppler_bins=64,
            num_delay_bins=64,
            sample_rate=2.4e6,
            max_doppler=500.0,
            dynamic_range_db=60.0,
            colormap='inferno',
            baseline_m=1000.0,
            carrier_freq_hz=510e6,
            cfar_train_range=4,
            cfar_train_doppler=2,
            cfar_guard_range=1,
            cfar_guard_doppler=1,
            cfar_pfa=1e-3,
            max_targets=8,
            antenna_spacing_m=0.3,
            mvdr_loading=0.01,
            mvdr_angle_step=2.0,
            mvdr_nb_r=1,
            mvdr_nb_d=1,
            use_mvdr=True,
            radar_lat=51.5,
            radar_lon=-0.1,
            tx_lat=51.51,
            tx_lon=-0.09,
            boresight_az_deg=0.0,
            xy_extent_m=15000.0,
        )
        self.assertIsNotNone(instance)
        # Verify it exposes a Qt widget
        self.assertIsNotNone(instance.pyqwidget())

    def test_001_empty_flowgraph(self):
        """Empty flowgraph runs without error (no input connections)."""
        self.tb.run()


if __name__ == '__main__':
    gr_unittest.run(qa_rd_plot_cfar_doa)
