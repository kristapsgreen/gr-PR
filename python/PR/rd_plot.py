#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  rd_plot.py
# Range-Doppler Qt Display Sink
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALGORITHM: Range-Doppler Map Rendering                            │
# │                                                                     │
# │  Receives the flattened CAF magnitude matrix from the caf block    │
# │  and renders it as a 2D intensity plot (Range-Doppler map).        │
# │                                                                     │
# │  AXES INTERPRETATION:                                               │
# │                                                                     │
# │    X-axis — Bistatic Range / Delay                                  │
# │      Each delay bin l corresponds to:                              │
# │        τ_l = l / fs   (seconds)                                    │
# │        r_l = c · τ_l / 2   (metres, bistatic range)               │
# │      Displayed in µs for compactness.                               │
# │                                                                     │
# │    Y-axis — Doppler Frequency                                      │
# │      CAF output rows are ordered via linspace:                     │
# │        row 0 = −max_doppler, centre row = 0, last = +max_doppler  │
# │      Displayed in Hz.                                               │
# │                                                                     │
# │  NORMALISATION:                                                     │
# │    1. Convert to dB:  P_dB = 10·log10(|CAF|² + ε)                │
# │    2. Normalise to peak: P_norm = (P_dB − (peak − dyn_range)) /   │
# │                                    dyn_range                        │
# │    3. Clip to [0, 1] for colourmap mapping                         │
# │    This preserves relative structure while suppressing noise floor. │
# │                                                                     │
# │  THREADING MODEL:                                                   │
# │    GNU Radio scheduler runs work() in a dedicated C++ thread.      │
# │    Qt widgets must only be updated from the Qt main (GUI) thread.  │
# │    Bridge: pyqtSignal with Qt.QueuedConnection                     │
# │      GR thread  →  emit(matrix)  →  Qt event queue                │
# │      Qt thread  ←  slot fires    ←  event loop picks it up        │
# │    The numpy array is copied before emit to prevent the GR         │
# │    scheduler from overwriting it before Qt reads it.               │
# │                                                                     │
# │  EMBEDDING:                                                         │
# │    GRC calls pyqwidget() to retrieve the Qt widget and embeds it   │
# │    in the main window when gui_hint is set.  The block must NOT    │
# │    call .show() itself — GRC manages widget visibility.            │
# └─────────────────────────────────────────────────────────────────────┘
#
# GNU Radio block type: sync_block (vector in, no output — sink)
#
# Input
# -----
#   Port 0  caf_in — float32 vector, length = num_doppler_bins * num_delay_bins
#                    Row-major, Doppler axis ordered −max to +max (linspace)
#

import time
import numpy as np
import gnuradio.gr as gr

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QObject

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────────────────────────
# Signal bridge: safely passes numpy frames from GR thread → Qt GUI thread
# ─────────────────────────────────────────────────────────────────────────────
class _Bridge(QObject):
    new_frame = pyqtSignal(np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib canvas — the actual Range-Doppler plot
# ─────────────────────────────────────────────────────────────────────────────
class _RDCanvas(FigureCanvas):

    def __init__(self, num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, parent=None):

        self._nd        = num_doppler_bins
        self._nl        = num_delay_bins
        self._fs        = sample_rate
        self._max_dop   = max_doppler
        self._dyn_db    = dynamic_range_db

        # ── Figure setup ─────────────────────────────────────────────────
        fig = Figure(figsize=(8, 5), facecolor='#0a0a0a')
        super().__init__(fig)
        self.setParent(parent)
        fig.subplots_adjust(left=0.11, right=0.93, top=0.91, bottom=0.13)

        # ── Axis extents ──────────────────────────────────────────────────
        # X: delay in µs  (0 → max_delay_µs)
        # Y: Doppler in Hz (−max_doppler → +max_doppler)
        delay_max_us  = (num_delay_bins / sample_rate) * 1e6
        self._extent = [0, delay_max_us, -max_doppler, max_doppler]

        # ── Initial blank image ───────────────────────────────────────────
        dummy = np.zeros((num_doppler_bins, num_delay_bins), dtype=np.float32)
        ax = fig.add_subplot(111)
        self._ax = ax
        self._im = ax.imshow(
            dummy,
            aspect='auto',
            origin='lower',
            extent=self._extent,
            cmap=colormap,
            vmin=0.0, vmax=1.0,
            interpolation='nearest',
        )

        # ── Colourbar ─────────────────────────────────────────────────────
        cbar = fig.colorbar(self._im, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label('Norm. Power (dB)', color='#cccccc', fontsize=9)
        cbar_ticks = np.linspace(0, 1, 6)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(
            [f'{v:.0f}' for v in np.linspace(-dynamic_range_db, 0, 6)]
        )
        for lbl in cbar.ax.yaxis.get_ticklabels():
            lbl.set_color('#cccccc')
        cbar.ax.yaxis.set_tick_params(color='#666666')
        self._cbar = cbar

        # ── Axes styling ──────────────────────────────────────────────────
        ax.set_facecolor('#0a0a0a')
        ax.set_xlabel('Delay  (µs)', color='#cccccc', fontsize=10)
        ax.set_ylabel('Doppler  (Hz)', color='#cccccc', fontsize=10)
        ax.set_title('Range–Doppler Map', color='#ffffff',
                     fontsize=12, fontweight='bold')
        ax.tick_params(colors='#888888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

        # Zero-Doppler reference line
        ax.axhline(0.0, color='#00e5ff', linewidth=0.7,
                   linestyle='--', alpha=0.6)

        # ── Performance metrics ──────────────────────────────────────────
        self._last_time    = None
        self._frame_dt_ms  = 0.0

        self.draw()

    # ── Called only from Qt GUI thread via QueuedConnection ──────────────
    def update_frame(self, matrix: np.ndarray):
        """
        Normalise and render one CAF magnitude matrix.

        matrix shape: (num_doppler_bins, num_delay_bins), dtype float32
        Values are |CAF|² (linear power).
        """
        # ── Timing ───────────────────────────────────────────────────────
        now = time.perf_counter()
        if self._last_time is not None:
            self._frame_dt_ms = (now - self._last_time) * 1e3
        self._last_time = now

        eps = 1e-30
        # Convert to dB
        mat_db = 10.0 * np.log10(matrix + eps)
        peak   = mat_db.max()

        # Normalise: map [peak−dyn_db … peak] → [0 … 1]
        mat_norm = np.clip(
            (mat_db - (peak - self._dyn_db)) / self._dyn_db,
            0.0, 1.0
        ).astype(np.float32)

        self._im.set_data(mat_norm)
        self._ax.set_title(
            f'Range–Doppler Map    peak = {peak:.1f} dB'
            f'    |    {self._frame_dt_ms:.1f} ms/frame',
            color='#ffffff', fontsize=12, fontweight='bold'
        )
        self.draw_idle()  # thread-safe redraw request

    # ── Runtime setters ───────────────────────────────────────────────────
    def set_dynamic_range(self, val):
        self._dyn_db = float(val)

    def set_colormap(self, name):
        self._im.set_cmap(name)
        self.draw_idle()

    def set_max_doppler(self, val):
        """Update the Doppler axis extent at runtime."""
        self._max_dop = float(val)
        self._extent[2] = -self._max_dop
        self._extent[3] =  self._max_dop
        self._im.set_extent(self._extent)
        self._ax.set_xlim(self._extent[0], self._extent[1])
        self._ax.set_ylim(self._extent[2], self._extent[3])
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Qt widget — canvas + runtime controls
# ─────────────────────────────────────────────────────────────────────────────
class _RDWidget(QtWidgets.QWidget):

    def __init__(self, num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget  { background: #111; color: #eee; }
            QLabel   { font-size: 10px; }
            QSlider::groove:horizontal { background:#333; height:4px; border-radius:2px; }
            QSlider::handle:horizontal { background:#00e5ff; width:12px; height:12px;
                                         margin:-4px 0; border-radius:6px; }
            QComboBox { background:#222; border:1px solid #444; padding:2px 6px; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Canvas
        self.canvas = _RDCanvas(num_doppler_bins, num_delay_bins,
                                sample_rate, max_doppler,
                                dynamic_range_db, colormap,
                                parent=self)
        layout.addWidget(self.canvas, stretch=1)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(14)

        # Dynamic range
        ctrl.addWidget(QtWidgets.QLabel('Dyn. Range:'))
        self._dr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._dr_slider.setRange(10, 120)
        self._dr_slider.setValue(int(dynamic_range_db))
        self._dr_slider.setFixedWidth(150)
        self._dr_lbl = QtWidgets.QLabel(f'{int(dynamic_range_db)} dB')
        self._dr_slider.valueChanged.connect(
            lambda v: (self.canvas.set_dynamic_range(v),
                       self._dr_lbl.setText(f'{v} dB'))
        )
        ctrl.addWidget(self._dr_slider)
        ctrl.addWidget(self._dr_lbl)

        # Colourmap
        ctrl.addWidget(QtWidgets.QLabel('Colourmap:'))
        self._cmap = QtWidgets.QComboBox()
        self._cmap.addItems(
            ['inferno', 'plasma', 'viridis', 'magma', 'hot', 'jet', 'turbo', 'RdBu_r']
        )
        self._cmap.setCurrentText(colormap)
        self._cmap.currentTextChanged.connect(self.canvas.set_colormap)
        ctrl.addWidget(self._cmap)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.resize(880, 580)

    def set_dynamic_range(self, val):
        self._dr_slider.setValue(int(val))   # also fires the connected lambda

    def set_colormap(self, name):
        self._cmap.setCurrentText(name)

    def set_max_doppler(self, val):
        self.canvas.set_max_doppler(val)


# ─────────────────────────────────────────────────────────────────────────────
# GNU Radio sink block
# ─────────────────────────────────────────────────────────────────────────────
class rd_plot(gr.sync_block):
    """
    Range-Doppler Qt Display Sink  —  gr-PR

    Receives the flattened float32 CAF magnitude output from the caf block
    and renders a live Range-Doppler map in a Qt5 window.

    Parameters
    ----------
    num_doppler_bins : int    — must match caf block
    num_delay_bins   : int    — must match caf block
    sample_rate      : float  — receiver sample rate in Hz
    max_doppler      : float  — ±Doppler extent in Hz (must match caf block)
    dynamic_range_db : float  — colour scale depth in dB (default 60)
    colormap         : str    — initial matplotlib colormap name
    """

    def __init__(self,
                 num_doppler_bins=128,
                 num_delay_bins=128,
                 sample_rate=2.4e6,
                 max_doppler=1000.0,
                 dynamic_range_db=60.0,
                 colormap='inferno'):

        gr.sync_block.__init__(
            self,
            name="PR RD Plot",
            in_sig=[
                np.dtype((np.float32, num_doppler_bins * num_delay_bins)),
            ],
            out_sig=[]   # sink — no output
        )

        self._nd = num_doppler_bins
        self._nl = num_delay_bins

        # ── Thread bridge ─────────────────────────────────────────────────
        self._bridge = _Bridge()

        # ── Qt widget ─────────────────────────────────────────────────────
        # Do NOT call .show() here — GRC manages widget visibility.
        # When gui_hint is set, GRC calls pyqwidget() and embeds this
        # widget in the main window's grid/tab layout.
        # When gui_hint is empty, GRC shows it as a top-level window.
        self._win = _RDWidget(
            num_doppler_bins, num_delay_bins,
            sample_rate, max_doppler, dynamic_range_db, colormap
        )

        # QueuedConnection: signal crosses thread boundary via event queue.
        self._bridge.new_frame.connect(
            self._win.canvas.update_frame,
            QtCore.Qt.QueuedConnection
        )

    # ── GNU Radio work() ─────────────────────────────────────────────────
    def work(self, input_items, output_items):
        """
        Runs in the GR scheduler thread.
        Emits each frame to the Qt thread via the bridge signal.
        .copy() is essential — prevents the scheduler from reusing the
        buffer before Qt has rendered it.
        """
        frames = input_items[0]   # shape: (n, nd * nl)
        for flat in frames:
            matrix = flat.reshape(self._nd, self._nl).copy()
            self._bridge.new_frame.emit(matrix)
        return len(frames)

    # ── GRC integration ───────────────────────────────────────────────────
    def pyqwidget(self):
        """
        Returns the Qt widget so GRC can embed it in the flow graph window.
        GRC calls this automatically when gui_hint is set.
        """
        return self._win

    # ── Runtime callbacks (wirable from GRC variable/QT Range blocks) ────
    def set_dynamic_range(self, val):
        self._win.set_dynamic_range(val)

    def set_colormap(self, name):
        self._win.set_colormap(name)

    def set_max_doppler(self, val):
        self._win.set_max_doppler(val)
