#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  rd_plot.py
# Range-Doppler Qt Display Sink
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALGORITHM: Range-Doppler Map Rendering                            │
# │                                                                     │
# │  Receives the flattened complex CAF matrix from the caf block      │
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
# │    call .show() itself when gui_hint is set.                       │
# └─────────────────────────────────────────────────────────────────────┘
#
# GNU Radio block type: sync_block (vector in, no output — sink)
#
# Input
# -----
#   Port 0  caf_in — complex64 vector, length = num_doppler_bins * num_delay_bins
#                    Row-major, Doppler axis ordered −max to +max (linspace)
#                    Display uses magnitude only; phase available for DOA
#

import csv
import os
import time
from datetime import datetime

import numpy as np
import gnuradio.gr as gr

# Speed of light (m/s) — used for bistatic range/velocity axis conversion.
_C = 299_792_458.0

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
    # Payload is a dict: {matrix, batch_size, pos_in_batch, batch_index,
    #                     batch_dt_ms, work_batch_time_ms}
    new_frame = pyqtSignal(object)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib canvas — the actual Range-Doppler plot
# ─────────────────────────────────────────────────────────────────────────────
class _RDCanvas(FigureCanvas):

    def __init__(self, num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, baseline_m=0.0, carrier_freq_hz=510e6,
                 parent=None):

        self._nd        = num_doppler_bins
        self._nl        = num_delay_bins
        self._fs        = sample_rate
        self._max_dop   = max_doppler
        self._dyn_db    = dynamic_range_db
        self._baseline  = float(baseline_m)
        self._fc        = float(carrier_freq_hz)

        # ── Figure setup ─────────────────────────────────────────────────
        fig = Figure(figsize=(8, 5), facecolor='#0a0a0a')
        super().__init__(fig)
        self.setParent(parent)
        fig.subplots_adjust(left=0.11, right=0.93, top=0.91, bottom=0.13)

        # ── Axis extents (bistatic geometry) ─────────────────────────────
        # X axis — Bistatic Range (m)
        #   τ = l / fs  (seconds, delay of echo vs direct path)
        #   R_b = L + c·τ   (total bistatic path: Tx→target→Rx)
        # Y axis — Bistatic Velocity (m/s)
        #   f_d measured in Hz; bistatic range-rate: |v_b| = f_d · c / f_c
        range_max_m = self._baseline + _C * num_delay_bins / sample_rate
        v_max_ms    = max_doppler * _C / self._fc
        self._extent = [self._baseline, range_max_m, -v_max_ms, v_max_ms]

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
        ax.set_xlabel('Bistatic Range  (m)', color='#cccccc', fontsize=10)
        ax.set_ylabel('Bistatic Velocity  (m/s)', color='#cccccc', fontsize=10)
        ax.set_title('Range–Doppler Map', color='#ffffff',
                     fontsize=12, fontweight='bold')
        ax.tick_params(colors='#888888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

        # Zero-velocity reference line
        ax.axhline(0.0, color='#00e5ff', linewidth=0.7,
                   linestyle='--', alpha=0.6)

        # ── Performance metrics ──────────────────────────────────────────
        self._last_time      = None
        self._frame_dt_ms    = 0.0
        self._render_time_ms = 0.0
        self._frame_num      = 0
        # Batch metadata — last values received from the GR thread.
        # Same value repeats across all rows of the same work() call.
        self._batch_size         = 0
        self._pos_in_batch       = 0
        self._batch_index        = 0
        self._batch_dt_ms        = 0.0
        self._work_batch_time_ms = 0.0

        # ── CSV recording state ──────────────────────────────────────────
        # _csv_path is None when not recording. When set, update_frame
        # appends one row per frame (open → write → close each time for
        # crash safety). _metrics_cb is a callable returning a dict of
        # pc_* metrics from the parent gr block; set via attach_block().
        self._csv_path    = None
        self._metrics_cb  = None

        self.draw()

    # ── Called only from Qt GUI thread via QueuedConnection ──────────────
    def update_frame(self, payload):
        """
        Normalise and render one CAF matrix.

        payload : dict emitted by the rd_plot block, carrying:
            matrix             : (nd, nl) complex64 CAF frame
            batch_size         : n_cpis passed to this work() call
            pos_in_batch       : 0..batch_size-1 — index within the batch
            batch_index        : monotonic counter of work() invocations
            batch_dt_ms        : time since the previous work() call's start
            work_batch_time_ms : total wall time inside that work() call
        """
        matrix = payload['matrix']
        self._batch_size         = payload.get('batch_size', 0)
        self._pos_in_batch       = payload.get('pos_in_batch', 0)
        self._batch_index        = payload.get('batch_index', 0)
        self._batch_dt_ms        = payload.get('batch_dt_ms', 0.0)
        self._work_batch_time_ms = payload.get('work_batch_time_ms', 0.0)

        # ── Timing: frame interval ───────────────────────────────────────
        render_start = time.perf_counter()
        if self._last_time is not None:
            self._frame_dt_ms = (render_start - self._last_time) * 1e3
        self._last_time = render_start

        eps = 1e-30
        # Extract magnitude squared from complex CAF, then convert to dB
        mag_sq = np.abs(matrix) ** 2
        mat_db = 10.0 * np.log10(mag_sq + eps)
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

        # ── Timing: render duration ──────────────────────────────────────
        self._render_time_ms = (time.perf_counter() - render_start) * 1e3
        self._frame_num += 1

        # ── CSV logging (crash-safe: open/write/close per row) ───────────
        if self._csv_path is not None:
            self._write_csv_row(peak)

    # ── Runtime setters ───────────────────────────────────────────────────
    def set_dynamic_range(self, val):
        self._dyn_db = float(val)

    def set_colormap(self, name):
        self._im.set_cmap(name)
        self.draw_idle()

    def set_max_doppler(self, val):
        """Update the velocity axis extent at runtime. Input is max Doppler (Hz)."""
        self._max_dop = float(val)
        self._rescale_axes()

    def set_baseline(self, val):
        """Update the baseline L (metres) — shifts the range axis origin."""
        self._baseline = float(val)
        self._rescale_axes()

    def set_carrier_freq(self, val):
        """Update the carrier frequency f_c (Hz) — rescales velocity axis."""
        self._fc = float(val)
        self._rescale_axes()

    def _rescale_axes(self):
        """Recompute X (range) and Y (velocity) extents from current params."""
        range_max_m = self._baseline + _C * self._nl / self._fs
        v_max_ms    = self._max_dop * _C / self._fc
        self._extent = [self._baseline, range_max_m, -v_max_ms, v_max_ms]
        self._im.set_extent(self._extent)
        self._ax.set_xlim(self._extent[0], self._extent[1])
        self._ax.set_ylim(self._extent[2], self._extent[3])
        self.draw_idle()

    # ── CSV recording ────────────────────────────────────────────────────
    # The GR scheduler writes frames very infrequently (a few Hz up to
    # maybe 30 Hz) so open/write/close per row is cheap and guarantees
    # the file is flushed to disk even if the flowgraph crashes or is
    # killed mid-run — matches user's requirement for crash-safe logging.
    _CSV_HEADER = [
        'timestamp_iso',
        'frame_num',
        'frame_dt_ms',       # interval between successive frames seen by Qt thread
        'render_time_ms',    # time spent inside update_frame (normalise + redraw)
        'peak_db',           # peak of |CAF|² in dB
        # ── Batch-level metrics (from the GR scheduler's work() call) ────
        # When the upstream block is slow, the scheduler gives work() multiple
        # frames at once. Same batch_* values repeat for all rows of one batch,
        # so you can tell batched from unbatched delivery and compute the
        # true arrival period (batch_dt_ms) independent of intra-batch bursts.
        'batch_index',       # monotonic work() call counter
        'batch_size',        # n_cpis in the current work() call
        'pos_in_batch',      # 0 .. batch_size-1
        'batch_dt_ms',       # time since the previous work() call's start
        'work_batch_time_ms',# total wall time inside the emitting work() call
        # ── GR performance counters ──────────────────────────────────────
        'pc_work_time_avg_us',      # GR pc_* — requires ENABLE_PERFORMANCE_COUNTERS
        'pc_input_buf_full_avg',    # avg fullness of caf_in buffer (0..1)
        'pc_throughput_avg',        # items/sec
        'nitems_read',              # cumulative items consumed on port 0
    ]

    def attach_block(self, metrics_cb):
        """Register a callback returning {pc_work_time_avg_us, ...} dict.
        Called by the parent gr block to plumb pc_* counters through."""
        self._metrics_cb = metrics_cb

    def start_recording(self, path):
        """Create the CSV file, write the header, and begin logging."""
        # Ensure parent dir exists
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(self._CSV_HEADER)
        self._csv_path  = path
        self._frame_num = 0  # reset so rows are 1, 2, 3, ... per run

    def stop_recording(self):
        self._csv_path = None

    def is_recording(self):
        return self._csv_path is not None

    def _write_csv_row(self, peak_db):
        """Append one row for the most recent frame. Called from update_frame."""
        if self._metrics_cb is not None:
            try:
                m = self._metrics_cb()
            except Exception:
                m = {}
        else:
            m = {}
        row = [
            datetime.now().isoformat(timespec='milliseconds'),
            self._frame_num,
            f'{self._frame_dt_ms:.3f}',
            f'{self._render_time_ms:.3f}',
            f'{peak_db:.2f}',
            self._batch_index,
            self._batch_size,
            self._pos_in_batch,
            f'{self._batch_dt_ms:.3f}',
            f'{self._work_batch_time_ms:.3f}',
            m.get('pc_work_time_avg_us', ''),
            m.get('pc_input_buf_full_avg', ''),
            m.get('pc_throughput_avg', ''),
            m.get('nitems_read', ''),
        ]
        try:
            with open(self._csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            # Disk full, permission error, etc. — stop recording rather
            # than spamming exceptions into the Qt event loop.
            print(f'[rd_plot] CSV write failed, stopping recording: {e}')
            self._csv_path = None


# ─────────────────────────────────────────────────────────────────────────────
# Qt widget — canvas + runtime controls
# ─────────────────────────────────────────────────────────────────────────────
class _RDWidget(QtWidgets.QWidget):

    def __init__(self, num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, baseline_m=0.0, carrier_freq_hz=510e6,
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget  { background: #111; color: #eee; }
            QLabel   { font-size: 10px; }
            QSlider::groove:horizontal { background:#333; height:4px; border-radius:2px; }
            QSlider::handle:horizontal { background:#00e5ff; width:12px; height:12px;
                                         margin:-4px 0; border-radius:6px; }
            QComboBox { background:#222; border:1px solid #444; padding:2px 6px; }
            QPushButton { background:#222; border:1px solid #444; padding:3px 10px; }
            QPushButton:hover { background:#2a2a2a; }
            QPushButton[recording="true"] { background:#8b1a1a; border:1px solid #c33;
                                            color:#fff; font-weight:bold; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Canvas
        self.canvas = _RDCanvas(num_doppler_bins, num_delay_bins,
                                sample_rate, max_doppler,
                                dynamic_range_db, colormap,
                                baseline_m=baseline_m,
                                carrier_freq_hz=carrier_freq_hz,
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

        # Record CSV toggle button — lives on the right of the control bar
        self._rec_btn = QtWidgets.QPushButton('● Record CSV')
        self._rec_btn.setCheckable(True)
        self._rec_btn.clicked.connect(self._on_record_toggled)
        ctrl.addWidget(self._rec_btn)

        layout.addLayout(ctrl)

        self.resize(880, 580)

    def set_dynamic_range(self, val):
        self._dr_slider.setValue(int(val))   # also fires the connected lambda

    def set_colormap(self, name):
        self._cmap.setCurrentText(name)

    def set_max_doppler(self, val):
        self.canvas.set_max_doppler(val)

    def set_baseline(self, val):
        self.canvas.set_baseline(val)

    def set_carrier_freq(self, val):
        self.canvas.set_carrier_freq(val)

    # ── Record button handler ────────────────────────────────────────────
    def _on_record_toggled(self, checked):
        """Toggle CSV recording. Prompts for path on start via file dialog."""
        if checked:
            # Default filename: rd_plot_YYYYMMDD_HHMMSS.csv in cwd
            default_name = 'rd_plot_{}.csv'.format(
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            default_path = os.path.join(os.getcwd(), default_name)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Record metrics to CSV', default_path,
                'CSV files (*.csv);;All files (*)'
            )
            if not path:
                # User cancelled — untick the button without recording
                self._rec_btn.setChecked(False)
                return
            try:
                self.canvas.start_recording(path)
            except OSError as e:
                QtWidgets.QMessageBox.warning(
                    self, 'Record failed', f'Could not create {path}:\n{e}'
                )
                self._rec_btn.setChecked(False)
                return
            self._rec_btn.setText('■ Stop  ({})'.format(os.path.basename(path)))
        else:
            self.canvas.stop_recording()
            self._rec_btn.setText('● Record CSV')

        # Re-apply stylesheet so the [recording="true"] selector kicks in
        self._rec_btn.setProperty('recording', 'true' if checked else 'false')
        self._rec_btn.style().unpolish(self._rec_btn)
        self._rec_btn.style().polish(self._rec_btn)


# ─────────────────────────────────────────────────────────────────────────────
# GNU Radio sink block
# ─────────────────────────────────────────────────────────────────────────────
class rd_plot(gr.sync_block):
    """
    Range-Doppler Qt Display Sink  —  gr-PR

    Receives the flattened complex64 CAF output from the caf block
    and renders a live Range-Doppler map in a Qt5 window (magnitude only).

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
                 colormap='inferno',
                 baseline_m=0.0,
                 carrier_freq_hz=510e6):

        gr.sync_block.__init__(
            self,
            name="PR RD Plot",
            in_sig=[
                np.dtype((np.complex64, num_doppler_bins * num_delay_bins)),
            ],
            out_sig=[]   # sink — no output
        )

        self._nd = num_doppler_bins
        self._nl = num_delay_bins

        # ── Batch-tracking state (GR thread) ──────────────────────────────
        # These live on the block and get stamped into every payload dict
        # before emit, so the Qt thread can log them to CSV.
        self._batch_index         = 0
        self._last_work_start_ts  = None
        # work_batch_time_ms reports the PREVIOUS batch's duration because
        # we emit payloads mid-work() (before the current batch ends).
        self._prev_work_time_ms   = 0.0

        # ── Thread bridge ─────────────────────────────────────────────────
        self._bridge = _Bridge()

        # ── Qt widget ─────────────────────────────────────────────────────
        # When gui_hint is set, GRC calls pyqwidget() and re-parents this
        # widget into the main window's grid/tab layout.
        # When gui_hint is empty, it stays as a standalone top-level window.
        self._win = _RDWidget(
            num_doppler_bins, num_delay_bins,
            sample_rate, max_doppler, dynamic_range_db, colormap,
            baseline_m=baseline_m, carrier_freq_hz=carrier_freq_hz,
        )

        # QueuedConnection: signal crosses thread boundary via event queue.
        self._bridge.new_frame.connect(
            self._win.canvas.update_frame,
            QtCore.Qt.QueuedConnection
        )

        # Plumb GR performance counters to the canvas so the CSV logger
        # can record them. pc_* only returns non-zero values when GNU
        # Radio was built with -DENABLE_PERFORMANCE_COUNTERS=ON; when
        # that flag is off the methods still exist and return 0.0.
        self._win.canvas.attach_block(self._collect_metrics)

        # Show as standalone window. If gui_hint is set, GRC will
        # re-parent and embed this widget into the main window layout.
        self._win.show()

    # ── Performance counter collection ───────────────────────────────────
    def _collect_metrics(self):
        """Return a dict of GR pc_* counters for the CSV logger.
        Wrapped in try/except so unsupported counters (depending on GR
        build/version) just yield empty cells rather than crashing."""
        def _safe(fn, *args):
            try:
                return fn(*args)
            except Exception:
                return ''
        return {
            'pc_work_time_avg_us':   _safe(self.pc_work_time_avg),
            'pc_input_buf_full_avg': _safe(self.pc_input_buffers_full_avg, 0),
            'pc_throughput_avg':     _safe(self.pc_throughput_avg),
            'nitems_read':           _safe(self.nitems_read, 0),
        }

    # ── GNU Radio work() ─────────────────────────────────────────────────
    def work(self, input_items, output_items):
        """
        Runs in the GR scheduler thread.
        Emits each frame to the Qt thread via the bridge signal.
        .copy() is essential — prevents the scheduler from reusing the
        buffer before Qt has rendered it.

        Each emitted payload is a dict carrying the frame matrix plus
        batch metadata so the CSV logger on the Qt side can distinguish
        genuinely slow arrivals from bursty scheduler batching.
        """
        work_start = time.perf_counter()
        frames = input_items[0]   # shape: (n_cpis, nd * nl)
        n_cpis = len(frames)

        # Time between the START of consecutive work() calls. This is the
        # *true* arrival period: frames within one batch emit back-to-back
        # through the Qt queue, so frame_dt_ms on the Qt side can collapse
        # toward zero for intra-batch rows even when upstream is slow.
        if self._last_work_start_ts is not None:
            batch_dt_ms = (work_start - self._last_work_start_ts) * 1e3
        else:
            batch_dt_ms = 0.0
        self._last_work_start_ts = work_start

        self._batch_index += 1

        for i, flat in enumerate(frames):
            matrix = flat.reshape(self._nd, self._nl).copy()
            self._bridge.new_frame.emit({
                'matrix':             matrix,
                'batch_size':         n_cpis,
                'pos_in_batch':       i,
                'batch_index':        self._batch_index,
                'batch_dt_ms':        batch_dt_ms,
                # Previous batch's duration — current one isn't done yet.
                'work_batch_time_ms': self._prev_work_time_ms,
            })

        self._prev_work_time_ms = (time.perf_counter() - work_start) * 1e3
        return n_cpis

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

    def set_baseline(self, val):
        self._win.set_baseline(val)

    def set_carrier_freq(self, val):
        self._win.set_carrier_freq(val)
