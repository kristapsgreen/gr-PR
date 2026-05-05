#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  rd_plot_cfar.py
# Range-Doppler Qt Display Sink with CA-CFAR detection
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALGORITHM: 2D Cell-Averaging CFAR                                 │
# │                                                                     │
# │  For each cell under test (CUT) in the |CAF|² map, estimate the    │
# │  local noise power from a ring of training cells surrounding it,   │
# │  excluding a guard band around the CUT to prevent target leakage:  │
# │                                                                     │
# │      ┌─────────────────────────────┐                               │
# │      │  T  T  T  T  T  T  T  T  T  │   T = training cell          │
# │      │  T  T  T  T  T  T  T  T  T  │   G = guard cell             │
# │      │  T  T  G  G  G  G  G  T  T  │   # = cell under test        │
# │      │  T  T  G  #  G  T  T  T  T  │                               │
# │      │  T  T  G  G  G  G  G  T  T  │                               │
# │      │  T  T  T  T  T  T  T  T  T  │                               │
# │      └─────────────────────────────┘                               │
# │                                                                     │
# │  Noise estimate:   P̂ = (1/N) · Σ training cells                  │
# │  Threshold:        T  = α · P̂                                     │
# │  Scaling factor:   α  = N · (Pfa^(−1/N) − 1)                      │
# │                       for exponentially distributed noise power    │
# │  Detection:        CUT > T                                         │
# │                                                                     │
# │  Efficient 2D sliding sums via numpy integral image (summed area   │
# │  table): O(N·M) setup, O(1) per cell. No scipy dependency.         │
# └─────────────────────────────────────────────────────────────────────┘

import collections
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
# CA-CFAR computation (pure numpy, integral-image based)
# ─────────────────────────────────────────────────────────────────────────────
def _ca_cfar_2d(power, train_r, train_d, guard_r, guard_d, pfa):
    """
    Run 2D Cell-Averaging CFAR on a power matrix.

    Parameters
    ----------
    power   : (nr, nd) float array — |CAF|² (linear power, NOT dB)
    train_r : half-window of training cells along range axis
    train_d : half-window of training cells along doppler axis
    guard_r : half-window of guard cells along range axis
    guard_d : half-window of guard cells along doppler axis
    pfa     : probability of false alarm (e.g. 1e-4)

    Returns
    -------
    detections : bool array (nr, nd) — True where power > threshold
    threshold  : float array (nr, nd) — per-cell threshold
    """
    nr, nd = power.shape

    # Integral image with zero-padded top/left: sum over [r0:r1, c0:c1]
    # = ii[r1, c1] − ii[r0, c1] − ii[r1, c0] + ii[r0, c0]
    ii = np.pad(
        np.cumsum(np.cumsum(power, axis=0), axis=1),
        ((1, 0), (1, 0))
    )

    # Big window = training + guard + CUT;  small window = guard + CUT
    br = train_r + guard_r
    bd = train_d + guard_d

    rr, cc = np.indices((nr, nd))

    # Big window bounds, clipped to map edges
    r0b = np.clip(rr - br, 0, nr); r1b = np.clip(rr + br + 1, 0, nr)
    c0b = np.clip(cc - bd, 0, nd); c1b = np.clip(cc + bd + 1, 0, nd)

    # Small (guard+CUT) window bounds
    r0s = np.clip(rr - guard_r, 0, nr); r1s = np.clip(rr + guard_r + 1, 0, nr)
    c0s = np.clip(cc - guard_d, 0, nd); c1s = np.clip(cc + guard_d + 1, 0, nd)

    sum_big = ii[r1b, c1b] - ii[r0b, c1b] - ii[r1b, c0b] + ii[r0b, c0b]
    sum_sm  = ii[r1s, c1s] - ii[r0s, c1s] - ii[r1s, c0s] + ii[r0s, c0s]

    n_train = (r1b - r0b) * (c1b - c0b) - (r1s - r0s) * (c1s - c0s)

    # Cells too close to the edge have no training cells → skip (no detect)
    valid = n_train > 0
    noise = np.zeros_like(power, dtype=np.float64)
    np.divide(sum_big - sum_sm, n_train, out=noise, where=valid)

    # Alpha derived from interior window size (edges use same alpha — slight
    # Pfa mismatch at edges is acceptable for a visualisation).
    n_t = max(1, int((2 * br + 1) * (2 * bd + 1) - (2 * guard_r + 1) * (2 * guard_d + 1)))
    alpha = n_t * (pfa ** (-1.0 / n_t) - 1.0)

    threshold = alpha * noise
    detections = np.logical_and(valid, power > threshold)
    return detections, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Signal bridge: GR thread → Qt GUI thread
# ─────────────────────────────────────────────────────────────────────────────
class _Bridge(QObject):
    # object payload: ndarray in 'single' mode, 4-tuple of ndarrays in 'sum4' mode
    new_frame = pyqtSignal(object)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib canvas — dual subplot: RD map + CFAR detections
# ─────────────────────────────────────────────────────────────────────────────
class _RDCfarCanvas(FigureCanvas):

    def __init__(self, num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, cfar_params,
                 baseline_m=0.0, carrier_freq_hz=510e6,
                 input_mode='single', cfar_trail_frames=0,
                 parent=None):

        self._nd         = num_doppler_bins
        self._nl         = num_delay_bins
        self._fs         = sample_rate
        self._max_dop    = max_doppler
        self._dyn_db     = dynamic_range_db
        self._baseline   = float(baseline_m)
        self._fc         = float(carrier_freq_hz)
        self._input_mode = str(input_mode)
        # cfar_params = dict(train_r, train_d, guard_r, guard_d, pfa)
        self._cfar       = dict(cfar_params)

        # Trail deque: rolling history of past CFAR bool arrays
        n_trail = max(0, int(cfar_trail_frames))
        self._trail = collections.deque(maxlen=n_trail) if n_trail > 0 else None

        # ── Figure: two side-by-side subplots ─────────────────────────────
        fig = Figure(figsize=(14, 5), facecolor='#0a0a0a')
        super().__init__(fig)
        self.setParent(parent)
        fig.subplots_adjust(left=0.06, right=0.97,
                            top=0.91, bottom=0.13, wspace=0.22)

        # Bistatic axes: X = Bistatic Range (m), Y = Bistatic Velocity (m/s)
        range_max_m  = self._baseline + _C * num_delay_bins / sample_rate
        v_max_ms     = max_doppler * _C / self._fc
        self._extent = [self._baseline, range_max_m, -v_max_ms, v_max_ms]
        dummy = np.zeros((num_doppler_bins, num_delay_bins), dtype=np.float32)

        # ── Left: dB RD map ───────────────────────────────────────────────
        ax_l = fig.add_subplot(121)
        self._ax_l = ax_l
        self._im_l = ax_l.imshow(
            dummy, aspect='auto', origin='lower', extent=self._extent,
            cmap=colormap, vmin=0.0, vmax=1.0, interpolation='nearest',
        )
        cbar_l = fig.colorbar(self._im_l, ax=ax_l, fraction=0.035, pad=0.02)
        cbar_l.set_label('Norm. Power (dB)', color='#cccccc', fontsize=9)
        cbar_l.set_ticks(np.linspace(0, 1, 6))
        cbar_l.set_ticklabels(
            [f'{v:.0f}' for v in np.linspace(-dynamic_range_db, 0, 6)]
        )
        for lbl in cbar_l.ax.yaxis.get_ticklabels():
            lbl.set_color('#cccccc')
        cbar_l.ax.yaxis.set_tick_params(color='#666666')
        self._cbar_l = cbar_l

        # ── Right: CFAR detection mask ────────────────────────────────────
        ax_r = fig.add_subplot(122)
        self._ax_r = ax_r
        self._im_r = ax_r.imshow(
            dummy, aspect='auto', origin='lower', extent=self._extent,
            cmap='Greys_r', vmin=0.0, vmax=1.0, interpolation='nearest',
        )

        # ── Shared styling ────────────────────────────────────────────────
        for ax, title in ((ax_l, f'Range–Doppler Map  [{input_mode}]'),
                          (ax_r, 'CA-CFAR Detections')):
            ax.set_facecolor('#0a0a0a')
            ax.set_xlabel('Bistatic Range  (m)', color='#cccccc', fontsize=10)
            ax.set_ylabel('Bistatic Velocity  (m/s)', color='#cccccc', fontsize=10)
            ax.set_title(title, color='#ffffff', fontsize=11, fontweight='bold')
            ax.tick_params(colors='#888888', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
            ax.axhline(0.0, color='#00e5ff', linewidth=0.7,
                       linestyle='--', alpha=0.6)

        # ── Performance metrics ──────────────────────────────────────────
        self._last_time      = None
        self._frame_dt_ms    = 0.0
        self._render_time_ms = 0.0
        self._cfar_time_ms   = 0.0
        self._frame_num      = 0
        self._n_detections   = 0

        # ── CSV recording ────────────────────────────────────────────────
        self._csv_path   = None
        self._metrics_cb = None

        self.draw()

    # ── Called only from Qt GUI thread via QueuedConnection ──────────────
    def update_frame(self, payload):
        """
        payload: single (nd,nl) complex array in 'single' mode,
                 or tuple (caf1,caf2,caf3,caf4) in 'sum4' mode.
        """
        render_start = time.perf_counter()
        if self._last_time is not None:
            self._frame_dt_ms = (render_start - self._last_time) * 1e3
        self._last_time = render_start

        # Coherent sum before |·|² — per input_mode
        if self._input_mode == 'sum4':
            caf1, caf2, caf3, caf4 = payload
            caf_combined = caf1 + caf2 + caf3 + caf4
        else:
            caf_combined = payload

        # |CAF|² in linear scale — CFAR operates on linear power
        mag_sq = (np.abs(caf_combined) ** 2).astype(np.float64)

        # dB for the left plot
        eps = 1e-30
        mat_db = 10.0 * np.log10(mag_sq + eps)
        peak   = mat_db.max()
        mat_norm = np.clip(
            (mat_db - (peak - self._dyn_db)) / self._dyn_db,
            0.0, 1.0,
        ).astype(np.float32)
        self._im_l.set_data(mat_norm)

        # CFAR detection map
        cfar_start = time.perf_counter()
        detections, _ = _ca_cfar_2d(
            mag_sq,
            self._cfar['train_r'], self._cfar['train_d'],
            self._cfar['guard_r'], self._cfar['guard_d'],
            self._cfar['pfa'],
        )
        self._cfar_time_ms = (time.perf_counter() - cfar_start) * 1e3
        self._n_detections = int(detections.sum())

        # Trail overlay: combine past frames with current detections
        if self._trail is not None and len(self._trail) > 0:
            overlay = np.zeros(detections.shape, dtype=np.float32)
            k = len(self._trail)
            for i, past in enumerate(self._trail):
                # Weight ramps from ~0.15 (oldest) to ~0.5 (most recent historical)
                w = 0.15 + 0.35 * (i / max(1, k - 1))
                np.maximum(overlay, past.astype(np.float32) * w, out=overlay)
            # Current detections always at 1.0 — clearly above trail cap of 0.5
            np.maximum(overlay, detections.astype(np.float32), out=overlay)
            self._im_r.set_data(overlay)
        else:
            self._im_r.set_data(detections.astype(np.float32))

        # Append current frame to trail AFTER rendering (don't self-trail)
        if self._trail is not None:
            self._trail.append(detections.copy())

        # Titles
        mode_tag = f'[{self._input_mode}]'
        self._ax_l.set_title(
            f'Range–Doppler Map  {mode_tag}    peak = {peak:.1f} dB',
            color='#ffffff', fontsize=11, fontweight='bold',
        )
        self._ax_r.set_title(
            f'CA-CFAR  (Pfa={self._cfar["pfa"]:.0e}, '
            f'{self._n_detections} det, {self._cfar_time_ms:.1f} ms)',
            color='#ffffff', fontsize=11, fontweight='bold',
        )

        self.draw_idle()

        self._render_time_ms = (time.perf_counter() - render_start) * 1e3
        self._frame_num += 1

        if self._csv_path is not None:
            self._write_csv_row(peak)

    # ── Runtime setters ───────────────────────────────────────────────────
    def set_dynamic_range(self, val):
        self._dyn_db = float(val)

    def set_colormap(self, name):
        self._im_l.set_cmap(name)
        self.draw_idle()

    def set_max_doppler(self, val):
        self._max_dop = float(val)
        self._rescale_axes()

    def set_baseline(self, val):
        self._baseline = float(val)
        self._rescale_axes()

    def set_carrier_freq(self, val):
        self._fc = float(val)
        self._rescale_axes()

    def _rescale_axes(self):
        """Recompute both subplots' X (range) and Y (velocity) extents."""
        range_max_m  = self._baseline + _C * self._nl / self._fs
        v_max_ms     = self._max_dop * _C / self._fc
        self._extent = [self._baseline, range_max_m, -v_max_ms, v_max_ms]
        for im, ax in ((self._im_l, self._ax_l), (self._im_r, self._ax_r)):
            im.set_extent(self._extent)
            ax.set_xlim(self._extent[0], self._extent[1])
            ax.set_ylim(self._extent[2], self._extent[3])
        self.draw_idle()

    def set_pfa(self, val):
        self._cfar['pfa'] = float(val)

    def set_train_cells(self, r, d):
        self._cfar['train_r'] = int(r)
        self._cfar['train_d'] = int(d)

    def set_guard_cells(self, r, d):
        self._cfar['guard_r'] = int(r)
        self._cfar['guard_d'] = int(d)

    def set_cfar_trail_frames(self, val):
        n = max(0, int(val))
        if n == 0:
            self._trail = None
        elif self._trail is None:
            self._trail = collections.deque(maxlen=n)
        else:
            # Preserve as much existing history as fits the new length
            old = list(self._trail)
            self._trail = collections.deque(old[-n:], maxlen=n)

    # ── CSV recording (crash-safe: open/write/close per row) ─────────────
    _CSV_HEADER = [
        'timestamp_iso', 'frame_num',
        'frame_dt_ms', 'render_time_ms', 'cfar_time_ms',
        'peak_db', 'n_detections',
        'pc_work_time_avg_us', 'pc_input_buf_full_avg',
        'pc_throughput_avg', 'nitems_read',
    ]

    def attach_block(self, metrics_cb):
        self._metrics_cb = metrics_cb

    def start_recording(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(self._CSV_HEADER)
        self._csv_path  = path
        self._frame_num = 0

    def stop_recording(self):
        self._csv_path = None

    def is_recording(self):
        return self._csv_path is not None

    def _write_csv_row(self, peak_db):
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
            f'{self._cfar_time_ms:.3f}',
            f'{peak_db:.2f}',
            self._n_detections,
            m.get('pc_work_time_avg_us', ''),
            m.get('pc_input_buf_full_avg', ''),
            m.get('pc_throughput_avg', ''),
            m.get('nitems_read', ''),
        ]
        try:
            with open(self._csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            print(f'[rd_plot_cfar] CSV write failed, stopping recording: {e}')
            self._csv_path = None


# ─────────────────────────────────────────────────────────────────────────────
# Qt widget — canvas + runtime controls
# ─────────────────────────────────────────────────────────────────────────────
class _RDCfarWidget(QtWidgets.QWidget):

    def __init__(self, num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, cfar_params,
                 baseline_m=0.0, carrier_freq_hz=510e6,
                 input_mode='single', cfar_trail_frames=0,
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget  { background: #111; color: #eee; }
            QLabel   { font-size: 10px; }
            QSlider::groove:horizontal { background:#333; height:4px; border-radius:2px; }
            QSlider::handle:horizontal { background:#00e5ff; width:12px; height:12px;
                                         margin:-4px 0; border-radius:6px; }
            QComboBox { background:#222; border:1px solid #444; padding:2px 6px; }
            QDoubleSpinBox, QSpinBox { background:#222; border:1px solid #444; padding:2px 4px; }
            QPushButton { background:#222; border:1px solid #444; padding:3px 10px; }
            QPushButton:hover { background:#2a2a2a; }
            QPushButton[recording="true"] { background:#8b1a1a; border:1px solid #c33;
                                            color:#fff; font-weight:bold; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.canvas = _RDCfarCanvas(
            num_doppler_bins, num_delay_bins,
            sample_rate, max_doppler,
            dynamic_range_db, colormap, cfar_params,
            baseline_m=baseline_m,
            carrier_freq_hz=carrier_freq_hz,
            input_mode=input_mode,
            cfar_trail_frames=cfar_trail_frames,
            parent=self,
        )
        layout.addWidget(self.canvas, stretch=1)

        # ── Controls row 1: display ───────────────────────────────────────
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(14)

        ctrl.addWidget(QtWidgets.QLabel('Dyn. Range:'))
        self._dr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._dr_slider.setRange(10, 120)
        self._dr_slider.setValue(int(dynamic_range_db))
        self._dr_slider.setFixedWidth(120)
        self._dr_lbl = QtWidgets.QLabel(f'{int(dynamic_range_db)} dB')
        self._dr_slider.valueChanged.connect(
            lambda v: (self.canvas.set_dynamic_range(v),
                       self._dr_lbl.setText(f'{v} dB'))
        )
        ctrl.addWidget(self._dr_slider)
        ctrl.addWidget(self._dr_lbl)

        ctrl.addWidget(QtWidgets.QLabel('Colourmap:'))
        self._cmap = QtWidgets.QComboBox()
        self._cmap.addItems(
            ['inferno', 'plasma', 'viridis', 'magma', 'hot', 'jet', 'turbo', 'RdBu_r']
        )
        self._cmap.setCurrentText(colormap)
        self._cmap.currentTextChanged.connect(self.canvas.set_colormap)
        ctrl.addWidget(self._cmap)

        # ── CFAR controls ─────────────────────────────────────────────────
        ctrl.addWidget(QtWidgets.QLabel('  Pfa:'))
        self._pfa = QtWidgets.QDoubleSpinBox()
        self._pfa.setDecimals(2)
        self._pfa.setRange(1e-8, 1e-1)
        self._pfa.setSingleStep(1e-5)
        self._pfa.setValue(cfar_params['pfa'])
        self._pfa.valueChanged.connect(self.canvas.set_pfa)
        ctrl.addWidget(self._pfa)

        ctrl.addWidget(QtWidgets.QLabel('  Trail:'))
        self._trail_spin = QtWidgets.QSpinBox()
        self._trail_spin.setRange(0, 20)
        self._trail_spin.setValue(max(0, int(cfar_trail_frames)))
        self._trail_spin.setSuffix(' fr')
        self._trail_spin.valueChanged.connect(self.canvas.set_cfar_trail_frames)
        ctrl.addWidget(self._trail_spin)

        ctrl.addStretch()

        self._rec_btn = QtWidgets.QPushButton('● Record CSV')
        self._rec_btn.setCheckable(True)
        self._rec_btn.clicked.connect(self._on_record_toggled)
        ctrl.addWidget(self._rec_btn)

        layout.addLayout(ctrl)

        self.resize(1260, 580)

    # ── Runtime setters (callable from GRC variable blocks) ──────────────
    def set_dynamic_range(self, val):      self._dr_slider.setValue(int(val))
    def set_colormap(self, name):          self._cmap.setCurrentText(name)
    def set_max_doppler(self, val):        self.canvas.set_max_doppler(val)
    def set_baseline(self, val):           self.canvas.set_baseline(val)
    def set_carrier_freq(self, val):       self.canvas.set_carrier_freq(val)
    def set_pfa(self, val):                self._pfa.setValue(float(val))
    def set_train_cells(self, r, d):       self.canvas.set_train_cells(r, d)
    def set_guard_cells(self, r, d):       self.canvas.set_guard_cells(r, d)
    def set_cfar_trail_frames(self, val):  self._trail_spin.setValue(max(0, int(val)))

    # ── Record button handler ────────────────────────────────────────────
    def _on_record_toggled(self, checked):
        if checked:
            default_name = 'rd_plot_cfar_{}.csv'.format(
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            default_path = os.path.join(os.getcwd(), default_name)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Record metrics to CSV', default_path,
                'CSV files (*.csv);;All files (*)'
            )
            if not path:
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

        self._rec_btn.setProperty('recording', 'true' if checked else 'false')
        self._rec_btn.style().unpolish(self._rec_btn)
        self._rec_btn.style().polish(self._rec_btn)


# ─────────────────────────────────────────────────────────────────────────────
# GNU Radio sink block
# ─────────────────────────────────────────────────────────────────────────────
class rd_plot_cfar(gr.sync_block):
    """
    Range-Doppler Qt Display Sink with CA-CFAR  —  gr-PR

    Identical input format to rd_plot. Displays two side-by-side plots:
      Left  : |CAF|² in dB (same as rd_plot)
      Right : 2D CA-CFAR detection mask

    Parameters
    ----------
    num_doppler_bins : int
    num_delay_bins   : int
    sample_rate      : float
    max_doppler      : float
    dynamic_range_db : float
    colormap         : str
    cfar_train_range   : int — half-window training cells along delay axis
    cfar_train_doppler : int — half-window training cells along doppler axis
    cfar_guard_range   : int — half-window guard cells along delay axis
    cfar_guard_doppler : int — half-window guard cells along doppler axis
    cfar_pfa           : float — probability of false alarm (e.g. 1e-4)
    """

    def __init__(self,
                 num_doppler_bins=128,
                 num_delay_bins=128,
                 sample_rate=2.4e6,
                 max_doppler=1000.0,
                 dynamic_range_db=60.0,
                 colormap='inferno',
                 baseline_m=0.0,
                 carrier_freq_hz=510e6,
                 cfar_train_range=8,
                 cfar_train_doppler=4,
                 cfar_guard_range=2,
                 cfar_guard_doppler=1,
                 cfar_pfa=1e-4,
                 # input_mode is startup-only: in_sig can't change after construction
                 input_mode='single',
                 cfar_trail_frames=0):

        vec_t = np.dtype((np.complex64, num_doppler_bins * num_delay_bins))
        # 'sum4': 4 inputs; coherent sum before |·|² drives RD/CFAR
        in_sig = [vec_t] * 4 if str(input_mode) == 'sum4' else [vec_t]

        gr.sync_block.__init__(
            self,
            name="PR RD Plot CFAR",
            in_sig=in_sig,
            out_sig=[],
        )

        self._nd         = num_doppler_bins
        self._nl         = num_delay_bins
        self._input_mode = str(input_mode)

        cfar_params = dict(
            train_r=int(cfar_train_range),
            train_d=int(cfar_train_doppler),
            guard_r=int(cfar_guard_range),
            guard_d=int(cfar_guard_doppler),
            pfa=float(cfar_pfa),
        )

        self._bridge = _Bridge()

        self._win = _RDCfarWidget(
            num_doppler_bins, num_delay_bins,
            sample_rate, max_doppler, dynamic_range_db,
            colormap, cfar_params,
            baseline_m=baseline_m, carrier_freq_hz=carrier_freq_hz,
            input_mode=input_mode,
            cfar_trail_frames=cfar_trail_frames,
        )

        self._bridge.new_frame.connect(
            self._win.canvas.update_frame,
            QtCore.Qt.QueuedConnection,
        )

        self._win.canvas.attach_block(self._collect_metrics)

        self._win.show()

    # ── Performance counter collection ───────────────────────────────────
    def _collect_metrics(self):
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
        frames = input_items[0]
        for i in range(len(frames)):
            if self._input_mode == 'sum4':
                # Emit tuple of 4 reshaped arrays; canvas does coherent sum
                payload = tuple(
                    input_items[p][i].reshape(self._nd, self._nl).copy()
                    for p in range(4)
                )
            else:
                payload = frames[i].reshape(self._nd, self._nl).copy()
            self._bridge.new_frame.emit(payload)
        return len(frames)

    # ── GRC integration ───────────────────────────────────────────────────
    def pyqwidget(self):
        return self._win

    # ── Runtime callbacks ────────────────────────────────────────────────
    def set_dynamic_range(self, val):     self._win.set_dynamic_range(val)
    def set_colormap(self, name):         self._win.set_colormap(name)
    def set_max_doppler(self, val):       self._win.set_max_doppler(val)
    def set_baseline(self, val):          self._win.set_baseline(val)
    def set_carrier_freq(self, val):      self._win.set_carrier_freq(val)
    def set_pfa(self, val):               self._win.set_pfa(val)
    def set_train_cells(self, r, d):      self._win.set_train_cells(r, d)
    def set_guard_cells(self, r, d):      self._win.set_guard_cells(r, d)
    # input_mode is startup-only (in_sig is fixed after construction)
    def set_cfar_trail_frames(self, val): self._win.set_cfar_trail_frames(val)
