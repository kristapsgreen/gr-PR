#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# gr-PR  —  rd_plot_cfar_doa.py
# Range-Doppler Qt Display Sink with CA-CFAR + CBF DOA + XY Geolocation Map


import collections
import csv
import math
import os
import time
from datetime import datetime

import numpy as np
import gnuradio.gr as gr

# Speed of light (m/s)
_C = 299_792_458.0

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QObject

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle


# ─────────────────────────────────────────────────────────────────────────────
# CA-CFAR (copied verbatim from rd_plot_cfar — pure numpy, integral-image)
# ─────────────────────────────────────────────────────────────────────────────
def _ca_cfar_2d(power, train_r, train_d, guard_r, guard_d, pfa):
    """
    2D Cell-Averaging CFAR on a power matrix.

    Parameters
    ----------
    power   : (nr, nd) float64 — |CAF|² (linear, NOT dB)
    train_r : int — half-window training cells along delay axis
    train_d : int — half-window training cells along Doppler axis
    guard_r : int — half-window guard cells along delay axis
    guard_d : int — half-window guard cells along Doppler axis
    pfa     : float — probability of false alarm

    Returns
    -------
    detections : bool (nr, nd)
    threshold  : float (nr, nd)
    """
    nr, nd = power.shape
    ii = np.pad(np.cumsum(np.cumsum(power, axis=0), axis=1), ((1, 0), (1, 0)))

    br = train_r + guard_r
    bd = train_d + guard_d
    rr, cc = np.indices((nr, nd))

    r0b = np.clip(rr - br, 0, nr); r1b = np.clip(rr + br + 1, 0, nr)
    c0b = np.clip(cc - bd, 0, nd); c1b = np.clip(cc + bd + 1, 0, nd)
    r0s = np.clip(rr - guard_r, 0, nr); r1s = np.clip(rr + guard_r + 1, 0, nr)
    c0s = np.clip(cc - guard_d, 0, nd); c1s = np.clip(cc + guard_d + 1, 0, nd)

    sum_big = ii[r1b, c1b] - ii[r0b, c1b] - ii[r1b, c0b] + ii[r0b, c0b]
    sum_sm  = ii[r1s, c1s] - ii[r0s, c1s] - ii[r1s, c0s] + ii[r0s, c0s]
    n_train = (r1b - r0b) * (c1b - c0b) - (r1s - r0s) * (c1s - c0s)

    valid = n_train > 0
    noise = np.zeros_like(power, dtype=np.float64)
    np.divide(sum_big - sum_sm, n_train, out=noise, where=valid)

    n_t   = max(1, int((2*br+1)*(2*bd+1) - (2*guard_r+1)*(2*guard_d+1)))
    alpha = n_t * (pfa ** (-1.0 / n_t) - 1.0)
    threshold  = alpha * noise
    detections = np.logical_and(valid, power > threshold)
    return detections, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Connected-components clustering (4-connectivity, no scipy)
# Returns list of (row, col) peak cells, one per cluster.
# ─────────────────────────────────────────────────────────────────────────────
def _cluster_detections(detections, power, max_targets):
    """
    Merge adjacent CFAR-positive cells (4-connectivity) into single targets.

    CFAR blobs span several cells around a bright target; picking only the
    peak cell per connected region avoids counting one target multiple times.

    Returns
    -------
    peaks : list of (row, col) — up to max_targets strongest cluster peaks
    """
    labeled  = np.zeros(detections.shape, dtype=np.int32)
    label_id = 0
    nr, nc   = detections.shape

    # Simple iterative flood-fill using a stack (avoids recursion limit)
    for r0 in range(nr):
        for c0 in range(nc):
            if detections[r0, c0] and labeled[r0, c0] == 0:
                label_id += 1
                stack = [(r0, c0)]
                labeled[r0, c0] = label_id
                while stack:
                    r, c = stack.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < nr and 0 <= cc < nc:
                            if detections[rr, cc] and labeled[rr, cc] == 0:
                                labeled[rr, cc] = label_id
                                stack.append((rr, cc))

    if label_id == 0:
        return []

    # Peak cell per cluster
    peaks = []
    for lid in range(1, label_id + 1):
        mask  = labeled == lid
        idx   = np.argmax(np.where(mask, power, 0.0))
        r, c  = divmod(int(idx), nc)
        peaks.append((r, c, float(power[r, c])))

    # Sort descending by power, cap at max_targets
    peaks.sort(key=lambda x: -x[2])
    return [(r, c) for r, c, _ in peaks[:max_targets]]


# ─────────────────────────────────────────────────────────────────────────────
# MVDR / Capon beamformer helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_steering_matrix(angles_deg, spacing_m, wavelength_m):
    """
    Precompute ULA steering matrix A of shape (4, n_angles).

    a(θ) = exp(j·2π·d/λ·sin(θ)·[0,1,2,3])
    All angles in degrees; spacing and wavelength in metres.
    """
    sin_th = np.sin(np.deg2rad(angles_deg))          # (n_angles,)
    phase  = (2.0 * np.pi * spacing_m / wavelength_m
              * np.arange(4)[:, np.newaxis]           # (4, 1)
              * sin_th[np.newaxis, :])                # (1, n_angles)
    return np.exp(1j * phase)                         # (4, n_angles) complex128


def _cbf_doa(caf_list, row, col, A_mat):
    """
    Estimate DOA for a single target cell using conventional beamforming.

    Snapshot x = [caf1[r,d], caf2[r,d], caf3[r,d], caf4[r,d]] (4,) complex.
    Beam power  P(θ) = |a(θ)^H x|²  — maximised at the angle whose steering
    vector matches the measured inter-element phase progression.
    No covariance matrix or inversion required; stable with a single snapshot.

    Returns the index into A_mat's column axis of the peak angle.
    """
    x    = np.array([caf[row, col] for caf in caf_list])   # (4,) complex
    resp = np.abs(A_mat.conj().T @ x) ** 2                  # (n_angles,)
    return int(np.argmax(resp))


# ─────────────────────────────────────────────────────────────────────────────
# Geodetic helpers (haversine — one-time init only, NOT in the hot path)
# ─────────────────────────────────────────────────────────────────────────────
_R_EARTH = 6_371_000.0  # mean spherical Earth radius (m)


def _haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in metres (spherical Earth, ≤0.3 % error < 50 km).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return _R_EARTH * 2.0 * math.asin(math.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """
    Forward bearing from point-1 → point-2 (degrees from North, clockwise).
    """
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _target_xy(bistatic_range_m, doa_deg, boresight_az_deg, tx_x, tx_y, baseline_m):
    """
    Compute target (x, y) via bistatic-ellipse / DOA-ray intersection.

    Geometry: radar at origin, +Y = North.
    Transmitter is at (tx_x, tx_y).
    Bistatic-range equation: ρ + |ρ·û − tx| = baseline + Rb
    Closed-form solution for the ray (bearing α from North):
        u  = tx_x·sin(α) + tx_y·cos(α)   (projection of tx onto target ray)
        ρ  = Rb·(Rb + 2·L) / (2·(Rb + L − u))
    where L = baseline (transmitter–receiver distance) and Rb = bistatic range.

    Returns (x_m, y_m) or None if geometry is degenerate.
    """
    alpha = math.radians((boresight_az_deg + doa_deg) % 360.0)
    sin_a = math.sin(alpha); cos_a = math.cos(alpha)
    u     = tx_x * sin_a + tx_y * cos_a
    Rb    = bistatic_range_m
    L     = baseline_m
    denom = 2.0 * (Rb + L - u)
    if abs(denom) < 1e-6:
        return None
    rho = Rb * (Rb + 2.0 * L) / denom
    if rho < 0.0:
        return None
    return (rho * sin_a, rho * cos_a)


# ─────────────────────────────────────────────────────────────────────────────
# Signal bridge: GR work thread → Qt GUI thread (QueuedConnection)
# ─────────────────────────────────────────────────────────────────────────────
class _Bridge(QObject):
    # Carries all 4 CAF matrices as a Python tuple; avoids copying in GR thread.
    new_frame = pyqtSignal(object)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib canvas — three subplots: RD map | CFAR | XY map
# ─────────────────────────────────────────────────────────────────────────────
class _RDCfarDoaCanvas(FigureCanvas):

    def __init__(self,
                 num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, cfar_params,
                 baseline_m, carrier_freq_hz,
                 # Beamforming
                 antenna_spacing_m, angle_step, max_targets,
                 # Geo
                 radar_lat, radar_lon, tx_lat, tx_lon,
                 boresight_az_deg, xy_extent_m,
                 rd_input_mode='caf1', cfar_trail_frames=0,
                 parent=None):

        self._nd           = num_doppler_bins
        self._nl           = num_delay_bins
        self._fs           = sample_rate
        self._max_dop      = float(max_doppler)
        self._dyn_db       = float(dynamic_range_db)
        self._baseline     = float(baseline_m)
        self._fc           = float(carrier_freq_hz)
        self._cfar         = dict(cfar_params)
        self._spacing      = float(antenna_spacing_m)
        self._angle_step   = float(angle_step)
        self._max_tgt      = int(max_targets)
        self._boresight    = float(boresight_az_deg)
        self._xy_extent    = float(xy_extent_m)
        self._rd_input_mode = str(rd_input_mode)

        # Trail deque: rolling history of past CFAR bool arrays
        n_trail = max(0, int(cfar_trail_frames))
        self._trail = collections.deque(maxlen=n_trail) if n_trail > 0 else None

        # ── Precompute angle grid and steering matrix ─────────────────────
        # Done ONCE at init — the (4, n_angles) matrix A is reused every frame.
        self._angles = np.arange(-90.0, 90.0 + self._angle_step, self._angle_step)
        lam = _C / self._fc
        self._A = _build_steering_matrix(self._angles, self._spacing, lam)

        # ── One-time geodetic computation (haversine, no hot-path geodesy) ─
        # Radar is fixed at origin; transmitter position is computed once.
        self._L   = _haversine_m(radar_lat, radar_lon, tx_lat, tx_lon)
        bearing   = _bearing_deg(radar_lat, radar_lon, tx_lat, tx_lon)
        b_rad     = math.radians(bearing)
        self._tx_x = self._L * math.sin(b_rad)   # East  (m)
        self._tx_y = self._L * math.cos(b_rad)   # North (m)

        # ── Figure: 2-row layout ──────────────────────────────────────────
        # Row 0: RD map (left) | CFAR detections (right)
        # Row 1: XY geolocation map (full width)
        from matplotlib.gridspec import GridSpec
        fig = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        super().__init__(fig)
        self.setParent(parent)
        gs = GridSpec(2, 2, figure=fig,
                      left=0.06, right=0.97,
                      top=0.95, bottom=0.07,
                      hspace=0.35, wspace=0.25)

        # Bistatic axes extents: X = Bistatic Excess Range Rb (m), starts at 0
        range_max_m = _C * num_delay_bins / sample_rate
        v_max_ms    = max_doppler * _C / self._fc
        self._extent = [0.0, range_max_m, -v_max_ms, v_max_ms]

        dummy = np.zeros((num_doppler_bins, num_delay_bins), dtype=np.float32)

        # ── Row 0 left: dB RD map (caf1 only) ────────────────────────────
        ax_l = fig.add_subplot(gs[0, 0])
        self._ax_l = ax_l
        self._im_l = ax_l.imshow(
            dummy, aspect='auto', origin='lower', extent=self._extent,
            cmap=colormap, vmin=0.0, vmax=1.0, interpolation='nearest',
        )
        cbar_l = fig.colorbar(self._im_l, ax=ax_l, fraction=0.035, pad=0.02)
        cbar_l.set_label('Norm. Power (dB)', color='#cccccc', fontsize=8)
        cbar_l.set_ticks(np.linspace(0, 1, 6))
        cbar_l.set_ticklabels(
            [f'{v:.0f}' for v in np.linspace(-dynamic_range_db, 0, 6)]
        )
        for lbl in cbar_l.ax.yaxis.get_ticklabels():
            lbl.set_color('#cccccc')
        cbar_l.ax.yaxis.set_tick_params(color='#666666')
        self._cbar_l = cbar_l

        # ── Row 0 right: CFAR detection mask ──────────────────────────────
        # White pixels = every cell passing CA-CFAR threshold.
        # Red X markers = cluster-peak cells sent to MVDR (one per target blob).
        ax_m = fig.add_subplot(gs[0, 1])
        self._ax_m = ax_m
        self._im_m = ax_m.imshow(
            dummy, aspect='auto', origin='lower', extent=self._extent,
            cmap='Greys_r', vmin=0.0, vmax=1.0, interpolation='nearest',
        )
        self._scat_m = ax_m.scatter([], [], s=40, c='#ff4444',
                                    marker='x', linewidths=1.5, zorder=5)

        # ── Row 1: XY geolocation map (spans both columns) ───────────────
        ax_r = fig.add_subplot(gs[1, :])
        self._ax_r = ax_r
        ax_r.set_facecolor('#0a0a0a')
        ax_r.set_xlim(-xy_extent_m, xy_extent_m)
        ax_r.set_ylim(-xy_extent_m, xy_extent_m)
        ax_r.set_aspect('equal', adjustable='datalim')
        ax_r.set_xlabel('East  (m)', color='#cccccc', fontsize=9)
        ax_r.set_ylabel('North (m)', color='#cccccc', fontsize=9)
        ax_r.set_title('XY Geolocation', color='#ffffff', fontsize=11,
                        fontweight='bold')
        ax_r.tick_params(colors='#888888', labelsize=7)
        for sp in ax_r.spines.values():
            sp.set_edgecolor('#333333')
        ax_r.grid(True, color='#222222', linewidth=0.5)
        ax_r.axhline(0, color='#333333', lw=0.5)
        ax_r.axvline(0, color='#333333', lw=0.5)

        # Range rings every 5 km for scale context
        for ring_m in range(5000, int(xy_extent_m) + 1, 5000):
            circle = Circle((0, 0), ring_m, fill=False,
                            edgecolor='#2a2a2a', linewidth=0.8)
            ax_r.add_patch(circle)
            ax_r.text(ring_m * 0.707, ring_m * 0.707,
                      f'{ring_m//1000}km', color='#444444',
                      fontsize=6, ha='left', va='bottom')

        # Fixed map markers: radar (cyan ▲) and transmitter (orange ★)
        ax_r.plot(0, 0, marker='^', color='#00e5ff', markersize=10,
                  label='Radar', zorder=6, linestyle='none')
        ax_r.plot(self._tx_x, self._tx_y, marker='*', color='#ff8800',
                  markersize=12, label='Tx', zorder=6, linestyle='none')
        ax_r.legend(loc='upper left', fontsize=7,
                    facecolor='#111', edgecolor='#444', labelcolor='#ccc')

        # Dynamic scatter + annotation for target dots — rebuilt each frame
        self._tgt_scatter = ax_r.scatter([], [], s=60, c='#ff4444',
                                         marker='o', zorder=7)
        self._tgt_texts   = []   # list of Text artists, recycled each frame

        # ── Shared styling for the two RD axes ───────────────────────────
        for ax, title in ((ax_l, f'Range–Doppler Map  [{rd_input_mode}]'),
                          (ax_m, 'CA-CFAR Detections')):
            ax.set_facecolor('#0a0a0a')
            ax.set_xlabel('Bistatic Range  (m)', color='#cccccc', fontsize=9)
            ax.set_ylabel('Bistatic Velocity  (m/s)', color='#cccccc', fontsize=9)
            ax.set_title(title, color='#ffffff', fontsize=11, fontweight='bold')
            ax.tick_params(colors='#888888', labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor('#333333')
            ax.axhline(0.0, color='#00e5ff', linewidth=0.7,
                       linestyle='--', alpha=0.6)

        # ── Performance / recording state ─────────────────────────────────
        self._last_time      = None
        self._frame_dt_ms    = 0.0
        self._render_time_ms = 0.0
        self._cfar_time_ms   = 0.0
        self._doa_time_ms    = 0.0
        self._frame_num      = 0
        self._n_detections   = 0
        self._n_targets      = 0
        self._target_list    = []    # list of dicts; shared with text widget
        self._csv_path       = None
        self._metrics_cb     = None
        self._target_list_cb = None  # called each frame with new target list

        self.draw()

    # ── Called only from Qt GUI thread via QueuedConnection ──────────────
    def update_frame(self, caf_tuple):
        """
        Main per-frame processing: runs in Qt thread only.

        caf_tuple : (caf1, caf2, caf3, caf4) as (nd, nl) complex64 arrays.
        """
        render_start = time.perf_counter()
        if self._last_time is not None:
            self._frame_dt_ms = (render_start - self._last_time) * 1e3
        self._last_time = render_start

        caf1, caf2, caf3, caf4 = caf_tuple
        caf_list = [caf1, caf2, caf3, caf4]

        # ── Step 1: RD/CFAR power — caf1 only, or coherent sum of all 4 ──
        if self._rd_input_mode == 'sum4':
            caf_rd = caf1 + caf2 + caf3 + caf4
        else:
            caf_rd = caf1
        mag_sq = (np.abs(caf_rd) ** 2).astype(np.float64)

        # dB for the left plot
        eps    = 1e-30
        mat_db = 10.0 * np.log10(mag_sq + eps)
        peak   = mat_db.max()
        mat_norm = np.clip(
            (mat_db - (peak - self._dyn_db)) / self._dyn_db, 0.0, 1.0
        ).astype(np.float32)
        self._im_l.set_data(mat_norm)

        # ── Step 2: CA-CFAR ───────────────────────────────────────────────
        cfar_start = time.perf_counter()
        detections, _ = _ca_cfar_2d(
            mag_sq,
            self._cfar['train_r'], self._cfar['train_d'],
            self._cfar['guard_r'], self._cfar['guard_d'],
            self._cfar['pfa'],
        )
        self._cfar_time_ms = (time.perf_counter() - cfar_start) * 1e3
        self._n_detections  = int(detections.sum())

        # Trail overlay: combine past frames with current detections
        if self._trail is not None and len(self._trail) > 0:
            overlay = np.zeros(detections.shape, dtype=np.float32)
            k = len(self._trail)
            for i, past in enumerate(self._trail):
                w = 0.15 + 0.35 * (i / max(1, k - 1))
                np.maximum(overlay, past.astype(np.float32) * w, out=overlay)
            np.maximum(overlay, detections.astype(np.float32), out=overlay)
            self._im_m.set_data(overlay)
        else:
            self._im_m.set_data(detections.astype(np.float32))

        # Append current frame AFTER rendering
        if self._trail is not None:
            self._trail.append(detections.copy())

        # ── Step 3: Cluster → peak cells ─────────────────────────────────
        peaks = _cluster_detections(detections, mag_sq, self._max_tgt)
        self._n_targets = len(peaks)

        # Convert cluster-peak bin indices to bistatic range / velocity
        range_max_m = _C * self._nl / self._fs
        v_max_ms    = self._max_dop * _C / self._fc

        # Overlay peaks as crosses on the CFAR plot
        if peaks:
            # Map (row, col) → physical axes coords for the scatter overlay
            p_range = [col * _C / self._fs for row, col in peaks]
            p_vel   = [-v_max_ms + (row / max(1, self._nd - 1))
                       * 2.0 * v_max_ms for row, col in peaks]
            self._scat_m.set_offsets(np.column_stack([p_range, p_vel]))
        else:
            self._scat_m.set_offsets(np.empty((0, 2)))

        # ── Step 4: CBF DOA + geolocation for each target ────────────────
        doa_start = time.perf_counter()
        targets = []
        for idx, (row, col) in enumerate(peaks):
            # Bistatic range (m): delay_bin * c / fs -- had a bug of adding a baseline
            Rb = col * _C / self._fs
            # Bistatic velocity (m/s): centred Doppler bin → velocity
            doppler_center_bin = row - self._nd / 2.0
            v_ms = doppler_center_bin * (_C / self._fc) * (self._max_dop / (self._nd / 2.0))

            # Conventional beamforming — single peak-cell snapshot
            angle_idx = _cbf_doa(caf_list, row, col, self._A)

            doa_deg     = float(self._angles[angle_idx])
            global_bear = (self._boresight + doa_deg) % 360.0

            # XY from bistatic-ellipse / DOA-ray closed form
            xy = _target_xy(Rb, doa_deg, self._boresight,
                            self._tx_x, self._tx_y, self._L)

            targets.append({
                'idx':      idx,
                'Rb_m':     Rb,
                'v_ms':     v_ms,
                'doa_deg':  doa_deg,
                'bear_deg': global_bear,
                'xy':       xy,
                'row':      row,
                'col':      col,
            })

        self._doa_time_ms = (time.perf_counter() - doa_start) * 1e3
        self._target_list  = targets

        # ── Step 5: XY map update ─────────────────────────────────────────
        valid_tgts = [t for t in targets if t['xy'] is not None]

        # Remove old text annotations
        for txt in self._tgt_texts:
            txt.remove()
        self._tgt_texts = []

        if valid_tgts:
            xs = [t['xy'][0] for t in valid_tgts]
            ys = [t['xy'][1] for t in valid_tgts]
            self._tgt_scatter.set_offsets(np.column_stack([xs, ys]))
            for t in valid_tgts:
                x, y = t['xy']
                label = f"{t['idx']}"
                txt = self._ax_r.text(
                    x, y, f"  {label}", color='#ffcccc',
                    fontsize=6, va='center', zorder=8,
                )
                self._tgt_texts.append(txt)
        else:
            self._tgt_scatter.set_offsets(np.empty((0, 2)))

        # ── Step 6: Title updates ─────────────────────────────────────────
        mode_tag = f'[{self._rd_input_mode}]'
        self._ax_l.set_title(
            f'Range–Doppler Map  {mode_tag}    peak = {peak:.1f} dB',
            color='#ffffff', fontsize=11, fontweight='bold',
        )
        self._ax_m.set_title(
            f'CA-CFAR  Pfa={self._cfar["pfa"]:.0e}  '
            f'{self._n_detections} det → {self._n_targets} tgt  '
            f'CFAR {self._cfar_time_ms:.1f}ms  DOA {self._doa_time_ms:.1f}ms  '
            f'frame {self._frame_dt_ms:.0f}ms',
            color='#ffffff', fontsize=9, fontweight='bold',
        )
        self._ax_r.set_title(
            f'XY Map  L={self._L/1e3:.1f} km  '
            f'{len(valid_tgts)} located',
            color='#ffffff', fontsize=11, fontweight='bold',
        )

        self.draw_idle()

        self._render_time_ms = (time.perf_counter() - render_start) * 1e3
        self._frame_num += 1

        # Notify the target-list widget
        if self._target_list_cb is not None:
            try:
                self._target_list_cb(targets)
            except Exception:
                pass

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
        self._rescale_rd_axes()

    def set_baseline(self, val):
        self._baseline = float(val)
        self._rescale_rd_axes()

    def set_carrier_freq(self, val):
        self._fc = float(val)
        lam = _C / self._fc
        self._A = _build_steering_matrix(self._angles, self._spacing, lam)
        self._rescale_rd_axes()

    def set_pfa(self, val):
        self._cfar['pfa'] = float(val)

    def set_train_cells(self, r, d):
        self._cfar['train_r'] = int(r)
        self._cfar['train_d'] = int(d)

    def set_guard_cells(self, r, d):
        self._cfar['guard_r'] = int(r)
        self._cfar['guard_d'] = int(d)

    def set_antenna_spacing(self, val):
        self._spacing = float(val)
        lam = _C / self._fc
        self._A = _build_steering_matrix(self._angles, self._spacing, lam)

    def set_max_targets(self, val):
        self._max_tgt = int(val)

    def set_angle_step(self, val):
        self._angle_step = float(val)
        self._angles = np.arange(-90.0, 90.0 + self._angle_step, self._angle_step)
        lam = _C / self._fc
        self._A = _build_steering_matrix(self._angles, self._spacing, lam)

    def set_rd_input_mode(self, val):
        self._rd_input_mode = str(val)

    def set_cfar_trail_frames(self, val):
        n = max(0, int(val))
        if n == 0:
            self._trail = None
        elif self._trail is None:
            self._trail = collections.deque(maxlen=n)
        else:
            old = list(self._trail)
            self._trail = collections.deque(old[-n:], maxlen=n)

    def _rescale_rd_axes(self):
        range_max_m = _C * self._nl / self._fs
        v_max_ms    = self._max_dop * _C / self._fc
        self._extent = [0.0, range_max_m, -v_max_ms, v_max_ms]
        for im, ax in ((self._im_l, self._ax_l), (self._im_m, self._ax_m)):
            im.set_extent(self._extent)
            ax.set_xlim(self._extent[0], self._extent[1])
            ax.set_ylim(self._extent[2], self._extent[3])
        self.draw_idle()

    # ── Block reference (for perf-counter metrics in CSV) ─────────────────
    def attach_block(self, metrics_cb):
        self._metrics_cb = metrics_cb

    def attach_target_list_cb(self, cb):
        self._target_list_cb = cb

    # ── CSV recording ─────────────────────────────────────────────────────
    _CSV_HEADER = [
        'timestamp_iso', 'frame_num',
        'frame_dt_ms', 'render_time_ms', 'cfar_time_ms', 'doa_time_ms',
        'peak_db', 'n_detections', 'n_targets',
        'pc_work_time_avg_us', 'pc_input_buf_full_avg',
        'pc_throughput_avg', 'nitems_read',
    ]

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
        m = {}
        if self._metrics_cb is not None:
            try:
                m = self._metrics_cb()
            except Exception:
                pass
        row = [
            datetime.now().isoformat(timespec='milliseconds'),
            self._frame_num,
            f'{self._frame_dt_ms:.3f}',
            f'{self._render_time_ms:.3f}',
            f'{self._cfar_time_ms:.3f}',
            f'{self._doa_time_ms:.3f}',
            f'{peak_db:.2f}',
            self._n_detections,
            self._n_targets,
            m.get('pc_work_time_avg_us', ''),
            m.get('pc_input_buf_full_avg', ''),
            m.get('pc_throughput_avg', ''),
            m.get('nitems_read', ''),
        ]
        try:
            with open(self._csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            print(f'[rd_plot_cfar_doa] CSV write failed, stopping recording: {e}')
            self._csv_path = None


# ─────────────────────────────────────────────────────────────────────────────
# Qt widget — canvas + runtime controls + target-list panel
# ─────────────────────────────────────────────────────────────────────────────
class _RDCfarDoaWidget(QtWidgets.QWidget):

    def __init__(self,
                 num_doppler_bins, num_delay_bins,
                 sample_rate, max_doppler, dynamic_range_db,
                 colormap, cfar_params,
                 baseline_m, carrier_freq_hz,
                 antenna_spacing_m, angle_step, max_targets,
                 radar_lat, radar_lon, tx_lat, tx_lon,
                 boresight_az_deg, xy_extent_m,
                 rd_input_mode='caf1', cfar_trail_frames=0,
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
            QPlainTextEdit { background:#0a0a0a; color:#4a5a6a; font-family:'Consolas','Courier New',monospace;
                             font-size:8px; border:1px solid #1e2830; }
        """)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Horizontal split: canvas (left) + target list (right) ────────
        h_split = QtWidgets.QHBoxLayout()
        outer.addLayout(h_split, stretch=1)

        self.canvas = _RDCfarDoaCanvas(
            num_doppler_bins, num_delay_bins,
            sample_rate, max_doppler, dynamic_range_db,
            colormap, cfar_params,
            baseline_m=baseline_m,
            carrier_freq_hz=carrier_freq_hz,
            antenna_spacing_m=antenna_spacing_m,
            angle_step=angle_step,
            max_targets=max_targets,
            radar_lat=radar_lat, radar_lon=radar_lon,
            tx_lat=tx_lat, tx_lon=tx_lon,
            boresight_az_deg=boresight_az_deg,
            xy_extent_m=xy_extent_m,
            rd_input_mode=rd_input_mode,
            cfar_trail_frames=cfar_trail_frames,
            parent=self,
        )
        h_split.addWidget(self.canvas, stretch=4)

        # ── Target list widget (right panel) ──────────────────────────────
        # Read-only QPlainTextEdit: monospace, auto-refreshed each frame.
        tgt_panel = QtWidgets.QVBoxLayout()
        tgt_panel.addWidget(QtWidgets.QLabel('Targets'))
        self._tgt_text = QtWidgets.QPlainTextEdit()
        self._tgt_text.setReadOnly(True)
        self._tgt_text.setFixedWidth(260)
        self._tgt_text.setPlaceholderText('No targets detected.')
        tgt_panel.addWidget(self._tgt_text)
        h_split.addLayout(tgt_panel, stretch=0)

        self.canvas.attach_target_list_cb(self._update_target_list)

        # ── Controls row ──────────────────────────────────────────────────
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(12)

        ctrl.addWidget(QtWidgets.QLabel('Dyn. Range:'))
        self._dr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._dr_slider.setRange(10, 120)
        self._dr_slider.setValue(int(dynamic_range_db))
        self._dr_slider.setFixedWidth(100)
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

        ctrl.addWidget(QtWidgets.QLabel('  Pfa:'))
        self._pfa = QtWidgets.QDoubleSpinBox()
        self._pfa.setDecimals(2)
        self._pfa.setRange(1e-8, 1e-1)
        self._pfa.setSingleStep(1e-5)
        self._pfa.setValue(cfar_params['pfa'])
        self._pfa.valueChanged.connect(self.canvas.set_pfa)
        ctrl.addWidget(self._pfa)

        ctrl.addWidget(QtWidgets.QLabel('  RD mode:'))
        self._rd_mode_combo = QtWidgets.QComboBox()
        self._rd_mode_combo.addItems(['caf1', 'sum4'])
        self._rd_mode_combo.setCurrentText(str(rd_input_mode))
        self._rd_mode_combo.currentTextChanged.connect(self.canvas.set_rd_input_mode)
        ctrl.addWidget(self._rd_mode_combo)

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

        outer.addLayout(ctrl)

        self.resize(1400, 900)

    # ── Target list refresh (called from canvas each frame) ───────────────
    def _update_target_list(self, targets):
        if not targets:
            self._tgt_text.setPlainText('No targets detected.')
            return
        lines = [
            f"{'#':>2}  {'Rb(km)':>7}  {'v(m/s)':>7}  "
            f"{'DOA°':>6}  {'Bear°':>6}  {'X(km)':>7}  {'Y(km)':>7}",
            '─' * 58,
        ]
        for t in targets:
            xy_str = '  —  ' if t['xy'] is None else (
                f"{t['xy'][0]/1e3:7.2f}  {t['xy'][1]/1e3:7.2f}"
            )
            lines.append(
                f"{t['idx']:>2}  "
                f"{t['Rb_m']/1e3:7.2f}  "
                f"{t['v_ms']:7.1f}  "
                f"{t['doa_deg']:6.1f}  "
                f"{t['bear_deg']:6.1f}  "
                f"{xy_str}"
            )
        self._tgt_text.setPlainText('\n'.join(lines))

    # ── Runtime setters ───────────────────────────────────────────────────
    def set_dynamic_range(self, val):      self._dr_slider.setValue(int(val))
    def set_colormap(self, name):          self._cmap.setCurrentText(name)
    def set_max_doppler(self, val):        self.canvas.set_max_doppler(val)
    def set_baseline(self, val):           self.canvas.set_baseline(val)
    def set_carrier_freq(self, val):       self.canvas.set_carrier_freq(val)
    def set_pfa(self, val):                self._pfa.setValue(float(val))
    def set_train_cells(self, r, d):       self.canvas.set_train_cells(r, d)
    def set_guard_cells(self, r, d):       self.canvas.set_guard_cells(r, d)
    def set_antenna_spacing(self, v):      self.canvas.set_antenna_spacing(v)
    def set_max_targets(self, v):          self.canvas.set_max_targets(v)
    def set_angle_step(self, v):           self.canvas.set_angle_step(v)
    def set_rd_input_mode(self, val):      self._rd_mode_combo.setCurrentText(str(val))
    def set_cfar_trail_frames(self, val):  self._trail_spin.setValue(max(0, int(val)))

    # ── Record CSV ────────────────────────────────────────────────────────
    def _on_record_toggled(self, checked):
        if checked:
            default_name = 'rd_plot_cfar_doa_{}.csv'.format(
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
class rd_plot_cfar_doa(gr.sync_block):
    """
    RD Plot + CA-CFAR + MVDR DOA + XY Geolocation  —  gr-PR

    4-input sink block. Accepts the four flattened CAF matrices from cu_caf_4x
    (caf1..caf4, each complex64 vector of length num_doppler_bins×num_delay_bins).

    CAF-1 drives the Range-Doppler map and CA-CFAR detection (same as
    rd_plot_cfar). The four CAFs together feed the MVDR spatial covariance
    estimator, yielding a DOA angle per detected target. Target positions are
    then placed on a 2D XY map using bistatic-ellipse / DOA-ray geometry.

    All heavy computation (CFAR, clustering, MVDR, geolocation, plotting)
    runs in the Qt GUI thread via a QueuedConnection signal — the GNU Radio
    work() method only copies and emits; no processing happens there.

    Parameters
    ----------
    num_doppler_bins  : int
    num_delay_bins    : int
    sample_rate       : float — Hz
    max_doppler       : float — Hz, max Doppler shift
    dynamic_range_db  : float — dB, display dynamic range
    colormap          : str   — matplotlib colormap name
    baseline_m        : float — transmitter–receiver baseline L (m)
    carrier_freq_hz   : float — illuminator carrier frequency (Hz)
    cfar_train_range  : int   — CFAR half-window training cells (delay)
    cfar_train_doppler: int   — CFAR half-window training cells (Doppler)
    cfar_guard_range  : int   — CFAR half-window guard cells (delay)
    cfar_guard_doppler: int   — CFAR half-window guard cells (Doppler)
    cfar_pfa          : float — probability of false alarm
    max_targets       : int   — cap on simultaneous tracked targets
    antenna_spacing_m : float — ULA inter-element spacing (m)
    angle_step        : float — angular resolution of DOA grid (deg)
    radar_lat         : float — receiver latitude  (WGS84 deg)
    radar_lon         : float — receiver longitude (WGS84 deg)
    tx_lat            : float — transmitter latitude  (WGS84 deg)
    tx_lon            : float — transmitter longitude (WGS84 deg)
    boresight_az_deg  : float — array boresight bearing from North (deg CW)
    xy_extent_m       : float — half-size of XY map window (m)
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
                 # CFAR
                 cfar_train_range=8,
                 cfar_train_doppler=4,
                 cfar_guard_range=2,
                 cfar_guard_doppler=1,
                 cfar_pfa=1e-4,
                 # Target clustering / DOA
                 max_targets=32,
                 antenna_spacing_m=0.3,
                 angle_step=1.0,
                 # Geo
                 radar_lat=0.0,
                 radar_lon=0.0,
                 tx_lat=0.0,
                 tx_lon=0.01,
                 boresight_az_deg=0.0,
                 xy_extent_m=15000.0,
                 # RD input mode: 'caf1' uses only caf1; 'sum4' coherently sums all 4
                 rd_input_mode='caf1',
                 cfar_trail_frames=0):

        vec_len = num_doppler_bins * num_delay_bins
        gr.sync_block.__init__(
            self,
            name="PR RD Plot CFAR DOA",
            in_sig=[np.dtype((np.complex64, vec_len))] * 4,   # caf1..caf4
            out_sig=[],
        )

        self._nd = num_doppler_bins
        self._nl = num_delay_bins

        cfar_params = dict(
            train_r=int(cfar_train_range),
            train_d=int(cfar_train_doppler),
            guard_r=int(cfar_guard_range),
            guard_d=int(cfar_guard_doppler),
            pfa=float(cfar_pfa),
        )

        self._bridge = _Bridge()

        self._win = _RDCfarDoaWidget(
            num_doppler_bins, num_delay_bins,
            sample_rate, max_doppler, dynamic_range_db,
            colormap, cfar_params,
            baseline_m=baseline_m,
            carrier_freq_hz=carrier_freq_hz,
            antenna_spacing_m=antenna_spacing_m,
            angle_step=angle_step,
            max_targets=max_targets,
            radar_lat=radar_lat, radar_lon=radar_lon,
            tx_lat=tx_lat, tx_lon=tx_lon,
            boresight_az_deg=boresight_az_deg,
            xy_extent_m=xy_extent_m,
            rd_input_mode=rd_input_mode,
            cfar_trail_frames=cfar_trail_frames,
        )

        # QueuedConnection: work() runs in GR thread, update_frame() in Qt thread
        self._bridge.new_frame.connect(
            self._win.canvas.update_frame,
            QtCore.Qt.QueuedConnection,
        )

        self._win.canvas.attach_block(self._collect_metrics)
        self._win.show()

    # ── Performance-counter collection ────────────────────────────────────
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

    # ── GNU Radio work() — copy & emit only; all processing in Qt thread ──
    def work(self, input_items, output_items):
        """
        Called by the GR scheduler when all 4 input ports have data.
        cu_caf_4x emits all 4 outputs atomically per CPI, so the scheduler
        will not call us until a full 4-channel frame is available — no
        extra synchronisation is needed here.
        """
        n = len(input_items[0])
        for i in range(n):
            caf1 = input_items[0][i].reshape(self._nd, self._nl).copy()
            caf2 = input_items[1][i].reshape(self._nd, self._nl).copy()
            caf3 = input_items[2][i].reshape(self._nd, self._nl).copy()
            caf4 = input_items[3][i].reshape(self._nd, self._nl).copy()
            self._bridge.new_frame.emit((caf1, caf2, caf3, caf4))
        return n

    # ── GRC integration ───────────────────────────────────────────────────
    def pyqwidget(self):
        return self._win

    # ── Runtime callbacks (wirable from GRC variable blocks) ─────────────
    def set_dynamic_range(self, val):     self._win.set_dynamic_range(val)
    def set_colormap(self, name):         self._win.set_colormap(name)
    def set_max_doppler(self, val):       self._win.set_max_doppler(val)
    def set_baseline(self, val):          self._win.set_baseline(val)
    def set_carrier_freq(self, val):      self._win.set_carrier_freq(val)
    def set_pfa(self, val):               self._win.set_pfa(val)
    def set_train_cells(self, r, d):      self._win.set_train_cells(r, d)
    def set_guard_cells(self, r, d):      self._win.set_guard_cells(r, d)
    def set_antenna_spacing(self, v):     self._win.set_antenna_spacing(v)
    def set_max_targets(self, v):         self._win.set_max_targets(v)
    def set_angle_step(self, v):          self._win.set_angle_step(v)
    def set_rd_input_mode(self, val):     self._win.set_rd_input_mode(val)
    def set_cfar_trail_frames(self, val): self._win.set_cfar_trail_frames(val)
