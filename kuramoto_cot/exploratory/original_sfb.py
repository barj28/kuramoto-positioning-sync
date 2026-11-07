"""VERBATIM copy (from line 7) of barj_backtester/src/research/kuramoto_sync.py, untracked in the main tree on
2026-11-07, kept as the object under study: the SFB of the rejected paper. Do not edit."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Acceleration dispatch — follows existing pattern (src/accel/layer4_kernels.py)
try:
    from src.accel import is_accel_enabled
    from src.accel.kuramoto_kernels import extract_all_phases_njit
    _USE_ACCEL = True
except ImportError:
    _USE_ACCEL = False


@dataclass
class KuramotoResult:
    """Output of Kuramoto synchronization analysis."""
    r: np.ndarray               # (T,) order parameter [0,1], NaN where invalid
    psi: np.ndarray             # (T,) cluster phase [0, 2pi], NaN where invalid
    n_active: np.ndarray        # (T,) int, active strategies per timestep
    phases: np.ndarray          # (N, T) individual phases, NaN where invalid
    peak_times: list[list[int]] # per-strategy retained peak indices
    params: dict                # {nu, W, dt_min, T_max}


# ---------------------------------------------------------------------------
# Pure-Python helpers (fallback path)
# ---------------------------------------------------------------------------

def _compute_flows(positions: np.ndarray, capital: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute position flows and normalized flux.

    Args:
        positions: (N, T) position array
        capital: (N,) capital per strategy

    Returns:
        flows: (N, T) — q_i(t) = pos_i(t) - pos_i(t-1), first col = 0
        phi: (N, T) — |q_i(t)| / C_i
    """
    N, T = positions.shape
    flows = np.zeros((N, T), dtype=np.float64)
    flows[:, 1:] = np.diff(positions, axis=1)
    cap = capital.reshape(N, 1)
    cap = np.where(cap == 0, 1.0, cap)  # avoid div-by-zero
    phi = np.abs(flows) / cap
    return flows, phi


def _detect_flux_peaks(flows_1d: np.ndarray, phi_1d: np.ndarray, nu: float, W: int) -> list[int]:
    """Detect signed flux peaks with adaptive threshold.

    Positive peaks: local max of flows where flows > 0, phi above threshold.
    Negative peaks: local min of flows where flows < 0, phi above threshold.
    Consecutive same-sign peaks are consolidated (keep largest phi).
    """
    T = len(flows_1d)
    raw_peaks = []
    for t in range(1, T - 1):
        win_start = max(0, t - W + 1)
        window = phi_1d[win_start:t + 1]
        mean = np.mean(window)
        std = np.std(window)  # ddof=0 (population)
        threshold = mean + nu * std

        if phi_1d[t] <= threshold:
            continue

        # Positive peak: local max of signed flow, flow > 0
        if (flows_1d[t] > 0
                and flows_1d[t] > flows_1d[t - 1]
                and flows_1d[t] > flows_1d[t + 1]):
            raw_peaks.append(t)
        # Negative peak: local min of signed flow, flow < 0
        elif (flows_1d[t] < 0
                and flows_1d[t] < flows_1d[t - 1]
                and flows_1d[t] < flows_1d[t + 1]):
            raw_peaks.append(t)

    # Consolidate consecutive same-sign peaks (keep largest phi)
    if len(raw_peaks) <= 1:
        return raw_peaks

    result = [raw_peaks[0]]
    for pk in raw_peaks[1:]:
        if np.sign(flows_1d[pk]) == np.sign(flows_1d[result[-1]]):
            if phi_1d[pk] > phi_1d[result[-1]]:
                result[-1] = pk
        else:
            result.append(pk)
    return result


def _merge_close_peaks(peaks: list[int], phi_1d: np.ndarray, dt_min: int) -> list[int]:
    """Greedily merge peaks closer than dt_min, keeping larger phi. Iterate until stable."""
    if len(peaks) <= 1:
        return list(peaks)

    buf = list(peaks)
    changed = True
    while changed:
        changed = False
        new_buf = []
        i = 0
        while i < len(buf):
            if i + 1 < len(buf) and (buf[i + 1] - buf[i]) < dt_min:
                # Keep larger
                if phi_1d[buf[i]] >= phi_1d[buf[i + 1]]:
                    new_buf.append(buf[i])
                else:
                    new_buf.append(buf[i + 1])
                i += 2
                changed = True
            else:
                new_buf.append(buf[i])
                i += 1
        buf = new_buf

    return buf


def _interpolate_phase(
    peaks: list[int], T: int, pos_1d: np.ndarray, flows_1d: np.ndarray, T_max: int,
) -> np.ndarray:
    """Interpolate phase between peaks. NaN outside valid range and during flat runs > T_max.

    Sign-aware base: first positive peak starts at 0, first negative peak starts at pi.
    Each subsequent peak increments by pi (unwrapped), then mod 2pi.
    """
    phase = np.full(T, np.nan)

    if len(peaks) < 2:
        return phase

    # Sign-aware base: 0 if first peak is a buy (positive flow), pi if sell (negative)
    base = 0.0 if flows_1d[peaks[0]] > 0 else math.pi

    # Interpolate between consecutive peaks
    for k in range(len(peaks) - 1):
        t_k = peaks[k]
        t_k1 = peaks[k + 1]
        span = t_k1 - t_k
        if span == 0:
            continue
        ts = np.arange(t_k, t_k1 + 1)
        unwrapped_k = base + k * math.pi
        raw = unwrapped_k + math.pi * (ts - t_k) / span
        phase[t_k:t_k1 + 1] = raw % (2.0 * math.pi)

    # NaN before first peak, after last peak
    phase[:peaks[0]] = np.nan
    phase[peaks[-1] + 1:] = np.nan

    # Flat mask: consecutive |pos| < 1e-10 for > T_max days
    flat_run = 0
    flat_start = 0
    for t in range(T):
        if abs(pos_1d[t]) < 1e-10:
            if flat_run == 0:
                flat_start = t
            flat_run += 1
        else:
            if flat_run > T_max:
                phase[flat_start:t] = np.nan
            flat_run = 0
    if flat_run > T_max:
        phase[flat_start:T] = np.nan

    return phase


def _compute_order_parameter(
    phases: np.ndarray, capital: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute capital-weighted Kuramoto order parameter.

    Args:
        phases: (N, T) with NaN where invalid
        capital: (N,)

    Returns:
        r: (T,) order parameter in [0, 1], NaN where no active strategies
        psi: (T,) cluster phase in [0, 2pi], NaN where no active strategies
        n_active: (T,) int count of active strategies
    """
    N, T = phases.shape
    valid = ~np.isnan(phases)  # (N, T) bool
    n_active = valid.sum(axis=0).astype(np.int64)  # (T,)

    # Complex phases: exp(i*theta) where valid, 0 elsewhere
    complex_phases = np.where(valid, np.exp(1j * phases), 0.0 + 0j)  # (N, T)

    # Capital-weighted sum
    cap_weights = capital.reshape(N, 1)  # (N, 1)
    weighted = (cap_weights * complex_phases)  # (N, T)
    weighted_sum = weighted.sum(axis=0)  # (T,)

    # Sum of valid capital at each timestep
    valid_cap = (cap_weights * valid).sum(axis=0)  # (T,)

    r = np.full(T, np.nan)
    psi = np.full(T, np.nan)
    active_mask = n_active > 0
    r[active_mask] = np.abs(weighted_sum[active_mask]) / valid_cap[active_mask]
    psi[active_mask] = np.angle(weighted_sum[active_mask]) % (2.0 * np.pi)

    # Clip r to [0, 1] for numerical safety
    r = np.clip(r, 0.0, 1.0)

    return r, psi, n_active


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_kuramoto_sync(
    positions: np.ndarray,
    capital: np.ndarray,
    nu: float = 0.0,
    W: int = 120,
    dt_min: int = 2,
    T_max: int = 60,
) -> KuramotoResult:
    """Compute Kuramoto synchronization for a set of strategy-instrument position time series.

    Args:
        positions: (N, T) position array — each row is a strategy's position over time
        capital: (N,) capital allocated to each strategy (for weighting)
        nu: peak detection threshold in std deviations above rolling mean (default 0.0)
        W: rolling window for peak detection statistics (default 120)
        dt_min: minimum gap between peaks; closer peaks get merged (default 2)
        T_max: maximum consecutive flat days before phase becomes NaN (default 60)

    Returns:
        KuramotoResult with order parameter r(t), phases, peak times, etc.
    """
    positions = np.asarray(positions, dtype=np.float64)
    capital = np.asarray(capital, dtype=np.float64)

    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    N, T = positions.shape
    assert capital.shape == (N,), f"capital shape {capital.shape} != ({N},)"

    flows, phi = _compute_flows(positions, capital)

    if _USE_ACCEL and is_accel_enabled():
        phases_arr, peak_counts, peak_buf = extract_all_phases_njit(
            flows, phi, positions, nu, W, dt_min, T_max
        )
        # Unpack peak_buf into list[list[int]]
        peak_times: list[list[int]] = []
        for i in range(N):
            cnt = int(peak_counts[i])
            peak_times.append([int(peak_buf[i, j]) for j in range(cnt)])
    else:
        # Pure-Python fallback
        phases_arr = np.full((N, T), np.nan)
        peak_times = []
        for i in range(N):
            peaks = _detect_flux_peaks(flows[i], phi[i], nu, W)
            peaks = _merge_close_peaks(peaks, phi[i], dt_min)
            phases_arr[i] = _interpolate_phase(peaks, T, positions[i], flows[i], T_max)
            peak_times.append(peaks)

    r, psi, n_active = _compute_order_parameter(phases_arr, capital)

    return KuramotoResult(
        r=r,
        psi=psi,
        n_active=n_active,
        phases=phases_arr,
        peak_times=peak_times,
        params={'nu': nu, 'W': W, 'dt_min': dt_min, 'T_max': T_max},
    )


def compute_kuramoto_sync_rolling(
    positions: np.ndarray,
    capital: np.ndarray,
    window: int = 252,
    step: int = 21,
    **kwargs,
) -> list[KuramotoResult]:
    """Compute Kuramoto sync over rolling windows.

    Args:
        positions: (N, T) position array
        capital: (N,) capital per strategy
        window: rolling window size in timesteps (default 252 = ~1 year daily)
        step: step size between windows (default 21 = ~1 month daily)
        **kwargs: passed to compute_kuramoto_sync (nu, W, dt_min, T_max)

    Returns:
        List of KuramotoResult, one per window.
    """
    positions = np.asarray(positions, dtype=np.float64)
    capital = np.asarray(capital, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)

    N, T = positions.shape
    results = []
    start = 0
    while start + window <= T:
        pos_window = positions[:, start:start + window]
        res = compute_kuramoto_sync(pos_window, capital, **kwargs)
        results.append(res)
        start += step

    return results
