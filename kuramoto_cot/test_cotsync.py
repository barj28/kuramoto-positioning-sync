"""Tests of the causal phase and order parameter used by the CFTC synchronization test."""
import numpy as np
import pandas as pd
import pytest

from research.kuramoto_cot import cotsync as cs


def _weekly_index(n):
    return pd.date_range("2006-06-13", periods=n, freq="7D")


def test_phase_plane_is_causal():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.standard_normal(400)) * 0.01, index=_weekly_index(400))
    base = cs.phase_plane(x)
    altered = x.copy()
    altered.iloc[300:] += rng.standard_normal(100)
    np.testing.assert_array_equal(base.iloc[:300].to_numpy(), cs.phase_plane(altered).iloc[:300].to_numpy())


def test_phase_plane_recovers_sinusoid_phase():
    t = np.arange(600)
    omega = 2 * np.pi / 52
    x = pd.Series(np.cos(omega * t), index=_weekly_index(600))
    th = cs.phase_plane(x).to_numpy()
    true = (omega * t) % (2 * np.pi)
    ok = ~np.isnan(th)
    assert ok.sum() > 400
    plv = np.abs(np.mean(np.exp(1j * (th[ok] - true[ok]))))
    assert plv > 0.97


def test_phase_plane_convention_long_is_zero_selling_is_half_pi():
    t = np.arange(600)
    omega = 2 * np.pi / 52
    x = pd.Series(np.cos(omega * t), index=_weekly_index(600))
    th = cs.phase_plane(x)
    at_peak = th.iloc[520]            # cos peak at t = 520 (= 10 periods)
    assert min(at_peak, 2 * np.pi - at_peak) < 0.25
    after = th.iloc[533]              # quarter period later: selling down from the long
    assert abs(after - np.pi / 2) < 0.25


def test_causal_sfb_is_causal():
    rng = np.random.default_rng(1)
    x = pd.Series(np.sin(np.arange(500) * 2 * np.pi / 40) + 0.1 * rng.standard_normal(500),
                  index=_weekly_index(500))
    base = cs.causal_sfb(x)
    altered = x.copy()
    altered.iloc[350:] = -altered.iloc[350:]
    np.testing.assert_array_equal(base.iloc[:350].to_numpy(), cs.causal_sfb(altered).iloc[:350].to_numpy())


def test_order_parameter_identical_and_opposite():
    idx = _weekly_index(3)
    same = pd.DataFrame({"a": [0.3, 1.0, 2.0], "b": [0.3, 1.0, 2.0]}, index=idx)
    np.testing.assert_allclose(cs.order_parameter(same), 1.0)
    opp = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": [np.pi, 1.0 + np.pi, 2.0]}, index=idx)
    r = cs.order_parameter(opp)
    np.testing.assert_allclose(r.iloc[:2], 0.0, atol=1e-12)
    assert np.isnan(r.iloc[2])  # a single valid market is not an order parameter


def test_circular_shift_keeps_each_series_and_breaks_alignment():
    rng = np.random.default_rng(2)
    th = pd.DataFrame(np.tile(rng.uniform(0, 2 * np.pi, 300)[:, None], (1, 20)), index=_weekly_index(300))
    shifted = cs.circular_shift(th, rng)
    for c in th.columns:
        assert sorted(th[c].dropna()) == sorted(shifted[c].dropna())
    assert cs.order_parameter(th).mean() == pytest.approx(1.0)
    assert cs.order_parameter(shifted).mean() < 0.5


def test_entry_date_is_monday_after_release():
    report = pd.Timestamp("2020-03-10")              # Tuesday
    days = pd.bdate_range("2020-03-09", "2020-03-31")
    assert cs.entry_date(report, days) == pd.Timestamp("2020-03-16")


def test_shutdown_weeks_flagged():
    assert cs.in_shutdown(pd.Timestamp("2019-01-08"))
    assert cs.in_shutdown(pd.Timestamp("2025-10-14"))
    assert not cs.in_shutdown(pd.Timestamp("2019-03-12"))
