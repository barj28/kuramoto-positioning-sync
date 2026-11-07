"""Tests of the pre-registered inference."""
import numpy as np

from research.kuramoto_cot import stats as st


def test_stationary_bootstrap_is_blocky_and_in_range():
    rng = np.random.default_rng(0)
    idx = st.stationary_bootstrap_indices(1000, 26, rng)
    assert idx.min() >= 0 and idx.max() < 1000
    runs = np.mean(np.diff(idx) == 1)
    assert 0.9 < runs < 1.0          # about 1 - 1/26 of steps continue a block


def _boot(y, X, B, rng):
    out = np.empty((B, X.shape[1]))
    for b in range(B):
        i = st.stationary_bootstrap_indices(len(y), 5, rng)
        out[b] = st.ols(y[i], X[i])
    return out


def test_romano_wolf_separates_real_and_null_effects():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    y_real = 0.5 * x + rng.standard_normal(n)
    y_null = rng.standard_normal(n)
    b_hat = np.array([st.ols(y_real, X)[1], st.ols(y_null, X)[1]])
    boots = np.column_stack([_boot(y_real, X, 400, rng)[:, 1], _boot(y_null, X, 400, rng)[:, 1]])
    p, _ = st.romano_wolf(b_hat, boots)
    assert p[0] < 0.01
    assert p[1] > 0.05


def test_newey_west_matches_ols_se_without_autocorrelation_at_lag_zero():
    rng = np.random.default_rng(2)
    n = 2000
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n)
    se0 = st.newey_west_se(y, X, 0)
    e = y - X @ st.ols(y, X)
    ols_se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var())
    np.testing.assert_allclose(se0, ols_se, rtol=0.1)


def test_clark_west_identical_forecasts_is_zero_and_better_model_wins():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(400)
    f = np.zeros(400)
    assert st.clark_west(y, f, f, 4)["cw_mean"] == 0.0
    signal = rng.standard_normal(400)
    y2 = signal + 0.5 * rng.standard_normal(400)
    res = st.clark_west(y2, np.zeros(400), 0.9 * signal, 4)
    assert res["p_one_sided"] < 0.01 and res["r2_oos_vs_small"] > 0.5
