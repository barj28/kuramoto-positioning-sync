# Pre-registration: synchronization of real speculative positioning (CFTC)

Frozen before any outcome is computed. The commit that adds this file is the
timestamp. Any later change is a new, dated section at the end, never an edit.

## 0. Why this test

The March 2026 paper "Endogenous Kuramoto Synchronization in Algorithmic
Trading Flows" was rejected (JFQA, Quantitative Finance). A re-audit on
2026-11-07 found its empirical core invalid:

1. Its order parameter r(t) is built from synthetic signals computed on prices,
   not from real positions.
2. The SFB phase is not causal: `_interpolate_phase` interpolates towards the
   NEXT flux peak and `_detect_flux_peaks` reads `flows[t + 1]`, so r(t) uses
   the future when it is correlated with forward drawdowns.
3. The 16-year Corwin-Schultz beta is mechanical: a simulated market with a
   constant true spread (true beta = 0) gives beta_CS = +0.71 to +0.87 through
   the paper's estimator (paper: mean 0.73).
4. The tick beta ranking does not survive the repository's own bid-ask test
   (lagged sigma: rank correlation 0.24 with the baseline, 24/43 sign changes).

The one question never tested is the paper's actual claim: do REAL speculative
positions across futures markets phase-lock, and does that synchronization
precede crowding unwinds? CFTC reports observe speculative positions directly.

## 1. Data

- Positions: CFTC Commitments of Traders, futures only, weekly, Tuesday as-of.
  Traders in Financial Futures (TFF) for financial markets: Leveraged Funds.
  Disaggregated for commodities: Managed Money. 2006-06 to 2026-11-03.
- Net speculative position x_i(t) = (Long - Short) / Open Interest.
- Prices: daily back-adjusted (additive) continuous futures in
  `barj_backtester/data/futures/*_daily.parquet`, 2010-06-07 to 2026-02-24.
  Returns use price CHANGES in points (invariant to the additive adjustment),
  scaled by the trailing EWMA (halflife 20 trading days) volatility of daily
  price changes known at the entry date.
- Universe: every root that has both a CFTC series (mapped by contract market
  code, the mapping is committed with the code) and a daily price file, with at
  least 85% of weeks non-missing over 2009-2026. Equal weight across markets.

## 2. Point-in-time rules

- Report for Tuesday t is released Friday t+3 at 15:30 ET. Information is used
  from the close of the first trading day on or after Monday t+6.
- Report weeks affected by US government shutdowns (late or catch-up releases)
  are excluded from every predictive test: 2013-10-01 to 2013-11-30,
  2018-12-18 to 2019-03-08, 2025-09-30 to 2025-12-31.
- All standardizations use trailing windows only.

## 3. Phase and order parameter

Primary phase (causal phase plane, no peak detection, no future data):

- z_i(t) = (x_i(t) - mean_156w) / std_156w, trailing 156 weeks (min 52).
- d_i(t) = (x_i(t) - x_i(t-1)) / std_156w of weekly changes.
- theta_i(t) = atan2(-d_i(t), z_i(t)). Phase 0 = maximal long, pi/2 = selling,
  pi = maximal short, 3pi/2 = buying (harmonic-oscillator convention, matches
  the paper's "0 = long entry, pi = short entry").
- r(t) = |(1/N_t) sum_i exp(i theta_i(t))| over markets with a valid phase.

Secondary (robustness, reported, not used for the decision):

- S1 nematic order r2(t) = |mean exp(2 i theta)| (long and short extremes count
  as aligned).
- S2 causal SFB: the paper's SFB on weekly x_i with W = 26, nu = 0,
  dt_min = 2, T_max = 13, where a peak is usable only one week after it occurs
  and the phase between the last peak and the next is extrapolated at pi per
  median of the last 4 half-cycle durations, capped at the next half cycle.
- S3 sub-universes: financial only (TFF) and commodities only (Managed Money).

## 4. Hypotheses and tests

H1 (synchronization exists). Statistic: mean of r(t) over 2009-06 to 2026-11.
Null: 2,000 surrogates in which each market's phase series is circularly
shifted by an independent uniform offset (preserves each market's dynamics,
destroys cross-market alignment). One-sided p = share of surrogates with mean
r at least as large. H1 holds if p < 0.01. Reported with the ratio
observed / null mean and the i.i.d. floor sqrt(pi / 4N).

H2 (synchronization precedes unwinds). Weekly, h = 4 weeks (20 trading days)
from the entry date. Speculator book: each market weighted by
sign(x_i(t)) / N, vol-scaled returns. Outcomes:

- Y1: 4-week return of the speculator book.
- Y2: maximum drawdown of the daily speculator book within the 4 weeks.
- Y3: cross-market mean of log(forward 20-day realized vol / trailing 60-day
  realized vol).

Regression Y = a + b r(t) + c'controls + e, on weeks with entry dates in
2010-06 to 2026-01, shutdown weeks excluded. Controls: trailing 4-week book
return, trailing 60-day realized vol ratio (log 20d/60d), mean pairwise
correlation of weekly position changes over the trailing 13 weeks, PC1 share of
weekly position changes over the trailing 26 weeks, cross-market mean |z_i(t)|.
Every regressor standardized. Inference: stationary block bootstrap of the
weekly rows (mean block 26 weeks, 5,000 draws), Romano-Wolf step-down across
Y1, Y2, Y3 for the coefficient b; Newey-West (lag 8) t reported alongside.
Predicted signs from the crowding thesis: b < 0 for Y1 and Y2, b > 0 for Y3.
H2 holds if at least one outcome has Romano-Wolf adjusted p < 0.05 with the
predicted sign. A significant opposite sign is reported as a rejection of the
thesis, not as support.

H3 (out of sample). Fit the H2 regression with controls on entry dates up to
2017-12-31, predict 2018-01 to 2026-01. Clark-West test against the same model
without r(t) (nested). H3 holds if one-sided p < 0.05 on an outcome where H2
holds.

Exploratory only (no decision weight): interaction of r(t) with the
market-level liquidity elasticity, since no beta estimate currently available
is valid (section 0, points 3 and 4).

## 5. Decision rule

- H1 fails: no measurable synchronization of real speculative positions. The
  Kuramoto crowding paper is abandoned; at most a short negative note
  (non-causal SFB, mechanical CS beta, null on real positions).
- H1 holds, H2 fails: synchronization exists but carries no risk information
  beyond simple crowding measures. Descriptive note only, not a QF paper.
- H1, H2 and H3 hold: GO for a full rewrite around real positioning, with the
  corrected theory (signed coupling, Ott-Antonsen reduction).

## 6. Fixed before seeing outcomes

Phase definition, windows (156, 52, 13, 26 weeks), horizon (4 weeks), outcome
definitions, controls, sample periods, surrogate count, bootstrap settings,
correction method, thresholds. The CFTC-to-root mapping and the universe filter
are mechanical and committed with the code before the first outcome run.

## Amendment 1 (2026-11-07 21:00 CET, after run 1)

Run 1 (commit 7fb9db6, outputs `results_run1_commodities_only/`) loaded only
the 2006-2016 TFF archive: the loader pattern `fin_fut_txt_*` missed the yearly
TFF files, named `fut_fin_txt_YYYY`. Every financial market therefore failed the
85% coverage filter and the universe had 16 commodity markets instead of 32.
This is a loading bug, not a design choice: the fix adds the missing pattern
and a test that every CFTC archive name is matched. Nothing else changes.
Run 2 on the 32-market universe is the pre-registered primary result. Run 1 is
kept and reported as it came out, as the commodity-only sub-universe (it
coincides with S3 "dis").

## Amendment 2 (2026-11-07 22:05 CET): correction of a timestamp

Amendment 1 is headed "21:00 CET". That time is wrong: amendment 1 was
committed at 20:58:26 CET (commit fd61875), after run 1 had written its
outputs (20:57:15 CET). The order stated in amendment 1 (run 1, then the fix,
then run 2) is right; only the clock time in its heading is wrong. All times
in this file are local commit times recorded by the author's machine. They
are self-reported and are not an external registration.
