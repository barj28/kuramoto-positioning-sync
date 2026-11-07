# Confirmatory pre-registration: synchronization of speculative positioning, held-out tests

To be deposited on OSF before any confirmatory outcome is computed. The code
that runs these tests (`confirm.py`) is committed with this file and refuses to
run until `OSF_REGISTRATION.txt` holds the registration URL.

## 1. Where the hypotheses come from

The first pre-registered study (`PREREG.md`) found that the order parameter of
32 futures markets predicts nothing pooled. Exploratory analyses after a
hostile review (`exploratory/revision.py`, `exploratory/revision_controls.py`,
outputs in `results_revision/`) found two sub-universe effects that survive
the registered controls plus aggregate positioning (|mean z| and mean z):

- Currencies (8 CME currency futures, leveraged funds): higher r precedes
  deeper 4-week drawdowns of the leveraged-fund book, -0.24 SD of the outcome
  (NW t = -3.47).
- Commodities (16 markets, managed money): higher r precedes higher 4-week
  returns of the managed-money book, +0.13 SD (NW t = 2.14).

Every one of these numbers comes from data already seen. They are hypotheses,
and the tests below use only outcomes that no analysis has looked at.

## 2. Held-out data

- Time hold-out: report dates whose entry date (first trading day on or after
  the Tuesday as-of date + 6 days) falls from 2026-03-02 to 2028-12-29. No
  price after 2026-02-24 has been loaded by any analysis of this project. The
  CFTC positions to 2026-11-03 were loaded and r(t) to that date was plotted;
  r uses positions only, never prices or returns.
- Cross-sectional hold-out: the six commodity markets that the first study's
  coverage filter excluded and whose returns no analysis has examined: cocoa
  (CC), cotton (CT), coffee (KC), sugar (SB), KC wheat (KE), palladium (PA).
  Their daily prices exist locally (CC from 2022-04, CT/KC/SB from 2021-04,
  KE from 2013-12, PA from 2022-12, all to 2026-02-24). Some of these markets
  were part of the strategy universe of the earlier rejected paper, which used
  synthetic price signals and no CFTC data.
- Prices for the time hold-out: daily back-adjusted (additive) continuous
  futures built with the same vendor (Databento CME) and construction as the
  files of the first study. The source actually used is recorded with the
  results.

## 3. Definitions (unchanged from PREREG.md unless stated)

Phase theta = atan2(-d, z) with 156-week trailing windows; r(t) the order
parameter over the stated market set; entry date, shutdown exclusion,
volatility scaling, outcomes Y1 (4-week return of the sign-of-position book)
and Y2 (its maximum drawdown), and the five registered controls as in
PREREG.md and `run_prereg.weekly_rows`. Two controls are added to every
regression: |mean z| and mean z over the markets of the book.

## 4. Hypotheses

- CH1 (currency crowding). Markets: the 8 currency futures of the first study.
  r from those 8. Outcome Y2 of the leveraged-fund book on those 8. Sample:
  time hold-out. Prediction: b < 0.
- CH2a (commodity continuation, new markets). r from the 16 commodity markets
  of the first study. Outcome Y1 of the managed-money book on the 6 held-out
  commodity markets, controls computed on those 6. Sample: every week with at
  least 5 of the 6 markets priced, entries to 2026-01-31. Prediction: b > 0.
- CH2b (commodity continuation, new period). r and outcome Y1 on the 16
  commodity markets. Sample: time hold-out. Prediction: b > 0.

## 5. Test and decision

OLS of the outcome on r and the seven controls (all standardized on the test
sample). One-sided Newey-West t (8 lags) on b. Each hypothesis is tested at
alpha = 0.05 / 3 (Bonferroni, one-sided). A stationary-bootstrap p (mean
block 26 weeks, 5,000 draws) is reported alongside, not used for the decision.
Effects are reported in SD of the outcome with 95% intervals.

CH2a can be run as soon as this registration is public. CH1 and CH2b are run
once, after 2029-02-01, on the full time hold-out; no interim look.

## 6. Power, stated before the tests

From the in-sample minimum detectable effects (80% power, two-sided 5%):
0.148 SD for currencies and 0.155 SD for commodities at 770 weeks.
Scaling the standard error by the square root of the sample size, at the
one-sided Bonferroni threshold z = 2.128:

- CH1, about 145 weeks: SE 0.122 SD. Power 44% at the in-sample effect
  (0.24 SD), 13% at half of it.
- CH2a, about 190 weeks: SE 0.111 SD. Power 17% at 0.13 SD, 30% at 0.18 SD.
- CH2b, about 145 weeks: SE 0.128 SD. Power 13% at 0.13 SD, 24% at 0.18 SD.

These tests can therefore confirm a large effect, but a failure to reject is
weak evidence of absence. Results are reported with their intervals whatever
they are.
