# Data

## CFTC Commitments of Traders (public)

Futures-only Traders in Financial Futures (leveraged funds) and Disaggregated
(managed money) reports, weekly, June 2006 to 3 November 2026, from
<https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm>.
`python -m kuramoto_cot.fetch_cftc --out cftc` downloads and unzips the 22
archives. The market-to-contract-code mapping is `MAPPING` in
`kuramoto_cot/cotsync.py`.

## Daily futures prices (licensed, not redistributed)

Back-adjusted (additive) continuous daily futures for the 32 markets, 7 June
2010 to 24 February 2026, built from Databento CME data. The licence does not
allow redistribution. To rerun H2 and H3, supply one Parquet file per market
named `<ROOT>_daily.parquet` with columns `date` and `close`, roots as in
`MAPPING` (for example `ES`, `ZN`, `EUR.FUT`, `CL`). Only daily price changes
are used, so any additive back-adjustment gives the same results; a
ratio-adjusted series will differ slightly.

The registered weekly outcome and control rows (`results/weekly_rows.parquet`,
volatility-scaled book returns, drawdowns and volatility ratios, no prices) are
included, so the regressions can be re-estimated without the price files.
