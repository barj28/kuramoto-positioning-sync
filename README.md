# Phase-Locking Without Prediction: replication package

Code, pre-registrations and results for *Synchronized speculative positioning
in futures markets: a pre-registered null and exploratory sub-market effects*
(A. Kathiravelu, 2026). The manuscript is `kuramoto_cot/note/note.pdf`; its
first version (commit "note 'Phase-Locking Without Prediction'") is kept in the
history together with the review that led to the current one.

## What is here

| Path | Content |
|---|---|
| `kuramoto_cot/PREREG.md` | Pre-registration (hypotheses, tests, decision rule) and its dated amendment |
| `kuramoto_cot/cotsync.py` | CFTC loading, market mapping, point-in-time rules, causal phases, order parameter, surrogates |
| `kuramoto_cot/stats.py` | Stationary bootstrap, Newey-West, Romano-Wolf step-down, Clark-West |
| `kuramoto_cot/run_prereg.py` | Runs H1, H2, H3 exactly as registered |
| `kuramoto_cot/figures.py` | Figures of the note |
| `kuramoto_cot/exploratory/` | Not registered: look-ahead Monte Carlo, Corwin-Schultz simulation, and the original event-anchored phase (`original_sfb.py`, the object under study, kept verbatim) |
| `kuramoto_cot/results/` | Registered run (32 markets): `results.json`, weekly rows, order-parameter series |
| `kuramoto_cot/results_run1_commodities_only/` | First run, affected by a loading bug (see PREREG amendment 1), reported as it came out |
| `kuramoto_cot/results_exploratory/` | Outputs of the exploratory scripts |
| `kuramoto_cot/test_*.py` | Tests (causality of both phases, phase convention, surrogates, PIT entry dates, inference) |

## Timeline and what the timestamps prove

The commits are in this repository's history with their original dates
(Europe/Zurich, UTC+1, 7 November 2026). The pre-registration commit precedes
the first outcome computation, and the analysis code was committed before each
run. These commit dates are self-reported: the first external timestamp is the
publication of this repository and its archived release, after the analysis.
The repository was extracted from a private research repository with
`git subtree split`, which rewrites commit hashes but keeps dates and content;
`PREREG.md` and `results.json` cite the private hashes, mapped below.

| Private hash | Public hash | Time (CET) | Step |
|---|---|---|---|
| `676c300` | `e629034` | 20:50:52 | Pre-registration frozen |
| `96bb2e3` | `742fce4` | 20:54:26 | Phases, order parameter, mapping, PIT rules, tests |
| `7fb9db6` | `5932446` | 20:56:29 | Inference and runner, before the first run |
| `fd61875` | `e675468` | 20:58:26 | Loader fix and amendment 1, before run 2 |
| `9558210` | `711a8ca` | 21:01:59 | Results of run 2 |
| `564cff4` | `af4d032` | 22:16:47 | Post-review results, confirmatory pre-registration (`CONFIRM_PREREG.md`) and its locked runner |

`CONFIRM_PREREG.md` is registered on OSF (link in the manuscript); its tests
use only held-out outcomes and `confirm.py` refuses to run before the OSF link
is recorded in `kuramoto_cot/OSF_REGISTRATION.txt`.

## Reproduce

Python 3.11+ with the packages in `requirements.txt`. From the repository root:

```
python -m pytest -q kuramoto_cot
python -m kuramoto_cot.fetch_cftc --out cftc
python -m kuramoto_cot.run_prereg --cftc cftc --prices <price dir> --out out
python -m kuramoto_cot.exploratory.lookahead_sim --reps 200 --out out/lookahead_sim.json
python kuramoto_cot/exploratory/cs_mechanical.py
```

The simulations need no data. The registered run needs daily futures prices,
see `DATA.md`.

## License

Code under the MIT license (`LICENSE`). The manuscript is the author's.
