"""Share of report weeks in which leveraged funds are net short, per financial market of the
registered universe (positions only, no price). Exploratory, not registered.

python -m kuramoto_cot.exploratory.net_short --cftc <dir> --out results_revision/net_short.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import cotsync as cs

HERE = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cftc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    universe = json.loads((HERE / "results" / "results.json").read_text())["universe"]
    tff = [m for m in universe if cs.MAPPING[m][0] == "tff"]
    x = cs.load_cftc(Path(a.cftc))[tff].loc["2010-06-01":"2026-01-31"]
    res = {"sample": [str(x.index.min().date()), str(x.index.max().date())],
           "share_net_short": {m: round(float((x[m].dropna() < 0).mean()), 4) for m in tff},
           "n_weeks": {m: int(x[m].notna().sum()) for m in tff}}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
