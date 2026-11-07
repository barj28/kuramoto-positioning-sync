"""Download and unzip the CFTC Commitments of Traders archives used by the study.

python -m kuramoto_cot.fetch_cftc --out <dir>      (then pass <dir> as --cftc)
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.cftc.gov/files/dea/history/"


def archives(last_year: int) -> list[str]:
    return (["fin_fut_txt_2006_2016", "fut_disagg_txt_hist_2006_2016"]
            + [f"fut_fin_txt_{y}" for y in range(2017, last_year + 1)]
            + [f"fut_disagg_txt_{y}" for y in range(2017, last_year + 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--last-year", type=int, default=2026, help="2028 or later for the time hold-out")
    a = ap.parse_args()
    out = Path(a.out)
    for name in archives(a.last_year):
        with urllib.request.urlopen(BASE + name + ".zip") as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(out / name)
        print("ok", name)


if __name__ == "__main__":
    main()
