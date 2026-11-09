"""Make note_ssrn.tex from note.tex for the SSRN revision of 6378278: replace the link
placeholders with accurate text, and add a note on this version plus an appendix listing the
claims of the March 2026 version that are withdrawn (SSRN version only).

python make_ssrn.py <note dir>
"""
import sys
from pathlib import Path

d = Path(sys.argv[1])
s = (d / "note.tex").read_bytes().decode("utf-8")

NOTE = r"""\end{frontmatter}

\noindent\textit{Note on this version.} This paper replaces the version posted
on SSRN in March 2026 under the title ``Endogenous Kuramoto Synchronization in
Algorithmic Trading Flows''. The empirical results of that version are
withdrawn; \ref{app:withdrawn} lists them with the reason for each.
"""

WITHDRAWN = r"""\section{Claims of the March 2026 version that are withdrawn}
\label{app:withdrawn}

\begin{enumerate}
\item The liquidity-elasticity spectrum across 1{,}543 instruments (1{,}500 US
equities and 43 CME futures, from Corwin--Schultz spreads), its sector
ordering, and the finding that Treasury-bill ETFs are the only sector with a
negative elasticity. With a constant true spread, so a true elasticity of zero,
the estimator returns 0.71 to 0.87 with contemporaneous volatility and 0.73 to
0.89 with the lagged 21-day volatility used for equities
(\ref{app:earlier}). The spectrum is not identified by this estimator.
\item The tick-level elasticities of 43 CME futures ($+1.49$ to $-0.56$) and
the four-regime taxonomy built on them. Replacing contemporaneous realized
volatility by its one-day lag, to remove bid-ask bounce, leaves a rank
correlation of 0.24 with the reported estimates, and 24 of the 43 signs change.
\item The order parameter of 4{,}786 synthetic strategy--instrument positions
and every result built on it: its correlation with forward drawdowns
($\rho = +0.196$), the March 2020 episode, the comparison with a
rolling-correlation benchmark (paired $t = 3.04$) and the per-instrument
cross-validations. The Signed Flux Bursts phase interpolated to the next flow
peak and therefore used future data, the strategies were functions of prices
rather than observed positions, and the drawdown correlations were tested with
i.i.d.\ $p$-values on overlapping horizons.
\item The calibration of the mean-field model ($K_0 = 7.47$, $\gamma = -0.147$,
$\delta = 8.45$): $\gamma < 0$ violates the model's own constraint
$\gamma > 0$, and $\delta = 8.45$ contradicts the square-root impact exponent
of 0.5 that the model assumes.
\item The endogenous coupling $K(r)$ and its consequences (stability regimes,
the ``liquidity black hole'', hysteresis, capacity bounds) are not tested in
this paper and are not maintained.
\end{enumerate}

\begin{thebibliography}"""

pairs = [
    (r"(\texttt{CONFIRM\_PREREG.md}, OSF: \url{OSF_LINK})",
     r"(\texttt{CONFIRM\_PREREG.md}, to be deposited on OSF before any held-out outcome is computed)"),
    (r"are at \url{REPO_LINK} (archived as \url{DOI_LINK}).",
     r"are at \url{https://github.com/barj28/kuramoto-positioning-sync}; an archived DOI will follow."),
    ("\\end{frontmatter}\n", NOTE),
    ("\\begin{thebibliography}", WITHDRAWN),
]
for a, b in pairs:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
assert "LINK" not in s
(d / "note_ssrn.tex").write_bytes(s.encode("utf-8"))
print("ok")
