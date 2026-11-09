"""Make note_ssrn.tex from note.tex: replace the three link placeholders with accurate text."""
import sys
from pathlib import Path

d = Path(sys.argv[1])
s = (d / "note.tex").read_bytes().decode("utf-8")
pairs = [
    (r"(\texttt{CONFIRM\_PREREG.md}, OSF: \url{OSF_LINK})",
     r"(\texttt{CONFIRM\_PREREG.md}, to be deposited on OSF before any held-out outcome is computed)"),
    (r"are at \url{REPO_LINK} (archived as \url{DOI_LINK}).",
     r"will be released in a public repository with an archived DOI; until then they are available from the author."),
]
for a, b in pairs:
    assert a in s, a
    s = s.replace(a, b)
assert "LINK" not in s
(d / "note_ssrn.tex").write_bytes(s.encode("utf-8"))
print("ok")
