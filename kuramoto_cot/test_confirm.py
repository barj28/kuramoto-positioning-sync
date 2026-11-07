"""The confirmatory runner refuses to run before the OSF registration is recorded."""
import pandas as pd
import pytest

from . import confirm


def test_refuses_without_registration(tmp_path):
    with pytest.raises(SystemExit, match="OSF"):
        confirm.registration_url(tmp_path)


def test_refuses_a_non_osf_url(tmp_path):
    (tmp_path / "OSF_REGISTRATION.txt").write_text("https://example.com/x\n")
    with pytest.raises(SystemExit, match="OSF"):
        confirm.registration_url(tmp_path)


def test_accepts_an_osf_url(tmp_path):
    (tmp_path / "OSF_REGISTRATION.txt").write_text("https://osf.io/abcde\n")
    assert confirm.registration_url(tmp_path) == "https://osf.io/abcde"


def test_time_holdout_is_locked_before_2029_02_01():
    with pytest.raises(SystemExit, match="2029-02-01"):
        confirm.check_time_holdout_date(pd.Timestamp("2028-12-31"))
    confirm.check_time_holdout_date(pd.Timestamp("2029-02-01"))
