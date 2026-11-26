import subprocess
import time

import pytest

from outliers.config import load_config
from outliers.gitops import commit_reports
from outliers.providers.nba_api_provider import run_with_deadline


def test_overrides_parse_types():
    cfg = load_config(overrides=['baseline.method="exp_decay"', "baseline.seasons_back=1",
                                 "general.min_score=2.5", "baseline.distribution=normal"])
    assert cfg["baseline"]["method"] == "exp_decay"
    assert cfg["baseline"]["seasons_back"] == 1
    assert cfg["general"]["min_score"] == 2.5
    assert cfg["baseline"]["distribution"] == "normal"


def test_commit_only_report_files(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], capture_output=True, text=True, check=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "other.txt").write_text("staged but unrelated")
    git("add", "other.txt")
    report = tmp_path / "reports" / "2026-11-23.md"
    report.parent.mkdir()
    report.write_text("# report")
    assert commit_reports(tmp_path, [report], "Outliers for 2026-11-23", push=False)
    committed = git("show", "--name-only", "--format=", "HEAD").stdout.split()
    assert committed == ["reports/2026-11-23.md"]
    # Rerunning with identical content is a no-op, not an error.
    assert commit_reports(tmp_path, [report], "Outliers for 2026-11-23", push=False)


def test_run_with_deadline_abandons_stalled_call():
    assert run_with_deadline(lambda: 42, 1) == 42
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        run_with_deadline(lambda: time.sleep(5), 0.2)
    assert time.monotonic() - start < 1
    with pytest.raises(ValueError):
        run_with_deadline(lambda: int("x"), 1)
