"""Commit (and optionally push) the day's report files."""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def commit_reports(repo: Path, files: list[Path], message: str, push: bool) -> bool:
    """Commit only `files`, leaving anything else staged or modified alone.

    Never raises: a failed commit is logged and the run carries on.
    """
    rel = [str(f.relative_to(repo)) for f in files]
    try:
        add = _git(repo, "add", "--", *rel)
        if add.returncode:
            log.error("git add failed: %s", add.stderr.strip())
            return False
        if not _git(repo, "diff", "--cached", "--quiet", "--", *rel).returncode:
            log.info("Reports unchanged; nothing to commit")
            return True
        res = _git(repo, "commit", "-m", message, "--", *rel)
        if res.returncode:
            log.error("git commit failed: %s", (res.stderr or res.stdout).strip())
            return False
        log.info("Committed %s", ", ".join(rel))
        if push:
            res = _git(repo, "push")
            if res.returncode:
                log.error("git push failed: %s", res.stderr.strip())
                return False
        return True
    except OSError as e:
        log.error("git unavailable: %s", e)
        return False
