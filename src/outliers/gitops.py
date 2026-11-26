"""Commit (and optionally push) the day's report files."""

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _git(repo: Path, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    # Never prompt for credentials: the scheduled run has no terminal to answer.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


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
            res = _git(repo, "push", timeout=120)
            if res.returncode:
                log.error("git push failed: %s", res.stderr.strip())
                return False
            log.info("Pushed to %s", _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip())
        return True
    except subprocess.TimeoutExpired:
        log.error("git push timed out; the commit is saved locally")
        return False
    except OSError as e:
        log.error("git unavailable: %s", e)
        return False
