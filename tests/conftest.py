"""Shared pytest configuration.

Two jobs:

1. Set ``TEST_MODE`` once for the whole session. ``src/config/settings.py``
   reads it at import time to redirect every data path at the tiny fixture
   dataset, so it has to be set before the first ``src.*`` import — hence at
   module scope here rather than in a fixture. Previously
   ``test_e2e_pipeline.py`` and ``test_golden.py`` each carried their own
   session-scoped autouse fixture that did ``del os.environ["TEST_MODE"]`` on
   teardown; running both in one pytest session made the second deletion raise
   ``KeyError``. CI never caught it because it invokes pytest once per
   directory.

2. Skip the data-dependent tests when ``tests/fixtures/raw/`` is absent. Those
   fixtures are derived from licensed NEED microdata and Ordnance Survey
   boundaries by ``tests/generate_fixtures.py``, so they cannot be
   redistributed; a fresh clone will not have them. Skipping keeps ``pytest``
   green out of the box while still running these tests for anyone who holds
   the inputs.
"""
import functools
import os
import shutil
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("TEST_MODE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURE_RAW_DIR = BASE_DIR / "tests" / "fixtures" / "raw"
FIXTURE_SENTINELS = (
    FIXTURE_RAW_DIR / "energy" / "need_2024_official_50k.csv",
    FIXTURE_RAW_DIR / "physics" / "lhs_results_combined.csv",
)

NO_FIXTURES_REASON = (
    "tests/fixtures/raw/ is absent. These inputs are derived from licensed data "
    "and are not redistributable - run tests/generate_fixtures.py once you have "
    "populated data/raw/. See README.md."
)

NO_R_INLA_REASON = (
    "R with the INLA package is not installed. The INLA component test drives a "
    "real Rscript subprocess. See README.md for installation."
)


def fixtures_available() -> bool:
    """Return true only when the complete fixture set needed by integration tests exists.

    A checkout can contain an ignored or partially-created ``raw`` directory
    without containing either licensed sentinel input. Treating any directory
    entry as sufficient caused CI to run data-dependent tests and fail with a
    misleading FileNotFoundError.
    """
    return all(path.is_file() for path in FIXTURE_SENTINELS)


@functools.lru_cache(maxsize=1)
def r_inla_available() -> bool:
    """True only if Rscript exists AND can load INLA.

    Checking for the Rscript binary alone is not enough: INLA is not on CRAN
    and is absent from a default R install, which is exactly the situation on a
    stock CI runner.
    """
    if shutil.which("Rscript") is None:
        return False
    try:
        return subprocess.run(
            ["Rscript", "-e", "library(INLA)"],
            capture_output=True,
            timeout=120,
        ).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_fixtures: needs tests/fixtures/raw/, built from licensed inputs",
    )
    config.addinivalue_line(
        "markers",
        "requires_r_inla: needs a working R installation with the INLA package",
    )


def pytest_collection_modifyitems(config, items):
    skip_fixtures = None if fixtures_available() else pytest.mark.skip(
        reason=NO_FIXTURES_REASON
    )

    # Only probe for R if something actually needs it - the probe starts an R
    # process, so it should not run during a collection that never touches INLA.
    skip_r = None
    if any("requires_r_inla" in item.keywords for item in items):
        if not r_inla_available():
            skip_r = pytest.mark.skip(reason=NO_R_INLA_REASON)

    for item in items:
        if skip_fixtures is not None and "requires_fixtures" in item.keywords:
            item.add_marker(skip_fixtures)
        if skip_r is not None and "requires_r_inla" in item.keywords:
            item.add_marker(skip_r)
