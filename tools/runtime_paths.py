# ==========================================================================
# Master Thesis - Runtime Data Path Helpers
#   - André Filipe Gomes Silvestre, 20240502
#
#   Hosted-deployment detection and writable runtime path helpers for
#   generated databases.
# ==========================================================================

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

# Explicit override: "true" or "false" wins over every detected signal.
HOSTED_OVERRIDE_ENV_VAR = "LISBOA_HOSTED"
# Hugging Face sets these in every Space container.
HOSTED_SPACE_ENV_VARS = ("SPACE_ID", "HF_SPACE_ID", "SPACE_HOST")
# Streamlit Community Cloud clones the repository under /mount/src/<repo>.
# The path is not a documented API, so deployments also set LISBOA_HOSTED.
COMMUNITY_CLOUD_SOURCE_ROOT = Path("/mount/src")

_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
_FALSE_VALUES = {"false", "0", "no", "n", "off"}


def is_hosted_deployment() -> bool:
    """Return whether LISBOA runs as a public hosted deployment.

    Hosted deployments refuse private visitor model hosts, keep generated data
    in a temporary directory and refresh release-backed data while running.

    Returns:
        bool: ``LISBOA_HOSTED`` when it is set to a boolean value; otherwise
        True on a Hugging Face Space or Streamlit Community Cloud.
    """
    override = os.getenv(HOSTED_OVERRIDE_ENV_VAR, "").strip().lower()
    if override in _TRUE_VALUES:
        return True
    if override in _FALSE_VALUES:
        return False
    if any(os.getenv(env_name) for env_name in HOSTED_SPACE_ENV_VARS):
        return True
    return COMMUNITY_CLOUD_SOURCE_ROOT in REPO_ROOT.parents


def get_runtime_data_root() -> Path:
    """Return the writable runtime data root.

    Local development keeps using the repository data directory by default.
    Hosted deployments use the system temporary directory unless
    ``LISBOA_RUNTIME_DATA_DIR`` overrides it, because the application files can
    be read-only for the runtime user and are replaced by redeploys.
    """
    explicit_root = os.getenv("LISBOA_RUNTIME_DATA_DIR")
    if explicit_root:
        return Path(explicit_root).expanduser()

    if is_hosted_deployment():
        return Path(tempfile.gettempdir()) / "lisboa_runtime"

    return REPO_ROOT / "data"


def resolve_runtime_data_dir(source_dir: Path, relative_subdir: str) -> Path:
    """Resolve the writable data directory for a generated dataset."""
    runtime_root = get_runtime_data_root()

    if runtime_root.resolve() == (REPO_ROOT / "data").resolve():
        return source_dir

    return runtime_root / relative_subdir


def seed_runtime_data_dir(source_dir: Path, target_dir: Path, filenames: Iterable[str]) -> None:
    """Copy existing repository seed files into a writable runtime directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() == target_dir.resolve():
        return

    for filename in filenames:
        source = source_dir / filename
        target = target_dir / filename
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)
