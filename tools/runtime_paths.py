# ==========================================================================
# Master Thesis - Runtime Data Path Helpers
#   - André Filipe Gomes Silvestre, 20240502
#
#   Writable runtime path helpers for generated transport databases.
# ==========================================================================

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Iterable


HOSTED_SPACE_ENV_VARS = ("SPACE_ID", "HF_SPACE_ID", "SPACE_HOST")


def is_hosted_space_runtime() -> bool:
    """Return whether the app is running inside a hosted Hugging Face Space."""
    return any(os.getenv(env_name) for env_name in HOSTED_SPACE_ENV_VARS)


def get_runtime_data_root() -> Path:
    """Return the writable runtime data root.

    Local development keeps using the repository data directory by default.
    Hosted Spaces use ``/tmp`` unless ``LISBOA_RUNTIME_DATA_DIR`` overrides it,
    because image files under ``/app`` can be read-only for the runtime user.
    """
    explicit_root = os.getenv("LISBOA_RUNTIME_DATA_DIR")
    if explicit_root:
        return Path(explicit_root).expanduser()

    if is_hosted_space_runtime():
        return Path(tempfile.gettempdir()) / "lisboa_runtime"

    return Path(__file__).resolve().parent.parent / "data"


def resolve_runtime_data_dir(source_dir: Path, relative_subdir: str) -> Path:
    """Resolve the writable data directory for a generated dataset."""
    runtime_root = get_runtime_data_root()
    repository_data_root = Path(__file__).resolve().parent.parent / "data"

    if runtime_root.resolve() == repository_data_root.resolve():
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
