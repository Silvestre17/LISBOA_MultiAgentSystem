# ==========================================================================
# Master Thesis - Hugging Face Space Entrypoint
#   - André Filipe Gomes Silvestre, 20240502
#
#   Starts the Docker Space after preloading shared LISBOA runtime resources.
#   Features:
#     - Warms release-backed vector DB and transport caches before first user
#     - Keeps warmed Python singletons alive by launching Streamlit in-process
#     - Falls back to the standard Streamlit CLI if the bootstrap API changes
# ==========================================================================

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app.py"
DEFAULT_STREAMLIT_PORT = 8501


def _ensure_repo_import_path() -> None:
    """Make repository packages importable when this script runs from /app/scripts."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a permissive boolean value from environment variables."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _streamlit_port() -> int:
    """Return the Streamlit port configured for the hosted container."""
    raw_port = os.getenv("PORT") or os.getenv("STREAMLIT_SERVER_PORT")
    if not raw_port:
        return DEFAULT_STREAMLIT_PORT

    try:
        return int(raw_port)
    except ValueError:
        print(
            f"⚠️ Invalid Streamlit port '{raw_port}', using {DEFAULT_STREAMLIT_PORT}.",
            flush=True,
        )
        return DEFAULT_STREAMLIT_PORT


def _build_streamlit_flag_options() -> dict[str, Any]:
    """Build Streamlit bootstrap options matching the previous Docker command."""
    return {
        "server.address": os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"),
        "server.port": _streamlit_port(),
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "server.fileWatcherType": os.getenv("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none"),
        "global.developmentMode": False,
    }


def _build_streamlit_cli_args() -> list[str]:
    """Build equivalent CLI arguments for the fallback startup path."""
    options = _build_streamlit_flag_options()
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        str(options["server.address"]),
        "--server.port",
        str(options["server.port"]),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        str(options["server.fileWatcherType"]),
        "--global.developmentMode",
        "false",
    ]


def _preload_startup_resources() -> bool:
    """Warm LISBOA shared resources before accepting Streamlit sessions."""
    if not _env_bool("LISBOA_STARTUP_PRELOAD_ENABLED", True):
        print("ℹ️ LISBOA startup preload disabled by LISBOA_STARTUP_PRELOAD_ENABLED.", flush=True)
        return True

    try:
        _ensure_repo_import_path()
        from agent.utils.startup_resources import run_startup_preload

        language = os.getenv("LISBOA_STARTUP_PRELOAD_LANGUAGE", "pt")
        status = run_startup_preload(language=language)
    except Exception:
        print("❌ LISBOA startup preload crashed before Streamlit launch.", flush=True)
        traceback.print_exc()
        return False

    if bool(status.get("ok", False)):
        print("✅ LISBOA startup preload completed before Streamlit launch.", flush=True)
        return True

    print("⚠️ LISBOA startup preload completed with degraded readiness.", flush=True)
    print(f"   Transport ready: {bool(status.get('transport_ok', False))}", flush=True)
    print(f"   KnowledgeBase ready: {bool(status.get('kb_ok', False))}", flush=True)
    return False


def _start_streamlit_in_process() -> None:
    """Start Streamlit inside the current Python process."""
    from streamlit.web import bootstrap

    os.chdir(REPO_ROOT)
    sys.argv = ["streamlit", "run", str(APP_PATH)]
    bootstrap.run(
        str(APP_PATH),
        False,
        [],
        _build_streamlit_flag_options(),
    )


def _start_streamlit_cli_fallback() -> None:
    """Replace this process with the standard Streamlit CLI command."""
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, _build_streamlit_cli_args())


def main() -> int:
    """Run the hosted Space entrypoint."""
    preload_ok = _preload_startup_resources()
    if not preload_ok and _env_bool("LISBOA_STARTUP_PRELOAD_REQUIRED", False):
        print(
            "❌ Refusing to start Streamlit because LISBOA_STARTUP_PRELOAD_REQUIRED=true.",
            flush=True,
        )
        return 1

    try:
        _start_streamlit_in_process()
    except Exception:
        print(
            "⚠️ In-process Streamlit bootstrap failed; falling back to standard CLI.",
            flush=True,
        )
        traceback.print_exc()
        _start_streamlit_cli_fallback()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
