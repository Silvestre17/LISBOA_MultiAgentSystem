# ===========================================================================
# Master Thesis - Shared Pytest Fixtures
#   - André Filipe Gomes Silvestre, 20240502
#
# Shared fixtures for strict live coverage and prompt-manifest validation.
# These fixtures are test-only and MUST NOT affect application runtime.
# ===========================================================================

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from tools import __all__ as EXPORTED_TOOL_NAMES

COVERAGE_MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "tool_coverage_manifest.json"
STRICT_PLACEHOLDER_TOKENS = (
    "your_",
    "placeholder",
    "changeme",
    "example",
    "<your",
)
VALID_LIVE_DOMAINS = {"weather", "transport", "researcher"}
REQUIRED_DATA_PATHS = [
    Config.VECTOR_DB_DIR,
    Config.PATH_VISIT_LISBOA_EVENTS,
    Config.PATH_VISIT_LISBOA_PLACES,
    PROJECT_ROOT / "data" / "carris" / "carris.db",
    PROJECT_ROOT / "data" / "cp" / "cp_gtfs.db",
]


def _is_missing_env_value(value: str | None) -> bool:
    """Return True when an environment value is blank or still a placeholder."""
    if value is None:
        return True

    normalized = value.strip()
    if not normalized:
        return True

    lowered = normalized.lower()
    return any(token in lowered for token in STRICT_PLACEHOLDER_TOKENS)



def _required_llm_env_vars() -> list[str]:
    """Resolve the currently required LLM environment variables."""
    provider = (Config.MODEL_PROVIDER or "").strip().lower()
    if provider == "azure":
        return ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    if provider == "openai":
        return ["OPENAI_API_KEY"]
    return []


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def exported_tool_names() -> set[str]:
    """Return the authoritative exported tool registry."""
    return set(EXPORTED_TOOL_NAMES)


@pytest.fixture(scope="session")
def coverage_manifest() -> list[dict[str, Any]]:
    """Load the prompt coverage manifest used by live coverage suites."""
    with open(COVERAGE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def agent_model_configs() -> dict[str, dict[str, Any]]:
    """Return model configs for the worker agents used in live coverage."""
    models = Config.get_agent_models()
    return {
        domain: models.get(domain, Config.get_default_agent_model())
        for domain in VALID_LIVE_DOMAINS
    }


@pytest.fixture(scope="session")
def strict_live_environment() -> dict[str, Any]:
    """Validate credentials and local assets required by strict live suites.

    This fixture intentionally raises RuntimeError when the environment is not
    ready. The live coverage suite is meant to fail loudly rather than skip
    silently when credentials or local assets are missing.
    """
    required_env = {
        env_name: os.getenv(env_name)
        for env_name in [
            *_required_llm_env_vars(),
            "METRO_CONSUMER_KEY",
            "METRO_CONSUMER_SECRET",
            "TAVILY_API_KEY",
        ]
    }

    missing_env = [
        env_name
        for env_name, env_value in required_env.items()
        if _is_missing_env_value(env_value)
    ]
    missing_paths = [str(path) for path in REQUIRED_DATA_PATHS if not Path(path).exists()]

    if missing_env or missing_paths:
        details: list[str] = []
        if missing_env:
            details.append("Missing environment variables: " + ", ".join(sorted(missing_env)))
        if missing_paths:
            details.append("Missing required files/directories: " + ", ".join(sorted(missing_paths)))
        raise RuntimeError(
            "Strict live evaluation prerequisites are missing. "
            "Fix the following before running live coverage:\n- "
            + "\n- ".join(details)
        )

    return {
        "provider": Config.MODEL_PROVIDER,
        "required_env": sorted(required_env.keys()),
        "validated_paths": [str(path) for path in REQUIRED_DATA_PATHS],
    }
