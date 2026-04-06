# ===========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Sequential cross-provider consistency runner for the LISBOA multi-agent
# assistant. It compares whether different response models preserve the same
# architectural presentation contract, even when wording differs.
#
# ===========================================================================

# Required libraries:
# pip install python-dotenv

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import MultiAgentAssistant
from config import Config
from eval.validators.response_heuristics import (
    compare_response_contracts,
    run_all_heuristics,
)

DEFAULT_PROVIDER_SEQUENCE = ["azure", "lmstudio"]
SUPPORTED_PROVIDERS = {"azure", "openai", "lmstudio"}
RESULTS_DIR = PROJECT_ROOT / "eval" / "results" / "consistency"

CONSISTENCY_PROMPTS = [
    {
        "id": "C01",
        "query": "How is the weather in Lisbon today?",
        "language": "en",
        "category": "weather",
    },
    {
        "id": "C02",
        "query": "Como vou do Rossio para Belém?",
        "language": "pt",
        "category": "transport",
    },
    {
        "id": "C03",
        "query": "Planeia a minha tarde em Belém, diz-me como lá chegar a partir do Rossio e considera o tempo.",
        "language": "pt",
        "category": "planner",
    },
]


@contextmanager
def temporary_model_provider(provider: str):
    """Temporarily switch the active provider family for assistant instantiation."""
    normalized = str(provider or "").strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider}'. Expected one of: {sorted(SUPPORTED_PROVIDERS)}")

    original_provider = Config.MODEL_PROVIDER
    Config.MODEL_PROVIDER = normalized
    try:
        yield normalized
    finally:
        Config.MODEL_PROVIDER = original_provider


def provider_ready(provider: str) -> tuple[bool, str | None]:
    """Check whether the minimum configuration exists for the provider."""
    normalized = str(provider or "").strip().lower()
    if normalized == "azure":
        if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
            return False, "Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT"
        return True, None
    if normalized == "openai":
        if not Config.OPENAI_API_KEY:
            return False, "Missing OPENAI_API_KEY"
        return True, None
    if normalized == "lmstudio":
        if not Config.LMSTUDIO_BASE_URL or not Config.LMSTUDIO_MODEL_NAME:
            return False, "Missing LM Studio base URL or model name"
        return True, None
    return False, f"Unsupported provider '{provider}'"


def run_single_prompt(provider: str, prompt: str, language: str) -> dict[str, Any]:
    """Run one prompt through the full assistant for a single provider."""
    with temporary_model_provider(provider):
        assistant = MultiAgentAssistant()
        assistant.reset()
        start_time = time.time()
        response = assistant.chat(prompt, verbose=False, language=language)
        elapsed = time.time() - start_time

        return {
            "provider": provider,
            "model_name": assistant.model_name,
            "latency_s": round(elapsed, 2),
            "response": response,
            "heuristics": run_all_heuristics(response, expected_language=language),
        }


def select_prompts(limit: int | None, category: str | None, prompt: str | None, language: str | None) -> list[dict[str, Any]]:
    """Resolve the prompt set for this consistency run."""
    if prompt:
        return [{
            "id": "CUSTOM",
            "query": prompt,
            "language": language or "en",
            "category": category or "custom",
        }]

    prompts = CONSISTENCY_PROMPTS
    if category:
        prompts = [item for item in prompts if item["category"] == category]
    if limit is not None:
        prompts = prompts[:limit]
    return deepcopy(prompts)


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an aggregate summary for the consistency run."""
    total_runs = len(records)
    completed_runs = [item for item in records if item.get("comparison")]
    consistent_runs = [item for item in completed_runs if item["comparison"].get("consistent", False)]

    per_provider: dict[str, dict[str, Any]] = {}
    for item in records:
        for result in item.get("results", []):
            provider = result["provider"]
            bucket = per_provider.setdefault(provider, {
                "count": 0,
                "avg_latency_s": 0.0,
                "overall_pass_rate": 0.0,
            })
            bucket["count"] += 1
            bucket["avg_latency_s"] += float(result.get("latency_s", 0.0) or 0.0)
            if (result.get("heuristics") or {}).get("overall_pass", False):
                bucket["overall_pass_rate"] += 1

    for provider, bucket in per_provider.items():
        count = max(int(bucket["count"]), 1)
        bucket["avg_latency_s"] = round(bucket["avg_latency_s"] / count, 2)
        bucket["overall_pass_rate"] = round(bucket["overall_pass_rate"] / count, 3)

    return {
        "total_prompts": total_runs,
        "completed_comparisons": len(completed_runs),
        "consistent_comparisons": len(consistent_runs),
        "consistency_rate": round(len(consistent_runs) / max(len(completed_runs), 1), 3) if completed_runs else 0.0,
        "per_provider": per_provider,
    }


def persist_results(payload: dict[str, Any]) -> Path:
    """Persist the consistency artefact to eval/results/consistency/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"provider_consistency_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sequential cross-provider consistency checks for the LISBOA assistant.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="Repeatable provider override. Defaults to azure then lmstudio.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected prompts.")
    parser.add_argument("--category", type=str, default=None, help="Filter built-in prompts by category.")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single custom prompt instead of the built-in set.")
    parser.add_argument("--language", type=str, default=None, help="Language for a custom prompt.")
    args = parser.parse_args()

    providers = [item.lower().strip() for item in (args.providers or DEFAULT_PROVIDER_SEQUENCE)]
    if len(providers) < 2:
        raise SystemExit("At least two providers are required for a consistency comparison.")

    for provider in providers:
        ready, reason = provider_ready(provider)
        if not ready:
            raise SystemExit(f"Provider '{provider}' is not ready: {reason}")

    prompts = select_prompts(args.limit, args.category, args.prompt, args.language)
    if not prompts:
        raise SystemExit("No prompts selected for the consistency run.")

    print("=" * 70)
    print("SEQUENTIAL PROVIDER CONSISTENCY CHECK")
    print("=" * 70)
    print(f"Providers: {providers}")
    print(f"Prompts:   {len(prompts)}")

    records: list[dict[str, Any]] = []
    for prompt_index, prompt_item in enumerate(prompts, start=1):
        print("\n" + "-" * 70)
        print(f"[{prompt_index}/{len(prompts)}] {prompt_item['id']} | {prompt_item['category']} | {prompt_item['language'].upper()}")
        print(f"Query: {prompt_item['query']}")

        run_results: list[dict[str, Any]] = []
        for provider in providers:
            print(f"   -> Running {provider} sequentially...")
            try:
                result = run_single_prompt(
                    provider=provider,
                    prompt=prompt_item["query"],
                    language=prompt_item["language"],
                )
            except Exception as exc:
                result = {
                    "provider": provider,
                    "model_name": None,
                    "latency_s": None,
                    "response": None,
                    "heuristics": None,
                    "error": str(exc),
                }
                print(f"      [ERROR] {provider}: {exc}")
            else:
                overall_pass = (result.get("heuristics") or {}).get("overall_pass", False)
                print(
                    f"      [OK] {provider} | model={result['model_name']} | latency={result['latency_s']:.2f}s | heuristics_pass={overall_pass}"
                )
            run_results.append(result)

        comparison = None
        baseline = next((item for item in run_results if item.get("response")), None)
        if baseline:
            comparisons = []
            for candidate in run_results:
                if candidate is baseline or not candidate.get("response"):
                    continue
                contract_comparison = compare_response_contracts(
                    baseline["response"],
                    candidate["response"],
                )
                comparisons.append({
                    "reference_provider": baseline["provider"],
                    "candidate_provider": candidate["provider"],
                    **contract_comparison,
                })
                print(
                    f"      [COMPARE] {baseline['provider']} vs {candidate['provider']} | consistent={contract_comparison['consistent']} | issues={contract_comparison['issues']}"
                )

            if comparisons:
                comparison = {
                    "consistent": all(item["consistent"] for item in comparisons),
                    "details": comparisons,
                }

        records.append({
            "id": prompt_item["id"],
            "query": prompt_item["query"],
            "language": prompt_item["language"],
            "category": prompt_item["category"],
            "results": run_results,
            "comparison": comparison,
        })

    summary = build_summary(records)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "providers": providers,
        "summary": summary,
        "records": records,
    }
    output_path = persist_results(payload)

    print("\n" + "=" * 70)
    print("CONSISTENCY SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved artefact to: {output_path}")

    return 0 if summary["consistency_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
