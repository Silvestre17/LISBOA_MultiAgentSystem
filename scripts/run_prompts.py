# ===========================================================================
# Master Thesis - Multi-Agent Prompt Runner
#   - André Filipe Gomes Silvestre, 20240502
#
# Manual prompt runner for two modes:
#   1. smoke    -> End-to-end MultiAgentAssistant sanity checks
#   2. coverage -> Isolated worker-agent real-service tool coverage prompts
#
# Usage:
#   python scripts/run_prompts.py --suite smoke              # Runs end-to-end sanity prompts through the full MultiAgentAssistant
#   python scripts/run_prompts.py --suite coverage           # Runs strict worker-agent coverage prompts and checks expected tool calls
#   python scripts/run_prompts.py --suite coverage --limit 5 # Runs only the first 5 coverage prompts for a quick spot-check
#   python scripts/run_prompts.py --prompt "Como está o tempo hoje?" --language pt
#   python scripts/run_prompts.py --interactive              # Ask for one ad-hoc prompt via stdin and run it immediately
#   python scripts/run_prompts.py --suite coverage --prompt "Next train from Rossio?" --domain transport --provider azure --model gpt-5-mini
# Parameters:
#   --suite {smoke,coverage}   choose the prompt suite
#   --limit N                  run only the first N selected prompts
#   --offset N                 skip the first N selected prompts
#   --category NAME            filter by smoke category or coverage domain
#   --quiet                    hide intermediate reasoning/previews
#   --prompt TEXT              run one custom prompt instead of the built-in suite
#   --interactive              ask for one custom prompt via stdin
#   --language CODE            language hint for custom prompt runs
#   --domain NAME              worker domain for custom coverage runs
#   --provider NAME            override provider family (smoke) or worker provider (coverage)
#   --model NAME               override worker model for coverage runs
#   --temperature FLOAT        override worker temperature for coverage runs
# Notes:
#   - Run this script from the repository root using the relative path above.
#   - Avoid absolute pytest-style paths in this workspace on Windows because
#     the folder name contains `[` and `]`, which pytest can interpret as glob
#     characters.
# ===========================================================================

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

# Fix Windows console encoding for emojis without replacing pytest's capture streams.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.graph import MultiAgentAssistant
from config import Config
from eval.run_benchmark import run_isolated_agent
from tools import __all__ as EXPORTED_TOOL_NAMES

COVERAGE_MANIFEST_PATH = Path(PROJECT_ROOT) / "tests" / "fixtures" / "tool_coverage_manifest.json"
SUPPORTED_MODEL_PROVIDERS = {"azure", "openai", "lmstudio"}
SUPPORTED_COVERAGE_DOMAINS = {"weather", "transport", "researcher"}

# Each prompt is a tuple: (prompt_text, language_code, category)
SMOKE_PROMPTS = [
    
    # CRITICAL Prompts
    ("Sugere um plano para uma tarde em Belém com detalhes históricos e onde comer um pastel.", "pt", "planner"),
    ("Quais os próximos autocarros da Carris no Rossio?", "pt", "transport"),    

    # BASIC Prompts
    ("How is the weather in Lisbon today?", "en", "weather"),
    ("Will it rain this weekend in Sintra?", "en", "weather"),
    ("What is the current temperature in downtown Lisbon?", "en", "weather"),
    ("How do I get from Lisbon Airport to Rossio using the metro?", "en", "transport"),
    ("Is the 28E tram running on time right now?", "en", "transport"),
    ("Next train from Cais do Sodré to Cascais.", "en", "transport"),
    ("Bus from Marquês de Pombal to Belém Tower.", "en", "transport"),
    ("Are there any subway strikes today?", "en", "transport"),
    ("Best seafood restaurants near the Tagus river.", "en", "researcher"),
    ("Where is the nearest pharmacy to Parque das Nações?", "en", "researcher"),
    ("Museums of modern art open today.", "en", "researcher"),
    ("Cheap sushi places in Saldanha.", "en", "researcher"),
    (
        "Plan a perfect afternoon in Belém visiting the Tower, Jerónimos Monastery, and eating Pastéis de Nata. Include transport from Chiado.",
        "en",
        "planner",
    ),
    ("I want to go for a drink in Bairro Alto tonight. Any recommendations?", "en", "researcher"),
    ("Como vou do Castelo de São Jorge para Belém de autocarro? Quero evitar o metro.", "pt", "transport"),
    ("Onde estão os elétricos agora em tempo real?", "pt", "transport"),
    ("Quais as linhas de elétrico que passam na Graça?", "pt", "transport"),
    ("Próximo comboio para Sintra a partir do Rossio.", "pt", "transport"),
    ("Quero ir de Entrecampos ao Marquês.", "pt", "transport"),
    ("Sugere um passeio em Alfama com poucas subidas, estou com uma pessoa idosa.", "pt", "planner"),
    ("Quero ir jantar e depois sair à noite em Lisboa. O que recomendas?", "pt", "researcher"),
    ("Museus grátis ao domingo em Lisboa.", "pt", "researcher"),
    ("Onde posso fazer um teste Covid hoje em Lisboa?", "pt", "researcher"),
    ("Há trotinetes elétricas perto do Jardim da Estrela?", "pt", "researcher"),
    ("Quero ir de metro para a Madeira.", "pt", "edge_case"),
    (
        "Wie komme ich vom Flughafen Lissabon ins Stadtzentrum mit öffentlichen Verkehrsmitteln?",
        "de",
        "transport",
    ),
    (
        "Quel temps fait-il à Lisbonne aujourd'hui et quel est le meilleur moyen d'aller à la Tour de Belém?",
        "fr",
        "multi",
    ),
]


def _load_coverage_prompts() -> list[dict]:
    """Load the strict live real-service coverage manifest used by the coverage suite."""
    with open(COVERAGE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_agent_models_for_provider(provider: str | None) -> dict[str, dict]:
    """Return the configured agent-model map for the requested provider family."""
    normalized = (provider or "").strip().lower()
    if normalized == "azure":
        return Config.AGENT_MODELS_AZURE
    if normalized == "openai":
        return Config.AGENT_MODELS_OPENAI
    return Config.AGENT_MODELS_LMSTUDIO


@contextmanager
def _temporary_model_provider(provider: str | None):
    """Temporarily switch the configured provider family for smoke runs."""
    normalized = (provider or "").strip().lower()
    if normalized and normalized not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. Expected one of: {sorted(SUPPORTED_MODEL_PROVIDERS)}"
        )

    original_provider = Config.MODEL_PROVIDER
    if normalized:
        Config.MODEL_PROVIDER = normalized

    try:
        yield Config.MODEL_PROVIDER
    finally:
        Config.MODEL_PROVIDER = original_provider


def _resolve_custom_prompt(args) -> str | None:
    """Return a custom prompt from CLI args or stdin when requested."""
    if args.prompt:
        prompt = args.prompt.strip()
        return prompt or None
    if not args.interactive:
        return None

    try:
        prompt = input("Enter a prompt to test: ").strip()
    except EOFError:
        return None
    return prompt or None


def _clone_model_config(raw_config: object, label: str) -> dict[str, Any]:
    """Return a deep-copied model config dict or raise a clear configuration error."""
    if not isinstance(raw_config, dict):
        raise TypeError(
            f"Invalid model configuration for '{label}'. Expected a dict, got {type(raw_config).__name__}."
        )
    return deepcopy(raw_config)


def _resolve_coverage_model_config(args, domain: str) -> dict[str, Any]:
    """Build the worker model config for coverage runs, applying CLI overrides when provided."""
    provider_models = _get_agent_models_for_provider(args.provider) if args.provider else Config.get_agent_models()
    raw_model_config = provider_models.get(domain)
    if raw_model_config is None:
        model_config = _clone_model_config(Config.get_default_agent_model(), "default")
    else:
        model_config = _clone_model_config(raw_model_config, domain)

    if args.provider:
        model_config["provider"] = args.provider
    if args.model:
        model_config["model"] = args.model
    if args.temperature is not None:
        model_config["temperature"] = args.temperature

    return model_config


def _should_echo_final_smoke_response() -> bool:
    """Return whether the smoke runner should print an extra final-response block."""
    return not bool(getattr(Config, "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL", False))


def _run_custom_prompt(args) -> int:
    """Run one ad-hoc prompt through either smoke or coverage mode."""
    prompt = _resolve_custom_prompt(args)
    if not prompt:
        print("❌ No custom prompt was provided.", flush=True)
        return 1

    language = args.language or "en"
    if args.suite == "coverage":
        if not args.domain:
            print("❌ --domain is required for custom coverage prompts.", flush=True)
            return 1

        model_config = _resolve_coverage_model_config(args, args.domain)
        print("=" * 60, flush=True)
        print("🧪 CUSTOM COVERAGE PROMPT", flush=True)
        print("=" * 60, flush=True)
        print(f"📝 Domain: {args.domain} | Language: {language.upper()}", flush=True)
        print(f"🤖 Model: {model_config['provider']}::{model_config['model']}", flush=True)
        print(f"👤 USER: {prompt}", flush=True)
        print("-" * 60, flush=True)

        response, tools_used, retrieved_context, elapsed, error, _usage = run_isolated_agent(
            domain=args.domain,
            query=prompt,
            config=model_config,
        )

        print(f"🔍 Tools used: {sorted(set(tools_used))}", flush=True)
        print(f"⏱️  Latency: {elapsed:.2f}s", flush=True)
        if retrieved_context and not args.quiet:
            preview = retrieved_context[:500] + ("..." if len(retrieved_context) > 500 else "")
            print("\n📚 Retrieved context preview:", flush=True)
            print(preview, flush=True)

        if error is not None:
            print(f"❌ Error: {error}", flush=True)
            return 1

        print("\n🤖 FINAL RESPONSE:", flush=True)
        print(response, flush=True)
        return 0

    with _temporary_model_provider(args.provider):
        print("=" * 60, flush=True)
        print("🧪 CUSTOM SMOKE PROMPT", flush=True)
        print("=" * 60, flush=True)
        print("\nInitializing Multi-Agent System...", flush=True)
        try:
            assistant = MultiAgentAssistant()
        except Exception as exc:
            print(f"❌ Error initializing assistant: {exc}", flush=True)
            return 1

        print(f"✅ Model family: {Config.MODEL_PROVIDER}", flush=True)
        print(f"✅ Assistant label: {assistant.model_name}", flush=True)
        print(f"📝 Language: {language.upper()}", flush=True)
        print(f"👤 USER: {prompt}", flush=True)
        print("-" * 60, flush=True)

        try:
            assistant.reset()
            start_time = time.time()
            response = assistant.chat(prompt, verbose=not args.quiet, language=language)
            elapsed = time.time() - start_time
            if not args.quiet:
                _print_smoke_tool_trace(assistant.state.get("messages", []), response, elapsed)
            if _should_echo_final_smoke_response():
                print(f"🤖 FINAL RESPONSE ({elapsed:.2f}s):", flush=True)
                print(response, flush=True)
            return 0
        except Exception as exc:
            print(f"❌ ERROR: {exc}", flush=True)
            import traceback

            traceback.print_exc()
            return 1


def _print_smoke_tool_trace(messages, response: str, elapsed: float) -> None:
    """Print intermediate tool activity for the smoke suite."""
    print("\n\033[1;34m--- 🕵️ INTERMEDIATE STEPS & TOOLS ---\033[0m", flush=True)
    tools_used = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(
                    f"  \033[1;33m[TOOL REQUEST]\033[0m {tool_call['name']}({tool_call.get('args', {})})",
                    flush=True,
                )
                tools_used += 1
        elif msg.__class__.__name__ == "ToolMessage":
            content_str = str(msg.content).replace("\n", " ")
            content_preview = (
                content_str[:100] + "..."
                if len(content_str) > 100
                else content_str
            )
            print(f"  \033[1;32m[TOOL RESULT]\033[0m {content_preview}", flush=True)
        elif msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", "") and not getattr(msg, "tool_calls", []):
            if msg.content != response:
                agent_name = getattr(msg, "name", "AI")
                print(
                    f"  \033[1;36m[{agent_name} THOUGHT/RESPONSE]\033[0m {str(msg.content)[:100]}...",
                    flush=True,
                )

    print(
        f"  \033[1;35m[METADATA]\033[0m Tools used: {tools_used} | Latency: {elapsed:.2f}s",
        flush=True,
    )
    print("\033[1;34m---------------------------------------\033[0m\n", flush=True)


def _select_subset(items, limit: int | None, offset: int, category: str | None, category_key: str):
    """Apply category filtering plus offset/limit slicing to a prompt list."""
    if category:
        filtered = [(i, item) for i, item in enumerate(items) if item[category_key] == category]
    else:
        filtered = list(enumerate(items))

    if limit is None:
        return filtered[offset:]
    return filtered[offset : offset + limit]


def _run_smoke_suite(args) -> int:
    """Run the end-to-end smoke prompts against the full multi-agent assistant."""
    print("=" * 60, flush=True)
    print("🧪 MULTI-AGENT SYSTEM TEST SUITE (SMOKE)", flush=True)
    print("=" * 60, flush=True)

    print("\nInitializing Multi-Agent System...", flush=True)
    try:
        with _temporary_model_provider(args.provider):
            assistant = MultiAgentAssistant()
            prompts_subset = _select_subset(SMOKE_PROMPTS, args.limit, args.offset, args.category, 2)

            print(f"✅ Model: {assistant.model_name}", flush=True)
            print(f"✅ Provider family: {Config.MODEL_PROVIDER}", flush=True)
            print(f"📊 Total smoke prompts available: {len(SMOKE_PROMPTS)}", flush=True)
            print(f"📋 Running {len(prompts_subset)} smoke prompt(s)", flush=True)
            print("=" * 60, flush=True)

            results = {"success": 0, "error": 0, "total_time": 0.0}

            for idx, (original_idx, (prompt, lang, category)) in enumerate(prompts_subset, 1):
                print(f"\n\n{'=' * 60}", flush=True)
                print(f"🔶 TEST {idx}/{len(prompts_subset)} (Prompt #{original_idx + 1})", flush=True)
                print(f"📝 Category: {category} | Language: {lang.upper()}", flush=True)
                print(f"👤 USER: {prompt}", flush=True)
                print("-" * 60, flush=True)

                try:
                    assistant.reset()
                    start_time = time.time()
                    response = assistant.chat(prompt, verbose=not args.quiet, language=lang)
                    elapsed = time.time() - start_time

                    results["success"] += 1
                    results["total_time"] += elapsed

                    if not args.quiet:
                        _print_smoke_tool_trace(assistant.state.get("messages", []), response, elapsed)

                    if _should_echo_final_smoke_response():
                        print("-" * 60, flush=True)
                        print(f"🤖 \033[1mFINAL AI RESPONSE\033[0m ({elapsed:.2f}s):", flush=True)
                        print(response, flush=True)
                    print("=" * 60, flush=True)
                except Exception as exc:
                    results["error"] += 1
                    print(f"❌ ERROR in Test {idx}: {exc}", flush=True)
                    import traceback

                    traceback.print_exc()
    except Exception as exc:
        print(f"❌ Error initializing assistant: {exc}", flush=True)
        return 1

    print("\n" + "=" * 60, flush=True)
    print("📊 SMOKE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"✅ Successful: {results['success']}/{len(prompts_subset)}", flush=True)
    print(f"❌ Errors: {results['error']}/{len(prompts_subset)}", flush=True)
    if results["success"] > 0:
        avg_time = results["total_time"] / results["success"]
        print(f"⏱️  Average response time: {avg_time:.2f}s", flush=True)
        print(f"⏱️  Total time: {results['total_time']:.2f}s", flush=True)
    print("=" * 60, flush=True)
    return 0 if results["error"] == 0 else 1


def _run_coverage_suite(args) -> int:
    """Run the isolated worker-agent real-service coverage suite against the prompt manifest."""
    coverage_prompts = _load_coverage_prompts()
    prompts_subset = _select_subset(coverage_prompts, args.limit, args.offset, args.category, "domain")
    aggregate_actual_tools: set[str] = set()
    aggregate_expected_tools: set[str] = set()
    mismatches: list[str] = []
    results = {"success": 0, "error": 0, "total_time": 0.0}

    print("=" * 60, flush=True)
    print("🧪 STRICT TOOL COVERAGE SUITE (REAL SERVICES)", flush=True)
    print("=" * 60, flush=True)
    print(f"📊 Total coverage prompts available: {len(coverage_prompts)}", flush=True)
    print(f"📋 Running {len(prompts_subset)} coverage prompt(s)", flush=True)
    print("=" * 60, flush=True)

    for idx, (_, item) in enumerate(prompts_subset, 1):
        prompt = item["query"]
        domain = item["domain"]
        language = item.get("language", "en")
        model_config = _resolve_coverage_model_config(args, domain)
        expected_tools = set(item.get("expected_tools", []))

        print(f"\n\n{'=' * 60}", flush=True)
        print(f"🔶 COVERAGE TEST {idx}/{len(prompts_subset)} ({item['id']})", flush=True)
        print(f"📝 Domain: {domain} | Language: {language.upper()}", flush=True)
        print(f"👤 USER: {prompt}", flush=True)
        print("-" * 60, flush=True)

        response, tools_used, _, elapsed, error, _response_usage = run_isolated_agent(
            domain=domain,
            query=prompt,
            config=model_config,
        )

        tool_set = set(tools_used)
        aggregate_expected_tools.update(expected_tools)
        aggregate_actual_tools.update(tool_set)
        results["total_time"] += elapsed

        print(f"🔧 Expected: {sorted(expected_tools)}", flush=True)
        print(f"🔍 Actual:   {sorted(tool_set)}", flush=True)
        print(f"⏱️  Latency: {elapsed:.2f}s", flush=True)

        if error is not None:
            results["error"] += 1
            mismatches.append(f"{item['id']}: runtime error -> {error}")
            print(f"❌ Error: {error}", flush=True)
        elif not expected_tools.issubset(tool_set):
            results["error"] += 1
            mismatches.append(
                f"{item['id']}: expected {sorted(expected_tools)} but observed {sorted(tool_set)}"
            )
            print(f"❌ Coverage mismatch for {item['id']}", flush=True)
        else:
            results["success"] += 1
            if not args.quiet:
                print("-" * 60, flush=True)
                print(f"🤖 Response preview: {str(response)[:220]}", flush=True)

    full_suite_requested = (
        args.limit is None
        and args.offset == 0
        and args.category is None
        and len(prompts_subset) == len(coverage_prompts)
    )

    missing_expected = sorted(aggregate_expected_tools - aggregate_actual_tools)
    missing_registry = sorted(set(EXPORTED_TOOL_NAMES) - aggregate_actual_tools) if full_suite_requested else []

    print("\n" + "=" * 60, flush=True)
    print("📊 COVERAGE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"✅ Successful: {results['success']}/{len(prompts_subset)}", flush=True)
    print(f"❌ Errors/Mismatches: {results['error']}/{len(prompts_subset)}", flush=True)
    print(f"🧰 Expected tools in this run: {len(aggregate_expected_tools)}", flush=True)
    print(f"🧪 Actual tools in this run:   {len(aggregate_actual_tools)}", flush=True)
    if results["success"] > 0:
        avg_time = results["total_time"] / max(results["success"], 1)
        print(f"⏱️  Average response time: {avg_time:.2f}s", flush=True)
        print(f"⏱️  Total time: {results['total_time']:.2f}s", flush=True)
    if missing_expected:
        print("⚠️  Missing expected tools in this run:", flush=True)
        for tool_name in missing_expected:
            print(f"   - {tool_name}", flush=True)
    if full_suite_requested and missing_registry:
        print("⚠️  Missing tools from the exported registry:", flush=True)
        for tool_name in missing_registry:
            print(f"   - {tool_name}", flush=True)
    print("=" * 60, flush=True)

    if mismatches:
        print("\n❌ Coverage mismatches detected:", flush=True)
        for mismatch in mismatches:
            print(f"   - {mismatch}", flush=True)

    if full_suite_requested:
        return 0 if not mismatches and not missing_registry else 1
    return 0 if not mismatches and not missing_expected else 1


def main() -> int:
    """Parse CLI arguments and dispatch the requested prompt suite."""
    parser = argparse.ArgumentParser(
        description="Manual smoke and coverage runner for the LISBOA assistant",
    )
    parser.add_argument("--suite", choices=["smoke", "coverage"], default="smoke")
    parser.add_argument("--limit", type=int, default=None, help="Max tests to run")
    parser.add_argument("--offset", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--quiet", action="store_true", help="Hide intermediate reasoning and previews")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category/domain depending on the selected suite",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Run a single custom prompt instead of the built-in suite")
    parser.add_argument("--interactive", action="store_true", help="Ask for one custom prompt via stdin and run it immediately")
    parser.add_argument("--language", type=str, default=None, help="Language hint for custom prompt runs")
    parser.add_argument(
        "--domain",
        choices=sorted(SUPPORTED_COVERAGE_DOMAINS),
        default=None,
        help="Worker domain for custom coverage prompts",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(SUPPORTED_MODEL_PROVIDERS),
        default=None,
        help="Override provider family for smoke runs or worker provider for coverage runs",
    )
    parser.add_argument("--model", type=str, default=None, help="Override the worker model for coverage runs")
    parser.add_argument("--temperature", type=float, default=None, help="Override the worker temperature for coverage runs")
    args = parser.parse_args()

    if args.prompt and args.interactive:
        parser.error("Use either --prompt or --interactive, not both.")
    if (args.model or args.temperature is not None) and args.suite == "smoke":
        parser.error("--model and --temperature are only supported with --suite coverage.")

    if args.prompt or args.interactive:
        return _run_custom_prompt(args)

    if args.suite == "coverage":
        return _run_coverage_suite(args)
    return _run_smoke_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
