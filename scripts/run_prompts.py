# ===========================================================================
# Master Thesis - Multi-Agent Prompt Runner
#   - André Filipe Gomes Silvestre, 20240502
#
# Manual prompt runner for two modes:
#   1. smoke    -> End-to-end MultiAgentAssistant sanity checks
#   2. coverage -> Isolated worker-agent real-service tool coverage prompts
#
# Usage:
#   > python scripts/run_prompts.py --suite smoke
#       Run the full built-in end-to-end smoke suite through MultiAgentAssistant.
#   > python scripts/run_prompts.py --suite smoke --limit 5 --offset 2 --category transport
#       Run only a filtered slice of the smoke suite.
#   > python scripts/run_prompts.py --prompt "Como está o tempo hoje?" --language pt --provider azure
#       Run one custom smoke prompt through MultiAgentAssistant with an optional provider override.
#   > python scripts/run_prompts.py --interactive --transcript-file test_queries_15.04.2026.txt
#       Ask for one ad-hoc smoke prompt via stdin and append the full terminal block to a transcript.
#   > python scripts/run_prompts.py --suite coverage
#       Run the strict worker-agent coverage manifest against live tools.
#   > python scripts/run_prompts.py --suite coverage --limit 5 --category transport
#       Run only a filtered slice of the coverage manifest.
#   > python scripts/run_prompts.py --suite coverage --prompt "Next train from Rossio?" --domain transport --provider azure --model gpt-5.4-mini --temperature 0
#       Run one custom coverage prompt against an isolated worker with provider/model overrides.
#   > python scripts/run_prompts.py --prompt "Quais museus estão abertos hoje?" --language pt --transcript-file test_queries_15.04.2026.txt --overwrite-transcript
#       Reset the transcript file and capture a fresh custom smoke run.
#
# Parameters:
#   --suite {smoke,coverage}   choose between the full assistant smoke suite and the isolated worker coverage suite
#   --limit N                  run only the first N selected prompts after filtering
#   --offset N                 skip the first N selected prompts after filtering
#   --category NAME            filter smoke prompts by category or coverage prompts by worker domain
#   --quiet                    hide the extra debug footer and retrieved-context previews
#   --prompt TEXT              run one custom prompt instead of the built-in suite
#   --interactive              ask for one custom prompt via stdin
#   --language CODE            language hint for custom prompt runs
#   --transcript-file PATH     append each captured terminal block to a transcript artifact
#   --overwrite-transcript     reset the transcript file before the run
#   --domain NAME              worker domain for custom coverage runs
#   --provider NAME            override provider family for smoke runs or worker provider for coverage runs
#   --model NAME               override the isolated worker model for coverage runs
#   --temperature FLOAT        override the isolated worker temperature for coverage runs
#
# Notes:
#   - Run this script from the repository root using the relative path above.
#   - `--prompt` and `--interactive` both dispatch one ad-hoc run; smoke mode uses MultiAgentAssistant, while coverage mode uses the isolated worker harness.
#   - The `🕵️ INTERMEDIATE STEPS & TOOLS` footer is a runner-side debug trace and only appears when `--quiet` is not set.
#   - Avoid absolute pytest-style paths in this workspace on Windows because
#     the folder name contains `[` and `]`, which pytest can interpret as glob
#     characters.
# ===========================================================================

import argparse
import io
import json
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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
from agent.utils.startup_resources import run_startup_preload
from config import Config
from eval.run_benchmark import run_isolated_agent
from tools import __all__ as EXPORTED_TOOL_NAMES

COVERAGE_MANIFEST_PATH = Path(PROJECT_ROOT) / "tests" / "fixtures" / "tool_coverage_manifest.json"
SUPPORTED_MODEL_PROVIDERS = {"azure", "openai", "lmstudio"}
SUPPORTED_COVERAGE_DOMAINS = {"weather", "transport", "researcher"}
DEFAULT_TRANSCRIPT_FILENAME = "test_queries_15.04.2026.txt"

# Each prompt is a tuple: (prompt_text, language_code, category)
SMOKE_PROMPTS = [
    # Additional test prompts (2026-04)
    ("Vai chover amahna em Lisboa?", "pt", "weather"),
    ("Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa.", "pt", "transport"),
    ("Quando é a Feira do Livro?", "pt", "event"),
    ("Fala-me do Web Summit", "pt", "event"),
    ("Onde fica o Museu do Livro?", "pt", "researcher"),
    ("Fala-me do Mosteiro dos Jerónimos", "pt", "researcher"),
    
    
    # Additional test prompts (2026-03)
    ("Qual o próximo autocarro do Marquês para Belém?", "pt", "test"),
    ("Quero ir de metro ou comboio entre Entrecampos e Sete Rios? Qual o mais rápido e o mais barato?", "pt", "test"),
    ("Estou em Entrecampos e quero fazer um passeio turistico mas não pelos sitios habituais turisticos. Quero algo diferente... sugere-me", "pt", "test"),
    ("Qual o hospital e a farmácia mais perto do Saldanha?", "pt", "test"),
    ("Quero ir de transportes públicos entre o ISCTE e a Zara do Rossio", "pt", "test"),
    ("Qual museu ou monumento recomendas ir neste domingo sendo que apenas tenho das 19 às 20h para visitar?", "pt", "test"),
    
    # CRITICAL end-to-end prompts
    (
        "Quais os próximos autocarros da Carris no Rossio para seguir para Belém agora?",
        "pt",
        "transport",
    ),
    (
        "Plan a full afternoon in Belém starting from Chiado, include historical context, realistic transport, and one pastry stop.",
        "en",
        "planner",
    ),

    # Weather
    ("How is the weather in Lisbon today and what should I wear for walking outdoors?", "en", "weather"),

    # Transport: realtime, future, deterministic routes, and scope limits
    ("How do I get from Lisbon Airport to Rossio using the metro right now?", "en", "transport"),
    ("Como vou amanhã do Rossio ao Aeroporto de metro e o que muda por ser uma viagem futura?", "pt", "transport"),
    ("Is the 28E tram running on time right now, and if not what fallback should I take?", "en", "transport"),
    ("Next train from Cais do Sodré to Cascais, and tell me if there are any obvious disruptions.", "en", "transport"),

    # Researcher: tourists and residents
    ("Best seafood restaurants near the Tagus river with a nice view and not overly touristy.", "en", "researcher"),
    ("Where is the nearest pharmacy to Parque das Nações that should still be useful this evening?", "en", "researcher"),
    ("Quais são os museus gratuitos este fim de semana em Lisboa e algum evento interessante igualmente gratuito?", "pt", "researcher"),

    # Multi-agent and multilingual
    (
        "Quel temps fait-il à Lisbonne aujourd'hui et quel est le meilleur moyen d'aller à la Tour de Belém depuis le centre?",
        "fr",
        "multi",
    ),

    # Guardrails / edge cases
    ("Quero ir de metro para a Madeira.", "pt", "edge_case"),
    ("Preciso do próximo Fertagus para Setúbal e de ferry para o Barreiro agora.", "pt", "edge_case")
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
    prompt = args.prompt.strip()
    if prompt:
        return prompt
    if not args.interactive:
        return None

    try:
        prompt = input("Enter a prompt to test: ").strip()
    except EOFError:
        return None
    return prompt or None


def _clone_model_config(raw_config: object, label: str) -> dict[str, Any]:
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


class _StreamTee:
    """Mirror console output to both the terminal and an in-memory buffer."""

    def __init__(self, *streams) -> None:
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def reconfigure(self, **kwargs) -> None:
        """Best-effort passthrough for wrapped streams that support reconfigure."""
        for stream in self.streams:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(**kwargs)

    def __getattr__(self, name: str):
        """Delegate unsupported stream attributes to the primary console stream."""
        if not self.streams:
            raise AttributeError(name)
        return getattr(self.streams[0], name)


def _resolve_transcript_path(raw_path: str | None) -> Path | None:
    """Resolve an optional transcript path relative to the repository root."""
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path(PROJECT_ROOT) / candidate
    return candidate


def _initialize_transcript(path: Path, overwrite: bool = False) -> None:
    """Create or reset the transcript file with a small session header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return

    header = (
        "# LISBOA prompt transcript\n"
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Repository root: {PROJECT_ROOT}\n\n"
    )
    path.write_text(header, encoding="utf-8")


def _append_transcript_block(path: Path, title: str, body: str) -> None:
    """Append a captured execution block to the transcript artifact."""
    divider = "=" * 80
    block = (
        f"{divider}\n"
        f"{title}\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{divider}\n"
        f"{body.rstrip()}\n\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(block)


@contextmanager
def _capture_transcript_block(transcript_path: Path | None, title: str):
    """Capture a prompt execution block while still echoing output to the console."""
    if transcript_path is None:
        yield
        return

    buffer = io.StringIO()
    with redirect_stdout(_StreamTee(sys.stdout, buffer)), redirect_stderr(_StreamTee(sys.stderr, buffer)):
        yield

    _append_transcript_block(transcript_path, title, buffer.getvalue())


def _run_custom_prompt(args) -> int:
    """Run one ad-hoc prompt through either smoke or coverage mode."""
    prompt = _resolve_custom_prompt(args)
    transcript_path = getattr(args, "transcript_file", None)
    if not prompt:
        print("❌ No custom prompt was provided.", flush=True)
        return 1

    language = args.language or "en"
    if args.suite == "coverage":
        if not args.domain:
            print("❌ --domain is required for custom coverage prompts.", flush=True)
            return 1

        model_config = _resolve_coverage_model_config(args, args.domain)
        transcript_title = f"CUSTOM COVERAGE | Domain: {args.domain} | Language: {language.upper()} | Prompt: {prompt}"
        with _capture_transcript_block(transcript_path, transcript_title):
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
        transcript_title = f"CUSTOM SMOKE | Language: {language.upper()} | Prompt: {prompt}"
        with _capture_transcript_block(transcript_path, transcript_title):
            print("=" * 60, flush=True)
            print("🧪 CUSTOM SMOKE PROMPT", flush=True)
            print("=" * 60, flush=True)

            preload_status = _warm_smoke_resources(language=language)
            if not preload_status.get("ok", False):
                print(
                    "❌ Shared resource preload failed; aborting before assistant startup.",
                    flush=True,
                )
                return 1

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
                    _print_smoke_tool_trace(
                        assistant.state.get("messages", []),
                        response,
                        elapsed,
                        execution_summary=getattr(assistant, "last_execution_summary", None),
                    )
                if _should_echo_final_smoke_response():
                    print(f"🤖 FINAL RESPONSE ({elapsed:.2f}s):", flush=True)
                    print(response, flush=True)
                return 0
            except Exception as exc:
                print(f"❌ ERROR: {exc}", flush=True)
                import traceback

                traceback.print_exc()
                return 1


def _print_smoke_tool_trace(messages, response: str, elapsed: float, execution_summary: dict | None = None) -> None:
    """Print intermediate tool activity for the smoke suite."""
    print("\n\033[1;34m--- 🕵️ INTERMEDIATE STEPS & TOOLS ---\033[0m", flush=True)

    if isinstance(execution_summary, dict):
        agent_tool_logs = execution_summary.get("agent_tool_logs", {})
        total_tool_invocations = int(execution_summary.get("total_tool_invocations", 0) or 0)
        if agent_tool_logs:
            for agent_name, tool_log in agent_tool_logs.items():
                display_name = "QA" if agent_name == "qa" else str(agent_name).title()
                print(f"  \033[1;36m[{display_name}]\033[0m {len(tool_log)} tool call(s)", flush=True)
                for item in tool_log:
                    print(
                        f"  \033[1;33m[TOOL REQUEST]\033[0m {item.get('tool_name', 'unknown')}({item.get('args', {})})",
                        flush=True,
                    )
            print(
                f"  \033[1;35m[METADATA]\033[0m Tools used: {total_tool_invocations} | Latency: {elapsed:.2f}s",
                flush=True,
            )
            print("\033[1;34m---------------------------------------\033[0m\n", flush=True)
            return

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


def _warm_smoke_resources(language: str = "pt") -> dict[str, Any]:
    """Preload shared runtime resources before smoke-mode assistant startup."""
    print("\n🚀 Preloading shared runtime resources...", flush=True)
    preload_status = run_startup_preload(
        language=language,
        use_multi_agent=Config.USE_MULTI_AGENT,
    )

    transport_status = str(preload_status.get("transport_status") or "")
    if transport_status:
        prefix = "✅" if preload_status.get("transport_ok") else "⚠️"
        print(f"{prefix} Transport preload: {transport_status}", flush=True)

    kb_status = preload_status.get("kb_status")
    if kb_status:
        prefix = "✅" if preload_status.get("kb_ok") else "⚠️"
        print(f"{prefix} {kb_status}", flush=True)

    return preload_status


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
    transcript_path = getattr(args, "transcript_file", None)
    print("=" * 60, flush=True)
    print("🧪 MULTI-AGENT SYSTEM TEST SUITE (SMOKE)", flush=True)
    print("=" * 60, flush=True)

    print("\nInitializing Multi-Agent System...", flush=True)
    try:
        with _temporary_model_provider(args.provider):
            prompts_subset = _select_subset(SMOKE_PROMPTS, args.limit, args.offset, args.category, 2)

            preload_language = prompts_subset[0][1][1] if prompts_subset else "pt"
            preload_status = _warm_smoke_resources(language=preload_language)
            if not preload_status.get("ok", False):
                print(
                    "❌ Shared resource preload failed; aborting smoke suite before assistant startup.",
                    flush=True,
                )
                return 1

            assistant = MultiAgentAssistant()

            print(f"✅ Model: {assistant.model_name}", flush=True)
            print(f"✅ Provider family: {Config.MODEL_PROVIDER}", flush=True)
            print(f"📊 Total smoke prompts available: {len(SMOKE_PROMPTS)}", flush=True)
            print(f"📋 Running {len(prompts_subset)} smoke prompt(s)", flush=True)
            print("=" * 60, flush=True)

            results = {"success": 0, "error": 0, "total_time": 0.0}

            for idx, (original_idx, (prompt, lang, category)) in enumerate(prompts_subset, 1):
                transcript_title = (
                    f"SMOKE TEST {idx}/{len(prompts_subset)} | Prompt #{original_idx + 1} "
                    f"| Category: {category} | Language: {lang.upper()} | Prompt: {prompt}"
                )
                with _capture_transcript_block(transcript_path, transcript_title):
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
                            _print_smoke_tool_trace(
                                assistant.state.get("messages", []),
                                response,
                                elapsed,
                                execution_summary=getattr(assistant, "last_execution_summary", None),
                            )

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
    transcript_path = getattr(args, "transcript_file", None)
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

        transcript_title = (
            f"COVERAGE TEST {idx}/{len(prompts_subset)} | {item['id']} "
            f"| Domain: {domain} | Language: {language.upper()} | Prompt: {prompt}"
        )
        with _capture_transcript_block(transcript_path, transcript_title):
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
        "--transcript-file",
        type=str,
        default=None,
        help=(
            "Optional transcript artifact path for appending full prompt outputs "
            f"(for example: {DEFAULT_TRANSCRIPT_FILENAME})"
        ),
    )
    parser.add_argument(
        "--overwrite-transcript",
        action="store_true",
        help="Reset the transcript file before the run instead of appending to it",
    )
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

    args.transcript_file = _resolve_transcript_path(args.transcript_file)
    if args.transcript_file is not None:
        _initialize_transcript(args.transcript_file, overwrite=args.overwrite_transcript)

    if args.prompt or args.interactive:
        return _run_custom_prompt(args)

    if args.suite == "coverage":
        return _run_coverage_suite(args)
    return _run_smoke_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
