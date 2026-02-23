import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import ToolMessage

from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.agents.weather_agent import WeatherAgent
from eval.llm_judge import LLMJudge

# LLM Providers configured in the factory (e.g. Azure & LM Studio)
MODELS_TO_TEST = [
    {"provider": "azure", "model": "gpt-4o", "temperature": 0.0},
    {"provider": "lmstudio", "model": "local-model", "temperature": 0.0},
]


def load_dataset(filepath="eval/dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_isolated_agent(domain: str, query: str, config: dict):
    agent = None
    if domain == "weather":
        agent = WeatherAgent()
    elif domain == "transport":
        agent = TransportAgent()
    elif domain == "researcher":
        agent = ResearcherAgent()
    else:
        raise ValueError(f"Unknown domain for isolated testing: {domain}")

    # Override the agent's LLM config dynamically
    agent.init_llm(
        provider=config["provider"],
        model=config["model"],
        temperature=config["temperature"]
    )
    
    # Track execution
    start_time = time.time()
    tools_called = []
    retrieved_context_blocks = []
    final_response = ""
    error = None

    try:
        # BaseAgent.invoke() returns just the string response.
        # However, to get the actual tools used, we might need a workaround or
        # BaseAgent doesn't return state dict directly from `invoke`.
        # Let's run the state graph directly using `execute_react_loop` or `build_subgraph().invoke()`
        
        graph = agent.build_subgraph()
        result = graph.invoke({"messages": [("user", query)]})

        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_called.append(tc["name"])
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if hasattr(msg, "name") and msg.name else "unknown_tool"
                retrieved_context_blocks.append(f"[{tool_name}] returned:\n{msg.content}")
        
        final_msg = result.get("messages", [])[-1]
        final_response = getattr(final_msg, "content", "")

    except Exception as e:
        error = str(e)
        final_response = f"Execution Error: {error}"

    latency = time.time() - start_time
    retrieved_context_str = "\n---\n".join(retrieved_context_blocks)
    return final_response, tools_called, retrieved_context_str, latency, error


def run_benchmark(limit: int = None, models: list = MODELS_TO_TEST):
    print("=" * 60)
    print("🚀 STARTING ACADEMIC LISBOA BENCHMARK")
    print("=" * 60)
    
    dataset = load_dataset()
    if limit:
        dataset = dataset[:limit]

    # Initialize the judge
    try:
        judge = LLMJudge()
    except ValueError as e:
        print(f"FAILED TO INIT JUDGE: {e}")
        return

    results = []

    for model_config in models:
        model_id = f"{model_config['provider']}::{model_config['model']}"
        print(f"\nEvaluating Model: {model_id}")
        
        for idx, item in enumerate(dataset):
            print(f"  [{idx+1}/{len(dataset)}] [{item['domain'].upper()}] {item['query'][:50]}...")
            
            response, tools, retrieved_context, latency, error = run_isolated_agent(
                domain=item['domain'], 
                query=item['query'], 
                config=model_config
            )

            # Judge response
            judge_scores = judge.evaluate(
                query=item['query'],
                expected_facts=item.get('expected_facts', []),
                expected_tools=item.get('expected_tools', []),
                actual_tools=tools,
                retrieved_context=retrieved_context,
                response=response
            )

            record = {
                "id": item["id"],
                "domain": item["domain"],
                "query": item["query"],
                "language": item.get("language", "en"),
                "edge_case": item.get("edge_case", False),
                "edge_type": item.get("edge_type", None),
                "model": model_id,
                "latency_s": round(latency, 2),
                "error": error,
                "response": response,
                "tools_used": tools,
                "expected_tools": item.get("expected_tools", []),
                "expected_facts": item.get("expected_facts", []),
                "retrieved_context": retrieved_context,
                "scores": judge_scores
            }
            results.append(record)

            print(f"          → Score: {judge_scores['composite_score']:.2f}/5.0 | Latency: {latency:.2f}s | Reason: {judge_scores['reasoning']}")

    # Build aggregate summary
    summary = _build_summary(results)

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"eval/benchmark_results_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(dataset),
                "models_tested": [f"{m['provider']}::{m['model']}" for m in models],
            },
            "summary": summary,
            "benchmark_results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Benchmark complete. Results saved to {out_file}")


def _build_summary(results: list) -> dict:
    """Builds aggregate summary statistics from benchmark results."""
    if not results:
        return {}
    
    # Overall averages
    scores = [r["scores"]["composite_score"] for r in results if r["scores"]["composite_score"] > 0]
    latencies = [r["latency_s"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    
    summary = {
        "overall": {
            "total_evaluated": len(results),
            "total_errors": len(errors),
            "avg_composite_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        },
        "per_domain": {},
        "per_model": {},
    }
    
    # Per-domain breakdown
    domains = set(r["domain"] for r in results)
    for domain in sorted(domains):
        domain_results = [r for r in results if r["domain"] == domain]
        domain_scores = [r["scores"]["composite_score"] for r in domain_results if r["scores"]["composite_score"] > 0]
        summary["per_domain"][domain] = {
            "count": len(domain_results),
            "avg_composite_score": round(sum(domain_scores) / len(domain_scores), 3) if domain_scores else 0,
            "avg_factual_accuracy": round(
                sum(r["scores"]["factual_accuracy"] for r in domain_results if r["scores"]["factual_accuracy"] > 0) /
                max(1, sum(1 for r in domain_results if r["scores"]["factual_accuracy"] > 0)), 3
            ),
        }
    
    # Per-model breakdown
    models = set(r["model"] for r in results)
    for model in sorted(models):
        model_results = [r for r in results if r["model"] == model]
        model_scores = [r["scores"]["composite_score"] for r in model_results if r["scores"]["composite_score"] > 0]
        summary["per_model"][model] = {
            "count": len(model_results),
            "avg_composite_score": round(sum(model_scores) / len(model_scores), 3) if model_scores else 0,
            "errors": sum(1 for r in model_results if r["error"] is not None),
        }
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run per model")
    args = parser.parse_args()
    
    run_benchmark(limit=args.limit)
