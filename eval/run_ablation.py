import json
import os
import sys
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import MultiAgentAssistant  # For the LISBOA full pipeline comparison

from agent.llm_factory import LLMFactory
from eval.llm_judge import LLMJudge


def load_dataset(filepath="eval/dataset.json", limit=20):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Select roughly a mix of 20 queries representing the 3 domains
    return data[:limit]


def run_zero_shot(query: str, provider: str = "azure", model_name: str = "gpt-4o"):
    llm = LLMFactory.get_llm(provider=provider, model=model_name, temperature=0.0)
    
    start = time.time()
    try:
        response = llm.invoke([HumanMessage(content=query)])
        latency = time.time() - start
        return response.content, [], "", latency, None
    except Exception as e:
        return f"Error: {e}", [], "", time.time() - start, str(e)


def run_lisboa(query: str, system: MultiAgentAssistant):
    start = time.time()
    try:
        # Instead of just using .chat(), we invoke the graph directly to get the ToolMessages
        state = {"messages": [("user", query)], "iteration_count": 0}
        result = system.agent_graph.invoke(state, config={"configurable": {"thread_id": "ablation"}})
        
        tools_called = []
        retrieved_context_blocks = []
        
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_called.append(tc["name"])
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if hasattr(msg, "name") and msg.name else "unknown_tool"
                retrieved_context_blocks.append(f"[{tool_name}] returned:\n{msg.content}")

        final_msg = result.get("messages", [])[-1]
        response = getattr(final_msg, "content", "")
        latency = time.time() - start
        
        retrieved_context_str = "\n---\n".join(retrieved_context_blocks)
        
        return response, tools_called, retrieved_context_str, latency, None
    except Exception as e:
        return f"Error: {e}", [], "", time.time() - start, str(e)


def run_ablation():
    print("=" * 60)
    print("🔬 STARTING ABLATION STUDY (Zero-Shot vs LISBOA Framework)")
    print("=" * 60)
    
    dataset = load_dataset()
    try:
        judge = LLMJudge()
    except ValueError as e:
        print(f"FAILED TO INIT JUDGE: {e}")
        return

    lisboa_system = MultiAgentAssistant()
    
    results = []

    for idx, item in enumerate(dataset):
        print(f"\n[{idx+1}/{len(dataset)}] ABLATING: {item['query']}")
        
        # 1. Zero-Shot
        zs_resp, zs_tools, zs_ctx, zs_lat, zs_err = run_zero_shot(item['query'])
        zs_score = judge.evaluate(
            query=item['query'],
            expected_facts=item.get('expected_facts', []),
            expected_tools=item.get('expected_tools', []),
            actual_tools=zs_tools,
            retrieved_context=zs_ctx,
            response=zs_resp
        )
        print(f"  [Zero-Shot] Score: {zs_score['composite_score']:.2f}/5.0 | Lat: {zs_lat:.2f}s")
        
        # 2. LISBOA (Tool Grounded)
        ls_resp, ls_tools, ls_ctx, ls_lat, ls_err = run_lisboa(item['query'], lisboa_system)
        ls_score = judge.evaluate(
            query=item['query'],
            expected_facts=item.get('expected_facts', []),
            expected_tools=item.get('expected_tools', []),
            actual_tools=ls_tools,
            retrieved_context=ls_ctx,
            response=ls_resp
        )
        print(f"  [LISBOA]    Score: {ls_score['composite_score']:.2f}/5.0 | Lat: {ls_lat:.2f}s | Tools: {len(ls_tools)}")

        results.append({
            "id": item["id"],
            "query": item["query"],
            "domain": item["domain"],
            "language": item.get("language", "en"),
            "edge_case": item.get("edge_case", False),
            "edge_type": item.get("edge_type", None),
            "expected_facts": item.get("expected_facts", []),
            "expected_tools": item.get("expected_tools", []),
            "metrics": {
                "zero_shot": {
                    "scores": zs_score,
                    "response": zs_resp,
                    "latency": round(zs_lat, 3),
                    "error": zs_err,
                },
                "lisboa": {
                    "scores": ls_score,
                    "response": ls_resp,
                    "tools_used": ls_tools,
                    "retrieved_context": ls_ctx,
                    "latency": round(ls_lat, 3),
                    "error": ls_err,
                }
            }
        })

    # Build aggregate summary
    zs_scores = [r["metrics"]["zero_shot"]["scores"]["composite_score"] for r in results
                 if r["metrics"]["zero_shot"]["scores"]["composite_score"] > 0]
    ls_scores = [r["metrics"]["lisboa"]["scores"]["composite_score"] for r in results
                 if r["metrics"]["lisboa"]["scores"]["composite_score"] > 0]
    
    summary = {
        "total_queries": len(results),
        "zero_shot_avg": round(sum(zs_scores) / len(zs_scores), 3) if zs_scores else 0,
        "lisboa_avg": round(sum(ls_scores) / len(ls_scores), 3) if ls_scores else 0,
        "lisboa_improvement": round(
            (sum(ls_scores) / len(ls_scores)) - (sum(zs_scores) / len(zs_scores)), 3
        ) if zs_scores and ls_scores else 0,
        "per_domain": {},
    }
    
    # Per-domain comparison
    domains = set(r["domain"] for r in results)
    for domain in sorted(domains):
        dr = [r for r in results if r["domain"] == domain]
        d_zs = [r["metrics"]["zero_shot"]["scores"]["composite_score"] for r in dr
                if r["metrics"]["zero_shot"]["scores"]["composite_score"] > 0]
        d_ls = [r["metrics"]["lisboa"]["scores"]["composite_score"] for r in dr
                if r["metrics"]["lisboa"]["scores"]["composite_score"] > 0]
        summary["per_domain"][domain] = {
            "count": len(dr),
            "zero_shot_avg": round(sum(d_zs) / len(d_zs), 3) if d_zs else 0,
            "lisboa_avg": round(sum(d_ls) / len(d_ls), 3) if d_ls else 0,
        }

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"eval/ablation_results_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "ablation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(results),
            },
            "summary": summary,
            "ablation_results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Ablation Study complete. Results saved to {out_file}")


if __name__ == "__main__":
    run_ablation()
