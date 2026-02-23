# ==========================================================================
# Master Thesis - Multi-Agent System Test Suite
#   - André Filipe Gomes Silvestre, 20240502
#
# Automated testing for the Lisbon Urban Assistant multi-agent system.
# Tests various query types in English, Portuguese, German, and French.
#
# Usage:
#   python test_prompts.py                    # Run all tests
#   python test_prompts.py --limit 5          # Run first 5 tests
#   python test_prompts.py --offset 10        # Start from test 11
#   python test_prompts.py --limit 5 --offset 10  # Tests 11-15
#   python test_prompts.py --verbose          # Show agent reasoning
# ==========================================================================

import argparse
import io
import os
import sys
import time

# To fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path (parent of tests folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.graph import MultiAgentAssistant

# ==========================================================================
# Test Prompts with Language Configuration
# ==========================================================================
# Each prompt is a tuple: (prompt_text, language_code, category)
# Language codes: "en" (English), "pt" (Portuguese), "de" (German), "fr" (French)

PROMPTS = [
    # ==========================================================================
    # ENGLISH PROMPTS
    # ==========================================================================
    
    # Simple & Weather
    ("How is the weather in Lisbon today?", "en", "weather"),
    ("Will it rain this weekend in Sintra?", "en", "weather"),
    ("What is the current temperature in downtown Lisbon?", "en", "weather"),

    # Transport: Routing & Status
    ("How do I get from Lisbon Airport to Rossio using the metro?", "en", "transport"),
    ("Is the 28E tram running on time right now?", "en", "transport"),
    ("Next train from Cais do Sodré to Cascais.", "en", "transport"),
    ("Bus from Marquês de Pombal to Belém Tower.", "en", "transport"),
    ("Are there any subway strikes today?", "en", "transport"),

    # Places & Recommendations
    ("Best seafood restaurants near the Tagus river.", "en", "researcher"),
    ("Where is the nearest pharmacy to Parque das Nações?", "en", "researcher"),
    ("Museums of modern art open today.", "en", "researcher"),
    ("Cheap sushi places in Saldanha.", "en", "researcher"),

    # Complex / Multi-step (Planning)
    ("Plan a perfect afternoon in Belém visiting the Tower, Jerónimos Monastery, and eating Pastéis de Nata. Include transport from Chiado.", "en", "planner"),
    ("I want to go for a drink in Bairro Alto tonight. Any recommendations?", "en", "researcher"),
    
    # ==========================================================================
    # PORTUGUESE PROMPTS
    # ==========================================================================

    # Transportes (Carris / Metro / CP)
    ("Como vou do Castelo de São Jorge para Belém de autocarro? Quero evitar o metro.", "pt", "transport"),
    ("Onde estão os elétricos agora em tempo real?", "pt", "transport"),
    ("Quais as linhas de elétrico que passam na Graça?", "pt", "transport"),
    ("Próximo comboio para Sintra a partir do Rossio.", "pt", "transport"),
    ("Quero ir de Entrecampos ao Marquês.", "pt", "transport"),  # Simple metro route test
    
    # Planeamento & Lazer
    ("Sugere um passeio em Alfama com poucas subidas, estou com uma pessoa idosa.", "pt", "planner"),
    ("Quero ir jantar e depois sair à noite em Lisboa. O que recomendas?", "pt", "researcher"),
    ("Museus grátis ao domingo em Lisboa.", "pt", "researcher"),
    
    # Casos Específicos / Edge Cases
    ("Onde posso fazer um teste Covid hoje em Lisboa?", "pt", "researcher"),
    ("Há trotinetes elétricas perto do Jardim da Estrela?", "pt", "researcher"),
    ("Quero ir de metro para a Madeira.", "pt", "edge_case"),  # Impossible query check

    # ==========================================================================
    # GERMAN PROMPTS
    # ==========================================================================
    ("Wie komme ich vom Flughafen Lissabon ins Stadtzentrum mit öffentlichen Verkehrsmitteln?", "de", "transport"),

    # ==========================================================================
    # FRENCH PROMPTS
    # ==========================================================================
    ("Quel temps fait-il à Lisbonne aujourd'hui et quel est le meilleur moyen d'aller à la Tour de Belém?", "fr", "multi"),
]


def run_tests():
    parser = argparse.ArgumentParser(description="Test the Multi-Agent Lisbon Assistant")
    parser.add_argument("--limit", type=int, default=len(PROMPTS), help="Max tests to run")
    parser.add_argument("--offset", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--quiet", action="store_true", help="Hide intermediate agent reasoning and tool calls")
    parser.add_argument("--category", type=str, default=None, 
                        help="Filter by category: weather, transport, researcher, planner, edge_case, multi")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("🧪 MULTI-AGENT SYSTEM TEST SUITE", flush=True)
    print("=" * 60, flush=True)
    
    print("\nInitializing Multi-Agent System...", flush=True)
    try:
        assistant = MultiAgentAssistant()
    except Exception as e:
        print(f"❌ Error initializing assistant: {e}", flush=True)
        return

    print(f"✅ Model: {assistant.model_name}", flush=True)
    print(f"📊 Total prompts available: {len(PROMPTS)}", flush=True)
    print("=" * 60, flush=True)

    # Filter by category if specified
    if args.category:
        filtered_prompts = [(i, p) for i, p in enumerate(PROMPTS) if p[2] == args.category]
        print(f"🔍 Filtering by category: {args.category} ({len(filtered_prompts)} prompts)", flush=True)
    else:
        filtered_prompts = list(enumerate(PROMPTS))
    
    # Apply offset and limit
    prompts_subset = filtered_prompts[args.offset : args.offset + args.limit]
    
    print(f"📋 Running tests {args.offset + 1} to {args.offset + len(prompts_subset)}", flush=True)
    print("=" * 60, flush=True)

    results = {"success": 0, "error": 0, "total_time": 0}

    for idx, (original_idx, (prompt, lang, category)) in enumerate(prompts_subset, 1):
        print(f"\n\n{'='*60}", flush=True)
        print(f"🔶 TEST {idx}/{len(prompts_subset)} (Prompt #{original_idx + 1})", flush=True)
        print(f"📝 Category: {category} | Language: {lang.upper()}", flush=True)
        print(f"👤 USER: {prompt}", flush=True)
        print("-" * 60, flush=True)
        
        try:
            start_time = time.time()
            assistant.reset()  # Reset conversation state between tests
            
            # Use the language parameter for proper response language
            response = assistant.chat(prompt, verbose=not args.quiet, language=lang)
            
            elapsed = time.time() - start_time
            results["success"] += 1
            results["total_time"] += elapsed
            
            # --- EXTRACT INTERMEDIATE STEPS AND TOOLS ---
            if not args.quiet:
                print("\n\033[1;34m--- 🕵️ INTERMEDIATE STEPS & TOOLS ---\033[0m", flush=True)
                messages = assistant.state.get("messages", [])
                tools_used = 0
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  \033[1;33m[TOOL REQUEST]\033[0m {tc['name']}({tc.get('args', {})})", flush=True)
                            tools_used += 1
                    elif msg.__class__.__name__ == "ToolMessage":
                        content_preview = str(msg.content)[:100].replace('\n', ' ') + "..." if len(str(msg.content)) > 100 else str(msg.content).replace('\n', ' ')
                        print(f"  \033[1;32m[TOOL RESULT]\033[0m {content_preview}", flush=True)
                    elif msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", "") and not getattr(msg, "tool_calls", []):
                        if msg.content != response:  # Don't double print final response
                            agent_name = getattr(msg, "name", "AI")
                            print(f"  \033[1;36m[{agent_name} THOUGHT/RESPONSE]\033[0m {str(msg.content)[:100]}...", flush=True)
                            
                print(f"  \033[1;35m[METADATA]\033[0m Tools used: {tools_used} | Latency: {elapsed:.2f}s", flush=True)
                print("\033[1;34m---------------------------------------\033[0m\n", flush=True)

            print("-" * 60, flush=True)
            print(f"🤖 \033[1mFINAL AI RESPONSE\033[0m ({elapsed:.2f}s):", flush=True)
            print(response, flush=True)
            print("=" * 60, flush=True)
            
        except Exception as e:
            results["error"] += 1
            print(f"❌ ERROR in Test {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("📊 TEST SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"✅ Successful: {results['success']}/{len(prompts_subset)}", flush=True)
    print(f"❌ Errors: {results['error']}/{len(prompts_subset)}", flush=True)
    if results["success"] > 0:
        avg_time = results["total_time"] / results["success"]
        print(f"⏱️  Average response time: {avg_time:.2f}s", flush=True)
        print(f"⏱️  Total time: {results['total_time']:.2f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_tests()
