# ==========================================================================
# Master Thesis - Config & Audit Fix Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Tests that validate audit fixes are correctly applied:
#   - A4: MARDKOWN typo corrected to MARKDOWN
#   - A6: No extra LLM instance in LisbonAssistant
#   - A7: No import json inside loops
#   - Config loads without errors
# ==========================================================================

import inspect
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ==========================================================================
# Test Result Tracking (matches test_lisbon_transport.py pattern)
# ==========================================================================

class TestResults:
    """Tracks test results with pass/fail counts."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"  \033[1;32m✅ PASS\033[0m: {test_name}")
    
    def add_fail(self, test_name, reason):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"  \033[1;31m❌ FAIL\033[0m: {test_name} - {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("\033[1m📊 TEST SUMMARY\033[0m")
        print("=" * 70)
        print(f"\033[1;32m✅ Passed: {self.passed}/{total}\033[0m")
        print(f"\033[1;31m❌ Failed: {self.failed}/{total}\033[0m")
        print(f"⏱️ Duration: {elapsed:.2f}s")
        
        if self.errors:
            print("\n\033[1;31mFailed tests:\033[0m")
            for err in self.errors:
                print(f"  • {err}")
        
        if self.failed == 0:
            print("\n\033[1;32m🎉 ALL TESTS PASSED!\033[0m")
        
        print("=" * 70 + "\n")
        return self.failed


# ==========================================================================
# A4: Config typo fix validation
# ==========================================================================

def test_config_typo_fix(results: TestResults):
    """Tests that the MARDKOWN typo is corrected."""
    print("\n\033[1m📋 A4: Config Typo Fix (MARDKOWN → MARKDOWN)\033[0m")
    print("-" * 50)
    
    try:
        from config import Config

        # Check that the corrected attribute exists
        if hasattr(Config, 'SHOW_MARKDOWN_RESPONSE_IN_TERMINAL'):
            results.add_pass("Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL exists")
        else:
            results.add_fail("SHOW_MARKDOWN_RESPONSE_IN_TERMINAL", "Attribute not found in Config")
        
        # Check that the OLD typo does NOT exist
        if hasattr(Config, 'SHOW_MARDKOWN_RESPONSE_IN_TERMINAL'):
            results.add_fail("SHOW_MARDKOWN_RESPONSE_IN_TERMINAL", "Old typo still exists!")
        else:
            results.add_pass("Old typo SHOW_MARDKOWN_RESPONSE_IN_TERMINAL removed")
        
        # Check the value is boolean
        val = Config.SHOW_MARKDOWN_RESPONSE_IN_TERMINAL
        if isinstance(val, bool):
            results.add_pass(f"SHOW_MARKDOWN_RESPONSE_IN_TERMINAL is bool ({val})")
        else:
            results.add_fail("SHOW_MARKDOWN_RESPONSE_IN_TERMINAL type", f"Expected bool, got {type(val)}")
            
    except Exception as e:
        results.add_fail("Config import", str(e))


# ==========================================================================
# A4: BaseAgent reference fix validation
# ==========================================================================

def test_base_agent_typo_fix(results: TestResults):
    """Tests that BaseAgent references the corrected config name."""
    print("\n\033[1m📋 A4: BaseAgent Config Reference Fix\033[0m")
    print("-" * 50)
    
    try:
        # Read the source file and check for old typo
        base_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'agents', 'base.py')
        with open(base_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'SHOW_MARDKOWN_RESPONSE_IN_TERMINAL' in content:
            results.add_fail("BaseAgent source", "Still contains MARDKOWN typo")
        else:
            results.add_pass("BaseAgent source has no MARDKOWN typo")
        
        if 'SHOW_MARKDOWN_RESPONSE_IN_TERMINAL' in content:
            results.add_pass("BaseAgent uses corrected SHOW_MARKDOWN_RESPONSE_IN_TERMINAL")
        else:
            results.add_fail("BaseAgent source", "Missing MARKDOWN reference")
            
    except Exception as e:
        results.add_fail("BaseAgent source check", str(e))


# ==========================================================================
# A7: import json not inside loops
# ==========================================================================

def test_no_import_json_in_loops(results: TestResults):
    """Tests that 'import json' is NOT inside while loops."""
    print("\n\033[1m📋 A7: No import json inside loops\033[0m")
    print("-" * 50)
    
    files_to_check = [
        ('agent/agents/researcher_agent.py', 'ResearcherAgent'),
        ('agent/agents/transport_agent.py', 'TransportAgent'),
    ]
    
    for filepath, name in files_to_check:
        try:
            full_path = os.path.join(os.path.dirname(__file__), '..', filepath)
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check if 'import json' appears at top-level (within first 20 lines)
            top_level_json = any(
                line.strip() == 'import json' 
                for line in lines[:20]
            )
            
            if top_level_json:
                results.add_pass(f"{name}: import json at top level")
            else:
                results.add_fail(f"{name}: import json", "Not found at file top level")
            
            # Check no 'import json' inside indented blocks (loop bodies)
            inside_loop_import = False
            for i, line in enumerate(lines[20:], start=21):
                stripped = line.strip()
                if stripped == 'import json' and line.startswith((' ', '\t')):
                    inside_loop_import = True
                    results.add_fail(f"{name}: import json", f"Found inside block at line {i}")
                    break
            
            if not inside_loop_import:
                results.add_pass(f"{name}: no import json inside loops")
                
        except Exception as e:
            results.add_fail(f"{name} check", str(e))


# ==========================================================================
# A3: No duplicate OpenAI elif block
# ==========================================================================

def test_no_duplicate_openai_block(results: TestResults):
    """Tests that app.py has no duplicate OpenAI provider block."""
    print("\n\033[1m📋 A3: No Duplicate OpenAI Block in app.py\033[0m")
    print("-" * 50)
    
    try:
        app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count occurrences of the OpenAI elif pattern
        count = content.count('elif selected_provider == "openai":')
        
        if count == 1:
            results.add_pass(f"app.py: exactly 1 OpenAI elif block (found {count})")
        elif count == 0:
            results.add_fail("app.py OpenAI block", "No OpenAI elif block found (expected 1)")
        else:
            results.add_fail("app.py OpenAI block", f"Found {count} OpenAI elif blocks (expected 1)")
            
    except Exception as e:
        results.add_fail("app.py check", str(e))


# ==========================================================================
# A2: ResearcherAgent uses full tool set
# ==========================================================================

def test_researcher_tools(results: TestResults):
    """Tests that ResearcherAgent gets all tools from base class."""
    print("\n\033[1m📋 A2: ResearcherAgent Tool Loading\033[0m")
    print("-" * 50)
    
    try:
        from agent.agents.base import get_agent_tools
        from agent.agents.researcher_agent import ResearcherAgent

        # Check that we get tools from get_agent_tools
        expected_tools = get_agent_tools("researcher")
        expected_count = len(expected_tools)
        
        if expected_count >= 10:
            results.add_pass(f"get_agent_tools('researcher') returns {expected_count} tools (>= 10)")
        else:
            results.add_fail("Tool count", f"Expected >= 10 tools, got {expected_count}")
        
        # Verify no hardcoded tool list in __init__
        source_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'agents', 'researcher_agent.py')
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check that __init__ doesn't have a hardcoded self.tools = [...] list
        init_match = re.search(r'def __init__\(self.*?\n(.*?)(?=\n    def )', source, re.DOTALL)
        if init_match:
            init_body = init_match.group(1)
            if 'self.tools = [' in init_body:
                results.add_fail("ResearcherAgent.__init__", "Still has hardcoded self.tools = [...]")
            else:
                results.add_pass("ResearcherAgent.__init__ has no hardcoded tool list")
        else:
            results.add_pass("ResearcherAgent.__init__ check (no override pattern found)")
            
    except Exception as e:
        results.add_fail("ResearcherAgent tools", str(e))


# ==========================================================================
# A6: No extra LLM instance in LisbonAssistant
# ==========================================================================

def test_no_extra_llm_instance(results: TestResults):
    """Tests that LisbonAssistant.__init__ doesn't create unnecessary LLM."""
    print("\n\033[1m📋 A6: No Extra LLM in LisbonAssistant\033[0m")
    print("-" * 50)
    
    try:
        graph_path = os.path.join(os.path.dirname(__file__), '..', 'agent', 'graph.py')
        with open(graph_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that LisbonAssistant.__init__ doesn't call LLMFactory.get_llm
        init_match = re.search(
            r'class LisbonAssistant.*?def __init__\(self.*?\n(.*?)(?=\n    def |\nclass )',
            content, re.DOTALL
        )
        
        if init_match:
            init_body = init_match.group(1)
            if 'LLMFactory.get_llm' in init_body:
                results.add_fail("LisbonAssistant.__init__", "Still calls LLMFactory.get_llm()")
            else:
                results.add_pass("LisbonAssistant.__init__ does NOT call LLMFactory.get_llm()")
            
            if 'Config' in init_body or 'Config.' in content:
                results.add_pass("Uses Config directly for model info")
        else:
            results.add_fail("LisbonAssistant", "Could not find __init__ in source")
            
    except Exception as e:
        results.add_fail("LisbonAssistant check", str(e))


# ==========================================================================
# Transport improvements: B1 & B4
# ==========================================================================

def test_transport_improvements(results: TestResults):
    """Tests travel time estimation and metro status integration."""
    print("\n\033[1m📋 B1/B4: Transport Improvements\033[0m")
    print("-" * 50)
    
    try:
        from tools.transport_api import (
            _count_metro_stations,
            _estimate_metro_time,
            _get_line_status,
            get_route_between_stations,
        )

        # B4: Station counting
        count = _count_metro_stations("amarela", "rato", "odivelas")
        if count == 12:
            results.add_pass(f"_count_metro_stations: Rato->Odivelas = {count}")
        else:
            results.add_fail("Station count", f"Expected 12, got {count}")
        
        # B4: Time estimation
        time_est = _estimate_metro_time(5, transfers=0)
        if "12" in time_est:
            results.add_pass(f"_estimate_metro_time(5, 0) = {time_est}")
        else:
            results.add_fail("Time estimate", f"Expected ~12 min, got {time_est}")
        
        time_est = _estimate_metro_time(8, transfers=1)
        if "21" in time_est:
            results.add_pass(f"_estimate_metro_time(8, 1) = {time_est}")
        else:
            results.add_fail("Time estimate with transfer", f"Expected ~21 min, got {time_est}")
        
        # B1: Line status (just verify it returns something)
        status = _get_line_status("verde")
        if status:
            results.add_pass(f"_get_line_status('verde') = '{status}'")
        else:
            results.add_fail("Line status", "Empty result")
        
        # Verify route output includes time estimate
        route = get_route_between_stations.invoke({
            "origin": "Aeroporto", "destination": "Saldanha"
        })
        if "Estimated travel time" in route:
            results.add_pass("Route output includes travel time estimate")
        else:
            results.add_fail("Route output", "Missing 'Estimated travel time'")
            
    except Exception as e:
        results.add_fail("Transport improvements", str(e))


# ==========================================================================
# B5: Transport prompt includes bus examples
# ==========================================================================

def test_transport_prompt(results: TestResults):
    """Tests that transport prompt includes bus/tram routing examples."""
    print("\n\033[1m📋 B5: Transport Prompt Bus/Tram Examples\033[0m")
    print("-" * 50)
    
    try:
        from agent.prompts.transport import get_transport_prompt
        
        prompt = get_transport_prompt()
        
        checks = [
            ("carris_find_routes_between", "Carris Urbana tool"),
            ("find_direct_bus_lines", "Carris Metropolitana tool"),
            ("find_bus_routes", "GPS-based bus tool"),
            ("get_transport_summary", "Transport summary tool"),
            ("Tempo estimado", "Travel time in template"),
        ]
        
        for term, description in checks:
            if term in prompt:
                results.add_pass(f"Prompt includes {description}")
            else:
                results.add_fail("Prompt missing", f"'{term}' ({description})")
                
    except Exception as e:
        results.add_fail("Transport prompt", str(e))


# ==========================================================================
# Main Execution
# ==========================================================================

def main():
    print("\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m🧪 AUDIT FIX VALIDATION - COMPREHENSIVE TEST SUITE\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")
    
    results = TestResults()
    
    # Run all test groups
    test_config_typo_fix(results)
    test_base_agent_typo_fix(results)
    test_no_import_json_in_loops(results)
    test_no_duplicate_openai_block(results)
    test_researcher_tools(results)
    test_no_extra_llm_instance(results)
    test_transport_improvements(results)
    test_transport_prompt(results)
    
    return results.summary()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
