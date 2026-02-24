# ==========================================================================
# Master Thesis - Quality Assurance Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Validates completeness of agent outputs before final response.
#   Identifies missing data and requests additional agent calls if needed.
#   Ensures no incomplete or hallucinated responses reach the user.
# ==========================================================================

import os
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agent.agents.base import BaseAgent, clean_response, parse_json_response, traceable
from agent.prompts.qa import get_qa_prompt


class QualityAssuranceAgent(BaseAgent):
    """
    Quality Assurance agent that validates data completeness.

    Responsibilities:
        - Analyze outputs from specialized agents
        - Identify missing critical data for the query type
        - Request additional agent calls if data is incomplete
        - Flag potential hallucinations or data gaps
        - Add disclaimers about known data limitations

    Note:
        This agent has NO tools. It only analyzes existing agent outputs
        and returns a structured validation result.
    """

    def __init__(self):
        """Initializes the QA agent."""
        super().__init__("qa")

    @traceable(name="qa_agent", run_type="chain", tags=["sub-agent", "qa"])
    def validate(
        self,
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Validates if gathered data is complete for answering the user query.

        Args:
            user_query: The user's original query.
            agent_outputs: Dict mapping agent names to their output strings.
            agents_called: List of agent names that were called.
            language: Language code ('en' or 'pt').

        Returns:
            Dict with validation result:
                - complete (bool): True if data is sufficient
                - missing_data (List[str]): List of missing data fields
                - required_agents (List[str]): Agents to call for missing data
                - reasoning (str): Explanation of the assessment
                - disclaimers (List[str]): Warnings about data limitations
        """
        system_prompt = get_qa_prompt(language)

        # Build context showing what was gathered
        context_parts = [f"**User Query:** {user_query}"]
        context_parts.append(f"**Agents Called:** {', '.join(agents_called)}")

        for agent_name, output in agent_outputs.items():
            # Truncate very long outputs to avoid token limits
            truncated = output[:4000] if len(output) > 4000 else output
            context_parts.append(
                f"\n**{agent_name.upper()} Agent Output:**\n{truncated}"
            )

        context = "\n".join(context_parts)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"# VALIDATION TASK\n\nValidate completeness of the following data:\n\n{context}"),
        ]

        # LLM call with retry for Azure content filter false positives
        response = self._safe_llm_invoke(self.llm, messages)
        content = clean_response(response.content, _print=False)

        # Parse JSON response
        result = parse_json_response(content)

        if result:
            # Normalize the result structure
            return {
                "complete": result.get("complete", True),
                "missing_data": result.get("missing_data", []),
                "required_agents": [
                    a for a in result.get("required_agents", [])
                    if a in ("weather", "transport", "researcher")
                ],
                "reasoning": result.get("reasoning", ""),
                "disclaimers": result.get("disclaimers", []),
            }

        # Fallback: if JSON parsing fails, assume complete (don't block response)
        return {
            "complete": True,
            "missing_data": [],
            "required_agents": [],
            "reasoning": "QA validation could not parse LLM response; assuming complete.",
            "disclaimers": [],
        }


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 QA Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        agent = QualityAssuranceAgent()
        print(f"\n\033[1m✅ QA Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"   Tools: {len(agent.tools)} (QA has no tools)")

        # Test 1: Incomplete planning query
        print("\n\033[1m📝 Test 1: Incomplete planning query\033[0m")
        result = agent.validate(
            user_query="Plan my day tomorrow in Lisbon",
            agent_outputs={
                "weather": "Tomorrow: 18°C, sunny, no rain expected.",
                "researcher": "1. Museu do Azulejo\n2. Castelo de São Jorge\n3. Belém Tower",
            },
            agents_called=["weather", "researcher"],
            language="en",
        )
        print(f"   Complete: {result['complete']}")
        print(f"   Missing: {result['missing_data']}")
        print(f"   Required agents: {result['required_agents']}")
        print(f"   Reasoning: {result['reasoning']}")

        # Test 2: Complete weather query
        print("\n\033[1m📝 Test 2: Complete weather query\033[0m")
        result = agent.validate(
            user_query="What's the weather today?",
            agent_outputs={
                "weather": "Today: 22°C max, 14°C min. Sunny. No rain. Wind: Moderate from NW.",
            },
            agents_called=["weather"],
            language="en",
        )
        print(f"   Complete: {result['complete']}")
        print(f"   Missing: {result['missing_data']}")
        print(f"   Reasoning: {result['reasoning']}")

        print("\n\033[1;32m✅ QA Agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
