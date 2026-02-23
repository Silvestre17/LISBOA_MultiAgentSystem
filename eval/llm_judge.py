import json
import os
from enum import IntEnum
from typing import cast

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.llm_factory import LLMFactory


class LikertScore(IntEnum):
    STRONGLY_DISAGREE = 1
    DISAGREE = 2
    NEUTRAL = 3
    AGREE = 4
    STRONGLY_AGREE = 5


class LLMJudgeScore(BaseModel):
    factual_accuracy: int = Field(..., description="Score 1-5: A 5 means ALL facts and routes are strictly supported by the RETRIEVED CONTEXT. If the agent hallucinates data not in the context, score 1 or 2.", ge=1, le=5)
    tool_usage: int = Field(..., description="Score 1-5: The tools selected were correct for the query, with no redundant or missing invocations.", ge=1, le=5)
    completeness: int = Field(..., description="Score 1-5: The response addresses all aspects of the user query.", ge=1, le=5)
    relevance: int = Field(..., description="Score 1-5: No extraneous, hallucinated, or off-topic information is included.", ge=1, le=5)
    response_quality: int = Field(..., description="Score 1-5: The text is clear, well-structured, and practically useful.", ge=1, le=5)
    reasoning: str = Field(..., description="A short, 2-sentence justification for the given scores.")

    def get_composite_score(self) -> float:
        return (self.factual_accuracy + self.tool_usage + self.completeness + self.relevance + self.response_quality) / 5.0


class LLMJudge:
    """
    Implements the LLM-as-a-Judge protocol defined in Methodology Section 3.6.2.
    Evaluates system responses against reference answers using structured rubrics.
    """
    def __init__(self, provider: str = "azure", model_name: str = "gpt-4o"):
        # Use our centralized LLM Factory to route through Azure or LM Studio
        base_llm = LLMFactory.get_llm(
            provider=provider,
            model=model_name,
            temperature=0.0
        )
        self.llm = base_llm.with_structured_output(LLMJudgeScore)
        
        self.prompt = PromptTemplate.from_template(
            """You are an impartial academic judge evaluating an AI agent's response for an urban mobility and tourism system (LISBOA).
            You will evaluate the system's generated response based on 5 parameters on a Likert scale of 1 to 5.
            1 = Strongly Disagree
            2 = Disagree
            3 = Neutral
            4 = Agree
            5 = Strongly Agree

            --- USER QUERY ---
            {query}

            --- EXPECTED FACTS TO BE PRESENT ---
            {expected_facts}
            
            --- RETRIEVED CONTEXT (Raw Output from Tools) ---
            {retrieved_context}

            --- ACTUAL TOOLS USED BY AGENT ---
            {actual_tools}

            --- GENERATED RESPONSE ---
            {response}

            Evaluate the response strictly following the descriptions of the 5 parameters.
            CRITICAL: For Factual Accuracy, you must verify that the GENERATED RESPONSE does not invent places, temperatures, or transport lines not found in the RETRIEVED CONTEXT.
            If the query is a greeting, out of scope, or an edge case where no facts or tools are expected, score highly if the agent handled it gracefully and politely directly using its parametric knowledge.
            """
        )

    def evaluate(self, query: str, expected_facts: list[str], expected_tools: list[str], actual_tools: list[str], retrieved_context: str, response: str) -> dict:
        """
        Runs the LLM as a judge on a single query-response pair.
        Returns a dictionary containing the scores and reasoning.
        """
        facts_str = "\n".join([f"- {i}" for i in expected_facts]) if expected_facts else "None expected."
        exp_tools_str = ", ".join(expected_tools) if expected_tools else "None expected."
        act_tools_str = ", ".join(actual_tools) if actual_tools else "None used."
        
        prompt_val = self.prompt.invoke({
            "query": query,
            "expected_facts": facts_str,
            "expected_tools": exp_tools_str,
            "retrieved_context": retrieved_context if retrieved_context.strip() else "No tools called / No context retrieved.",
            "actual_tools": act_tools_str,
            "response": response
        })
        
        try:
            result = cast(LLMJudgeScore, self.llm.invoke(prompt_val))
            return {
                "factual_accuracy": result.factual_accuracy,
                "tool_usage": result.tool_usage,
                "completeness": result.completeness,
                "relevance": result.relevance,
                "response_quality": result.response_quality,
                "composite_score": result.get_composite_score(),
                "reasoning": result.reasoning
            }
        except Exception as e:
            print(f"Error during LLM judgment: {e}")
            return {
                "factual_accuracy": 0, "tool_usage": 0, "completeness": 0, "relevance": 0, "response_quality": 0,
                "composite_score": 0.0, "reasoning": f"Judge Failed: {str(e)}"
            }


if __name__ == "__main__":
    # Smoke test (Warning: Costs OpenAI credits)
    import sys
    if "--test" in sys.argv:
        judge = LLMJudge(provider="azure", model_name="gpt-4o")
        res = judge.evaluate(
            query="What is the weather?",
            expected_facts=["Current temperature"],
            expected_tools=["get_current_weather_summary"],
            actual_tools=["get_current_weather_summary"],
            retrieved_context="{'temp': 15, 'condition': 'Sunny'}",
            response="The temperature is 15°C."
        )
        print(json.dumps(res, indent=2))
