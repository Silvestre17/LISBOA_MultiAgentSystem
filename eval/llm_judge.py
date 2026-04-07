# ==========================================================================
# Master Thesis - LLM-as-a-Judge
#   - André Filipe Gomes Silvestre, 20240502
#
# Implements the LLM-as-a-Judge protocol for evaluating LISBOA system
# responses. Uses structured Pydantic output with chain-of-thought
# reasoning BEFORE scoring (bias mitigation per 2025 best practices).
#
# References:
#   - Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench"
#   - arXiv:2511.21140v3 "How to Correctly Report LLM-as-a-Judge"
#   - Best practices: temperature=0, CoT-first, explicit rubric descriptors
# ==========================================================================

import json
import os
import re
from typing import Any, cast

from agent.llm_factory import LLMFactory
from eval.runtime_utils import (
    build_cost_payload,
    build_model_id,
    build_usage_payload,
    combine_usage_payloads,
)
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic Score Model (Chain-of-Thought FIRST, then scores)
# ---------------------------------------------------------------------------


class LLMJudgeScore(BaseModel):
    """Structured output for LLM-as-a-Judge evaluation.

    IMPORTANT: `reasoning` is placed FIRST to force the judge to think
    through its evaluation before committing to numeric scores. This
    chain-of-thought-first approach reduces mean-reversion bias on Likert
    scales (scores clustering around 3-4).
    """

    reasoning: str = Field(
        ...,
        description=(
            "Write a 3-5 sentence analysis BEFORE assigning scores. "
            "For each dimension, note specific evidence from the RETRIEVED "
            "CONTEXT and GENERATED RESPONSE. Mention any hallucinated facts, "
            "missing tools, or quality issues you found."
        ),
    )
    factual_accuracy: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "1 = Response contains fabricated facts NOT in RETRIEVED CONTEXT "
            "(hallucinated stations, temperatures, events). "
            "2 = Multiple factual errors or unsupported claims. "
            "3 = Mostly accurate but with minor unsupported details. "
            "4 = All major facts are grounded in context with trivial omissions. "
            "5 = Every fact and route is strictly supported by RETRIEVED CONTEXT."
        ),
    )
    tool_usage: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Compare EXPECTED TOOLS vs ACTUAL TOOLS USED. "
            "1 = Completely wrong tools or no tools when tools were required. "
            "2 = Some correct tools but critical ones missing. "
            "3 = Most tools correct but with redundant or missing invocations. "
            "4 = All expected tools used, minor redundancy acceptable. "
            "5 = Perfect tool selection: all expected tools used, no redundant calls."
        ),
    )
    completeness: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "1 = Response ignores the query or answers something else entirely. "
            "2 = Addresses the query but misses most EXPECTED FACTS. "
            "3 = Covers the main point but omits important details. "
            "4 = Addresses all major aspects with minor omissions. "
            "5 = Fully addresses every aspect of the query and all EXPECTED FACTS."
        ),
    )
    relevance: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "1 = Response is entirely off-topic or filled with irrelevant information. "
            "2 = Mostly irrelevant with some tangential connection to the query. "
            "3 = Relevant but includes notable extraneous content. "
            "4 = Focused on the query with minimal unnecessary information. "
            "5 = Perfectly focused: no extraneous, hallucinated, or off-topic content."
        ),
    )
    response_quality: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "1 = Incoherent, poorly formatted, or practically unusable. "
            "2 = Understandable but poorly structured or confusing. "
            "3 = Adequate structure and clarity but could be improved. "
            "4 = Well-structured, clear, and practically useful. "
            "5 = Excellent: clear, well-organized, with appropriate formatting."
        ),
    )

    def get_composite_score(self) -> float:
        """Weighted average of all 5 dimensions (equal weights)."""
        return (
            self.factual_accuracy
            + self.tool_usage
            + self.completeness
            + self.relevance
            + self.response_quality
        ) / 5.0


# ---------------------------------------------------------------------------
# Judge Prompt (with explicit rubric and CoT instructions)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are an impartial academic judge evaluating an AI agent's response for LISBOA, an urban mobility and tourism system for the Lisbon Metropolitan Area.

You will evaluate the response on 5 dimensions using a 1-5 Likert scale.

CRITICAL INSTRUCTIONS:
1. Write your reasoning FIRST (3-5 sentences). Analyze specific evidence before assigning any score.
2. For Factual Accuracy: verify that facts in the GENERATED RESPONSE are supported by the RETRIEVED CONTEXT. If the agent invents details NOT present in the context, lower the score accordingly.
3. For Tool Usage: compare EXPECTED TOOLS against ACTUAL TOOLS USED. Minor redundancies are acceptable if the expected tools are present.
4. If the query is a greeting, out-of-scope, or an edge case where no facts or tools are expected, score highly (5) if the agent handled it gracefully.
5. **Length Bias Mitigation**: Evaluate based on accuracy and conciseness. A concise, accurate answer is excellent. Do not penalize for being direct.
6. **Position Bias Mitigation**: Evaluate the entire response equally. Make sure to read all the way to the end.

--- CALIBRATION GUIDELINES ---
- Score 5: Excellent, accurate, uses the right tools, and addresses the query.
- Score 4: Good, mostly accurate, might have minor omissions or redundant tools.
- Score 3: Acceptable, addresses the main point but with notable omissions or minor errors.
- Score 2: Poor, contains significant factual errors, missed critical tools, or poor formatting.
- Score 1: Complete failure, hallucinates heavily, or ignores the query completely.

--- USER QUERY ---
{query}

--- EXPECTED FACTS TO BE PRESENT ---
{expected_facts}

--- EXPECTED TOOLS ---
{expected_tools}

--- ACTUAL TOOLS USED BY AGENT ---
{actual_tools}

--- RETRIEVED CONTEXT (Raw Output from Tools) ---
{retrieved_context}

--- GENERATED RESPONSE ---
{response}

Now write your reasoning FIRST, then assign scores for each dimension following the rubric descriptions."""


# ---------------------------------------------------------------------------
# LLMJudge Class
# ---------------------------------------------------------------------------


class LLMJudge:
    """Implements the LLM-as-a-Judge protocol defined in Methodology Section 3.6.2.

    Evaluates system responses against reference answers using structured
    rubrics with chain-of-thought reasoning before scoring.

    Bias mitigations applied:
        - Chain-of-thought FIRST (reasoning before scores)
        - Explicit rubric descriptors for each score level
        - Temperature = 0.0 for reproducibility
        - Structured Pydantic output to prevent format drift
        - Separate judge model from generator model recommended
    """

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        """Initialize the LLM Judge.

        Args:
            provider: LLM provider (azure, openai, lmstudio).
            model_name: Model to use as judge. Should differ from the
                generator model to avoid self-preference bias.
                Defaults to EVAL_JUDGE_MODEL_NAME if set, otherwise gpt-5.4-mini.
        """
        # TEST: Set the evaluator model here for thesis experiments.
        # Recommended workflow: keep these environment variables explicit so the
        # saved JSON artefacts always distinguish generator vs evaluator model.
        provider = provider or os.getenv("EVAL_JUDGE_PROVIDER", "azure")
        model_name = model_name or os.getenv("EVAL_JUDGE_MODEL_NAME", "gpt-5.4-mini")

        base_llm = LLMFactory.get_llm(
            provider=provider,
            model=model_name,
            temperature=0.0,
        )
        self.base_llm = base_llm
        self.llm = base_llm.with_structured_output(LLMJudgeScore, include_raw=True)

        self.prompt = PromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
        self.provider = provider
        self.model_name = model_name

    def _build_usage_payload(self, usage_source: Any) -> dict[str, Any]:
        """Normalize usage metadata for one judge invocation attempt."""
        evaluation_model_id = build_model_id(self.provider, self.model_name)
        return build_usage_payload(
            LLMFactory.extract_usage_metadata(usage_source),
            model_id=evaluation_model_id,
            call_count=1,
        )

    @staticmethod
    def _coerce_judge_score(parsed_result: Any) -> LLMJudgeScore:
        """Normalize structured or JSON-decoded payloads into ``LLMJudgeScore``."""
        if isinstance(parsed_result, LLMJudgeScore):
            return parsed_result
        if isinstance(parsed_result, BaseModel):
            return LLMJudgeScore.model_validate(parsed_result.model_dump())
        if isinstance(parsed_result, dict):
            return LLMJudgeScore.model_validate(parsed_result)
        raise TypeError(f"Unsupported judge payload type: {type(parsed_result)!r}")

    @staticmethod
    def _extract_response_text(response_obj: Any) -> str:
        """Extract textual content from a raw model response object."""
        if response_obj is None:
            return ""
        if hasattr(response_obj, "content"):
            content = getattr(response_obj, "content")
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content)
        if isinstance(response_obj, dict):
            if "content" in response_obj:
                return str(response_obj["content"])
            if "text" in response_obj:
                return str(response_obj["text"])
        return str(response_obj)

    @staticmethod
    def _extract_json_candidate(text: str) -> dict[str, Any]:
        """Extract the first plausible JSON object from an LLM response body."""
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("Judge fallback returned empty content.")

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized_text, re.DOTALL)
        json_candidate = fenced_match.group(1) if fenced_match else normalized_text
        if not fenced_match:
            start = json_candidate.find("{")
            end = json_candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Judge fallback did not return a JSON object.")
            json_candidate = json_candidate[start:end + 1]

        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", json_candidate)
        return cast(dict[str, Any], json.loads(sanitized))

    def _try_parse_raw_response(self, raw_response: Any) -> LLMJudgeScore | None:
        """Attempt to salvage a structured score from an unparsable raw response."""
        response_text = self._extract_response_text(raw_response)
        if not response_text.strip():
            return None
        try:
            return self._coerce_judge_score(self._extract_json_candidate(response_text))
        except Exception:
            return None

    def _invoke_json_fallback(self, prompt_val: Any) -> tuple[LLMJudgeScore, dict[str, Any]]:
        """Run a second-pass plain-text JSON judge request when structured output fails."""
        fallback_prompt = (
            f"{prompt_val.to_string() if hasattr(prompt_val, 'to_string') else str(prompt_val)}\n\n"
            "Return ONLY valid JSON with these keys: reasoning, factual_accuracy, tool_usage, "
            "completeness, relevance, response_quality. Each score must be an integer from 1 to 5. "
            "Do not wrap the JSON in markdown fences."
        )
        fallback_raw = self.base_llm.invoke(fallback_prompt)
        fallback_usage = self._build_usage_payload(fallback_raw)
        fallback_payload = self._extract_json_candidate(self._extract_response_text(fallback_raw))
        return self._coerce_judge_score(fallback_payload), fallback_usage

    def evaluate(
        self,
        query: str,
        expected_facts: list[str],
        expected_tools: list[str],
        actual_tools: list[str],
        retrieved_context: str,
        response: str,
        pricing_by_model: dict[str, Any] | None = None,
    ) -> dict:
        """Run the LLM judge on a single query-response pair.

        Args:
            query: The user's original query.
            expected_facts: Ground truth facts that should appear in the response.
            expected_tools: Tools that should have been called.
            actual_tools: Tools that were actually called by the agent.
            retrieved_context: Raw output from tool calls.
            response: The agent's final generated response.
            pricing_by_model: Optional pricing catalog keyed by model name or
                provider::model, with prices in USD per million tokens.

        Returns:
            Dict with scores (1-5) for each dimension, composite_score,
            and reasoning.
        """
        facts_str = (
            "\n".join([f"- {f}" for f in expected_facts])
            if expected_facts
            else "None expected (greeting/edge case)."
        )
        exp_tools_str = ", ".join(expected_tools) if expected_tools else "None expected."
        act_tools_str = ", ".join(actual_tools) if actual_tools else "None used."

        prompt_val = self.prompt.invoke({
            "query": query,
            "expected_facts": facts_str,
            "expected_tools": exp_tools_str,
            "actual_tools": act_tools_str,
            "retrieved_context": (
                retrieved_context.strip()
                if retrieved_context and retrieved_context.strip()
                else "No tools called / No context retrieved."
            ),
            "response": response,
        })

        evaluation_model_id = build_model_id(self.provider, self.model_name)
        usage_payloads: list[dict[str, Any]] = []

        try:
            try:
                raw_result = self.llm.invoke(prompt_val)
            except Exception:
                fallback_result, fallback_usage = self._invoke_json_fallback(prompt_val)
                usage_payloads.append(fallback_usage)
                evaluation_usage = combine_usage_payloads(usage_payloads)
                evaluation_cost = build_cost_payload(
                    evaluation_usage,
                    pricing_by_model,
                    model_id=evaluation_model_id,
                )
                return {
                    "factual_accuracy": fallback_result.factual_accuracy,
                    "tool_usage": fallback_result.tool_usage,
                    "completeness": fallback_result.completeness,
                    "relevance": fallback_result.relevance,
                    "response_quality": fallback_result.response_quality,
                    "composite_score": fallback_result.get_composite_score(),
                    "reasoning": fallback_result.reasoning,
                    "evaluation_usage": evaluation_usage,
                    "evaluation_cost_usd": evaluation_cost,
                }

            parsed_result = raw_result
            usage_source = raw_result

            if isinstance(raw_result, dict):
                parsing_error = raw_result.get("parsing_error")
                parsed_result = raw_result.get("parsed")
                usage_source = raw_result.get("raw", raw_result)
                usage_payloads.append(self._build_usage_payload(usage_source))

                if parsing_error or parsed_result is None:
                    parsed_result = self._try_parse_raw_response(usage_source)
                    if parsed_result is None and parsing_error:
                        raise parsing_error
            else:
                usage_payloads.append(self._build_usage_payload(usage_source))

            try:
                result = self._coerce_judge_score(parsed_result)
            except Exception:
                fallback_result, fallback_usage = self._invoke_json_fallback(prompt_val)
                usage_payloads.append(fallback_usage)
                result = fallback_result

            evaluation_usage = combine_usage_payloads(usage_payloads)
            evaluation_cost = build_cost_payload(
                evaluation_usage,
                pricing_by_model,
                model_id=evaluation_model_id,
            )

            return {
                "factual_accuracy": result.factual_accuracy,
                "tool_usage": result.tool_usage,
                "completeness": result.completeness,
                "relevance": result.relevance,
                "response_quality": result.response_quality,
                "composite_score": result.get_composite_score(),
                "reasoning": result.reasoning,
                "evaluation_usage": evaluation_usage,
                "evaluation_cost_usd": evaluation_cost,
            }
        except Exception as e:
            print(f"Error during LLM judgment: {e}")
            empty_usage = (
                combine_usage_payloads(usage_payloads)
                if usage_payloads
                else build_usage_payload({}, model_id=evaluation_model_id, call_count=1)
            )
            return {
                "factual_accuracy": 0,
                "tool_usage": 0,
                "completeness": 0,
                "relevance": 0,
                "response_quality": 0,
                "composite_score": 0.0,
                "reasoning": f"Judge Failed: {str(e)}",
                "evaluation_usage": empty_usage,
                "evaluation_cost_usd": build_cost_payload(
                    empty_usage,
                    pricing_by_model,
                    model_id=evaluation_model_id,
                ),
            }


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        judge = LLMJudge()
        res = judge.evaluate(
            query="What is the weather?",
            expected_facts=["Current temperature"],
            expected_tools=["get_current_weather_summary"],
            actual_tools=["get_current_weather_summary"],
            retrieved_context="{'temp': 15, 'condition': 'Sunny'}",
            response="The temperature is 15°C and it's sunny in Lisbon.",
        )
        print(json.dumps(res, indent=2))
