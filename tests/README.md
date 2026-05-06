# LISBOA Lean Test Policy

This folder is intentionally small. Pytest is a safety net for stable code
contracts, not the quality gate for user-facing answers.

Keep tests here only when they validate one of these stable behaviours:

- startup, provider, or configuration contracts;
- deterministic parser, resolver, formatter, or tool helper behaviour;
- app control-flow helpers that cannot be checked well through a single prompt;
- optional live API smoke checks marked with `@pytest.mark.live`.

Do not add tests that freeze complete assistant answers, exact prompt wording,
temporary audit examples, or one-off regressions from a single query. For
agents, prompts, routing, QA, or final response quality, validate with real
system prompts through `scripts/run_prompts.py` and, when rendering matters,
with Streamlit/browser screenshots.

Useful commands:

```powershell
python -X utf8 -m pytest tests/ eval/tests/ -q
python -X utf8 -m pytest tests/test_lisbon_transport.py -q --run-live -m live
python -X utf8 (Join-Path $root 'scripts\run_prompts.py') --prompt "How do I get from Baixa-Chiado to Aeroporto?" --language en --quiet
```
