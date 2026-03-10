# ===========================================================================
# Master Thesis - Tool Coverage Manifest Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Fast validation for the prompt coverage manifest used by strict live suites.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_tool_coverage_manifest.py -q
# Useful parameters:
#   -vv             verbose mode
#   -k manifest     focus on manifest checks only
#   -x              stop on first failure
#   --tb=short      shorter tracebacks
# Notes:
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ===========================================================================

from __future__ import annotations

VALID_DOMAINS = {"weather", "transport", "researcher"}
REQUIRED_FIELDS = {"id", "domain", "language", "query", "expected_tools"}


class TestToolCoverageManifest:
    """Fast integrity checks for the live coverage manifest."""

    def test_manifest_has_required_fields(self, coverage_manifest):
        for item in coverage_manifest:
            missing = REQUIRED_FIELDS - set(item.keys())
            assert not missing, f"{item.get('id', '?')} missing fields: {sorted(missing)}"

    def test_manifest_domains_are_valid(self, coverage_manifest):
        invalid = [item["id"] for item in coverage_manifest if item["domain"] not in VALID_DOMAINS]
        assert not invalid, f"Invalid manifest domains: {invalid}"

    def test_manifest_expected_tools_exist(self, coverage_manifest, exported_tool_names):
        invalid = []
        for item in coverage_manifest:
            for tool_name in item.get("expected_tools", []):
                if tool_name not in exported_tool_names:
                    invalid.append(f"{item['id']}: {tool_name}")
        assert not invalid, "Invalid tool references in manifest:\n" + "\n".join(invalid)

    def test_manifest_covers_all_exported_tools(self, coverage_manifest, exported_tool_names):
        covered = {tool for item in coverage_manifest for tool in item.get("expected_tools", [])}
        missing = sorted(exported_tool_names - covered)
        assert not missing, "Manifest does not cover all exported tools:\n" + "\n".join(missing)

    def test_manifest_ids_are_unique(self, coverage_manifest):
        ids = [item["id"] for item in coverage_manifest]
        assert len(ids) == len(set(ids)), "Coverage manifest contains duplicate IDs"
