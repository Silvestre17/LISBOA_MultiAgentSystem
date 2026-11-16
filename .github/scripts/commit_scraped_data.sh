#!/usr/bin/env bash
# ==========================================================================
# Master Thesis - Commit Scraped VisitLisboa Data (GitHub Actions)
#   - André Filipe Gomes Silvestre, 20240502
#
# Commits and pushes the scraper outputs when they changed. Called after each
# scraper, so the Events data is saved even if the Places run fails later.
#
# Usage: commit_scraped_data.sh "<commit label>"
# ==========================================================================

set -euo pipefail

LABEL="${1:-Data}"
PLACES_MARKER="data_collection/webscraping/places_last_run.txt"

git config --global user.name "github-actions[bot]"
git config --global user.email "github-actions[bot]@users.noreply.github.com"

git add data_collection/webscraping/*.json
if [ -f "$PLACES_MARKER" ]; then
  git add "$PLACES_MARKER"
fi

if git diff --staged --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "Data Auto-Update: $(date -u +'%Y-%m-%d') ($LABEL)"
git pull --rebase origin "${GITHUB_REF_NAME:-main}"
git push
