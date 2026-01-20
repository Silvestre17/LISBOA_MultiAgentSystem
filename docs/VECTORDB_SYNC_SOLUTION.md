# Vector Database Sync Solution

**Problem:** GitHub Actions workflow timed out after 37 minutes when processing large document updates (1171 places).

**Root Cause:** Python script processed all documents in a single run without batch limiting.

**Context:** This workflow maintains the knowledge base for the **Researcher Agent**, ensuring it has access to the latest events and places from VisitLisboa.

---

## Solution: Incremental Batch Processing

### 1. Python Script Changes (`tools/vector_store.py`)

**Added `--max-docs` parameter:**
```python
parser.add_argument('--max-docs', type=int, default=200,
                   help='Max documents to process per run (default: 200)')
```

**Modified `_sync_json_collection()` to limit batches:**
```python
# Limit number of documents to add per run
if max_docs and len(ids_to_add) > max_docs:
    has_more_work = True
    pending = len(ids_to_add) - max_docs
    ids_to_add = ids_to_add[:max_docs]
```

**Exit code system:**
- **0** = All work complete
- **2** = More work pending (processed max_docs, more remain)
- **Other** = Error

**Quota management:**
- If places collection uses the quota, events are skipped (not limited)
- Ensures fair distribution of processing time

---

### 2. GitHub Actions Workflow Changes (`.github/workflows/sync_vector_db.yml`)

**Reduced timeout:**
```yaml
timeout-minutes: 90  # Down from 360 (6 hours)
```

**Added max_docs input:**
```yaml
max_docs:
  description: 'Max docs per run (default: 200)'
  default: '200'
  type: string
```

**Improved loop logic:**
```bash
for i in $(seq 1 $MAX_ITERATIONS); do
  $SYNC_CMD
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    # All done
    break
  elif [ $EXIT_CODE -eq 2 ]; then
    # More work pending
    continue
  else
    # Error
    exit $EXIT_CODE
  fi
done
```

---

## Performance Characteristics

### Expected Runtimes

| Scenario | Documents | Iterations | Time per Iteration | Total Time |
|----------|-----------|------------|-------------------|------------|
| Daily Events | ~10-50 | 1 | ~2-5 min | ~5 min |
| Weekly Places (small) | ~100 | 1 | ~15 min | ~15 min |
| Weekly Places (large) | ~1171 | 6 | ~17 min | ~102 min |
| Full Rebuild | ~1171 | 6 | ~20 min | ~120 min |

### Batch Capacity

- **Per iteration:** 200 documents (~15-20 min)
- **Per workflow run:** 2000 documents (10 × 200)
- **Per day:** Unlimited (multiple workflow runs)

---

## Usage Examples

### Automatic Sync (Default)
Triggered after "Update Lisbon Data" workflow completes:
- Processes 200 docs per batch
- Loops up to 10 times (2000 docs max per run)
- If more work remains, re-run workflow

### Manual Sync with Custom Batch Size
```yaml
workflow_dispatch:
  inputs:
    max_docs: '500'  # Process 500 docs per iteration
```

### Force Rebuild
```yaml
workflow_dispatch:
  inputs:
    rebuild_places: true
    max_docs: '100'  # Smaller batches for rebuild
```

---

## Monitoring

### Check Workflow Progress
1. Go to **Actions** tab in GitHub
2. Select **Sync Vector Database** workflow
3. Check logs for:
   - Batch numbers (e.g., "Batch 3/10")
   - Exit codes (0 = done, 2 = pending)
   - Pending document counts

### Expected Output
```
────────────────────────────────────────────────────────────
📦 Batch 1/10 (max 200 docs per batch)
────────────────────────────────────────────────────────────

Syncing Places Collection...
  ✅ Added: 200
  ⚠️ Pending: 971 documents remain

Exit code: 2
⏳ More work pending, continuing...

────────────────────────────────────────────────────────────
📦 Batch 2/10 (max 200 docs per batch)
────────────────────────────────────────────────────────────
...
```

---

## Troubleshooting

### Workflow Still Times Out
**Cause:** Batch size too large for available compute  
**Solution:** Reduce `max_docs` to 100 or 150

### Max Iterations Reached
**Cause:** More than 2000 documents changed  
**Solution:** Re-run workflow to continue processing

### Exit Code 143 (SIGTERM)
**Cause:** Workflow timeout (90 min exceeded)  
**Solution:** Reduce `max_docs` parameter

---

## Technical Details

### Why 200 Documents Per Batch?
- Based on observed processing time: ~17 seconds/document
- 200 docs × 17s = ~3400s = ~57 minutes
- Leaves buffer for cache loading, git operations (~15-20 min total)
- Stays well under 90-minute timeout

### Why Exit Code 2?
- POSIX standard exit codes:
  - 0 = Success
  - 1 = General error
  - 2+ = Custom (we use 2 for "partial success, more work pending")
- Allows bash loop to distinguish completion from continuation

### Why Not Process Everything?
- GitHub Actions free tier: 2000 minutes/month
- Processing 1171 docs in one run: ~120 minutes (6% of monthly quota)
- Batch processing: Only runs when needed, stops early if no changes
- Better resource utilization and fault tolerance

---

## Next Steps

1. **Test with real data update:**
   - Update places.json with new entries
   - Observe workflow processing batches
   - Verify completion without timeout

2. **Monitor monthly usage:**
   - Check GitHub Actions usage dashboard
   - Adjust max_docs if approaching quota limits

3. **Consider optimizations:**
   - Cache vector embeddings for unchanged documents
   - Use GPU-enabled runners for faster processing (costs money)
   - Implement delta encoding for smaller commits
