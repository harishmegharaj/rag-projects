# Drift detection and index refresh (Project E)

## Scheduled refresh

`scripts/run_refresh_job.py` rebuilds the TF-IDF matrix and `chunks.json` from everything under `data/corpus/` (or `CORPUS_DIR`). Schedule it when:

- New Markdown or PDF-derived text lands in blob storage.
- You promote a new documentation release.

In production, mount corpus from a volume or sync from S3 before running the job.

## What “drift” means for RAG

| Drift type | Symptom | What to measure |
| --- | --- | --- |
| **Corpus drift** | User questions reference policies or products not in the index | Rising “no hit” rate, support tickets, or low retrieval scores |
| **Query drift** | Language or intent shifts (new product names, abbreviations) | Clustering of embedding vectors for queries over time vs. training slice |
| **Quality drift** | Right chunks retrieved but answers still wrong | Downstream LLM evals, thumbs-down rate, human spot checks |

## Practical signals (retrieval quality)

1. **Retrieval hit rate** — fraction of queries with at least one chunk above a similarity threshold (this repo exposes chunk counts via metrics and logs).
2. **nDCG / MRR** — if you maintain a labeled eval set of `(query, relevant_chunk_ids)`, re-score after each index version.
3. **Embedding distance** — for dense retrieval, track average distance from query embedding to top-1 chunk; sudden shifts can indicate corpus or query drift.
4. **Feedback loop** — thumbs down + free-text corrections stored in SQLite (`FEEDBACK_DB`) for offline review and golden-set expansion.

## When to re-embed vs. re-chunk

- **Re-chunk** when document structure changes (sections too large, mixed topics).
- **Re-embed** when you change embedding models or fine-tune; keep a version tag on the vector store.

This demo uses TF-IDF only; swapping in dense embeddings (e.g. `text-embedding-3-small`) does not change the monitoring pattern—only the latency and cost profiles.
