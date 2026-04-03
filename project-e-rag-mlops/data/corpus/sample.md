# Sample product documentation

This section describes the monitoring stack for the RAG service. Latency and error rates are exported as Prometheus metrics at the `/metrics` endpoint.

## Retrieval

Hybrid retrieval combines lexical and dense signals. For this demo project we use TF-IDF over chunked Markdown for a lightweight baseline.

## Feedback

Users can submit thumbs up or down with optional corrections. Store these events for labeling pipelines and periodic quality reviews.

## Refresh jobs

A scheduled job rebuilds the index when new documents land in object storage. Pair refreshes with drift checks on retrieval quality.
