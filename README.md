# bc-bench

Benchmark suite for evaluating retrieval quality, latency, and diversity of AI agent context systems.

## Usage

The generated benchmark dataset is stored at `assets/sample_data/bc-bench.json`.

bc-bench has commands: `generate`, `evaluate`.

### Evaluate (measure retrieval quality)

```bash
cd ~/workspace/your-project
uv run python -m bc_bench evaluate --input datasets/your-project/ground_truth.json --dry-run
```

## Metrics

| Metric | What it Measures |
|--------|------------------|