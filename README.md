# AutoRAGResearch

Autonomous RAG pipeline optimization system. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an LLM agent iteratively tunes RAG hyperparameters (chunking, embeddings, retrieval) and evaluates with RAGAS metrics until a target quality threshold is reached.

## Architecture

```
program.md (config)
      |
      v
+--------------+     +---------------+     +-------------+
| Data Sources |---->| RAG Pipeline  |---->|  Evaluator   |
|  (loaders)   |     | chunk/embed/  |     |   (RAGAS)    |
+--------------+     |   retrieve    |     +------+------+
                     +---------------+            |
                            ^                     v
                            |           +-----------------+
                     +------+-------+   |   LLM Agent     |
                     | experiment   |<--| analyze scores,  |
                     | _config.json |   | suggest config   |
                     +--------------+   +-----------------+
                                               |
                             +-----------------+-----------------+
                             v                 v                 v
                       agent_notes.md   experiment_       best_config.json
                       (reasoning)      history.jsonl     (git checkpoint)
                                        (MLflow)
```

**Loop flow:** Load data -> run RAG pipeline with current config -> evaluate with RAGAS -> log to MLflow -> if improved, git checkpoint -> LLM agent suggests next config -> repeat.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (for LLM and embeddings)

### Installation

```bash
git clone https://github.com/shivamshinde123/Auto-RAG-Research.git
cd Auto-RAG-Research
uv sync
```

### Set API Keys

```bash
export OPENAI_API_KEY="sk-..."

# Optional -- for specific data sources:
export NOTION_API_KEY="ntn_..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### Configure

Edit `program.md` to define your experiment:

```markdown
## Data Sources

[[data_sources]]
type: huggingface
enabled: true
dataset_name: squad
split: validation
sample_size: 50

## Search Space
chunk_size: [256, 512, 1024]
chunk_overlap: [25, 50, 100]
top_k: [3, 5, 10]
embedding_model: [all-MiniLM-L6-v2, BGE-large, text-embedding-ada-002]
llm_model: [gpt-4o-mini, gpt-3.5-turbo]

## Optimization Target
primary_metric: context_recall
secondary_metric: faithfulness
min_threshold: 0.80

## Constraints
max_iterations: 20
max_cost_usd: 5.0

## Experiment
experiment_name: my_experiment
git_checkpoints: true
```

### Run

```bash
# Validate config without running
uv run python main.py --dry-run

# Run experiment
uv run python main.py --config program.md

# Resume interrupted experiment
uv run python main.py --config program.md --resume
```

### View Results

```bash
uv run mlflow ui
# Open http://localhost:5000
```

## Data Sources

| Source | Type | Config Fields | Credentials |
|--------|------|--------------|-------------|
| Local PDF | `local_pdf` | `path` | None |
| Local TXT | `local_txt` | `path` | None |
| Local CSV | `local_csv` | `path`, `text_column` | None |
| Google Drive | `gdrive` | `folder_id`, `credentials_path` | OAuth2 credentials.json in `.secrets/` |
| AWS S3 | `s3` | `bucket`, `prefix` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` env vars |
| Notion | `notion` | `database_id` | `NOTION_API_KEY` env var |
| Web URLs | `web` | `urls` (list) | None |
| HuggingFace | `huggingface` | `dataset_name`, `split`, `sample_size` | None (public datasets) |

### Example: Multiple Sources

```markdown
[[data_sources]]
type: local_pdf
enabled: true
path: ./data/papers

[[data_sources]]
type: huggingface
enabled: true
dataset_name: squad
split: validation
sample_size: 100
```

## Configuration Reference

### Search Space

| Parameter | Description | Example |
|-----------|-------------|---------|
| `chunk_size` | Document chunk sizes to try | `[256, 512, 1024]` |
| `chunk_overlap` | Overlap between chunks | `[25, 50, 100]` |
| `top_k` | Number of retrieved chunks | `[3, 5, 10]` |
| `embedding_model` | Embedding models | `[all-MiniLM-L6-v2, BGE-large]` |
| `llm_model` | LLM for generation | `[gpt-4o-mini, gpt-3.5-turbo]` |

### Optimization Target

| Parameter | Description | Default |
|-----------|-------------|---------|
| `primary_metric` | Main metric to optimize | `context_recall` |
| `secondary_metric` | Secondary metric | `faithfulness` |
| `min_threshold` | Stop when composite >= this | `0.80` |

### Constraints

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iterations` | Maximum experiment iterations | `20` |
| `max_cost_usd` | Maximum API cost budget | `5.0` |

## Output Files

| File | Description |
|------|-------------|
| `experiment_config.json` | Current iteration's hyperparameters (agent-writable) |
| `experiment_history.jsonl` | Append-only log of all runs |
| `agent_notes.md` | Agent's reasoning chain per iteration |
| `best_config.json` | Best config found so far |

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing
```

## License

MIT
