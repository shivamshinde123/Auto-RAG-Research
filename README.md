# AutoRAGResearch

Autonomous RAG pipeline optimization system. Inspired by concept of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an LLM agent iteratively tunes RAG hyperparameters (chunking, embeddings, retrieval) and evaluates with RAGAS metrics until a target quality threshold is reached.

## Detailed Documentation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shivamshinde123/Auto-RAG-Research)

## How It Works

1. Drop a PDF in `data/pdfs/`
2. Edit `program.md` to configure your experiment
3. Run `uv run python main.py`

The system automatically:
- Loads and parses your PDF
- Generates QA pairs from the PDF content using an LLM
- Runs the RAG pipeline with the current config
- Evaluates with RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
- Logs results to MLflow
- Uses an LLM agent to suggest the next hyperparameter config
- Repeats until the target score is reached or budget is exhausted

## Important: Single-File Configuration

**`program.md` is the ONLY file you need to edit to run experiments.** Do not modify `main.py`, `src/`, or any other source files. Just update the config and run:

```bash
uv run python main.py
```

## Architecture

```
program.md (config)
      |
      v
+--------------+     +---------------+     +-------------+
|   PDF Loader |---->| RAG Pipeline  |---->|  Evaluator   |
|  + QA Gen    |     | chunk/embed/  |     |   (RAGAS)    |
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

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Installation

```bash
git clone https://github.com/shivamshinde123/Auto-RAG-Research.git
cd Auto-RAG-Research
uv sync
```

### Set API Key

> **Warning:** Never commit your `.env` file. It is already in `.gitignore`, but double-check before pushing.

```bash
# Create a .env file (or export directly)
cp .env.example .env
# Then edit .env and add your API key
```

### Configure

Edit `program.md` to define your experiment:

```markdown
## Data Sources

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## QA Generation
num_qa_pairs: 20

## Search Space
chunk_size: [256, 512, 1024]
chunk_overlap: [0, 50, 100]
top_k: [3, 5, 8, 10]
embedding_model: [text-embedding-ada-002, BGE-large, all-MiniLM-L6-v2]
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
uv run python main.py

# Resume interrupted experiment
uv run python main.py --resume
```

### View Results

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## Configuration Reference

### Data Source

Place your PDF files in `data/pdfs/`. The system reads all `.pdf` files from this directory.

### QA Generation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_qa_pairs` | Number of QA pairs to generate from PDF content | `20` |

### Search Space

| Parameter | Description | Example |
|-----------|-------------|---------|
| `chunk_size` | Document chunk sizes to try | `[256, 512, 1024]` |
| `chunk_overlap` | Overlap between chunks | `[0, 50, 100]` |
| `top_k` | Number of retrieved chunks | `[3, 5, 10]` |
| `embedding_model` | Embedding models (see below) | `[all-MiniLM-L6-v2, BGE-large]` |
| `llm_model` | LLM for generation (see below) | `[gpt-4o-mini, gpt-3.5-turbo]` |

### Supported Models

**LLMs** — Any OpenAI chat model can be used. The model name is passed directly to `ChatOpenAI`, so any model available on your OpenAI account works:

| Model | Notes |
|-------|-------|
| `gpt-4o-mini` | Fast, cheap, good default |
| `gpt-3.5-turbo` | Cheapest option |
| `gpt-4o` | Most capable, higher cost |
| `gpt-4` | Previous generation |
| `gpt-4-turbo` | Faster GPT-4 variant |

**Embedding Models** — Only these 3 are supported (hardcoded in the pipeline):

| Model | Type | Notes |
|-------|------|-------|
| `text-embedding-ada-002` | OpenAI API | Best quality, requires API key |
| `BGE-large` | Local (HuggingFace) | Good quality, runs locally, slower first load |
| `all-MiniLM-L6-v2` | Local (HuggingFace) | Fastest local option, smaller model |

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
