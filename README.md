# AutoRAGResearch

Autonomous RAG pipeline optimization system. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) concept -- an LLM agent iteratively tunes RAG hyperparameters (chunking, embeddings, retrieval) and evaluates with RAGAS metrics until a target quality threshold is reached or the cost budget is exhausted.

## Overview and Motivation

Building a high-quality RAG (Retrieval-Augmented Generation) pipeline involves tuning many interdependent hyperparameters: chunk size, chunk overlap, number of retrieved documents, embedding model, and LLM model. Manually searching this space is tedious and error-prone.

AutoRAGResearch automates this process with a closed-loop optimization cycle. An LLM agent analyzes evaluation results, reasons about which parameters to change, and proposes the next configuration to try. Every iteration is logged to MLflow for experiment tracking, and the best configuration is saved via git checkpoints.

Key features:

- **Autonomous optimization** -- an LLM agent analyzes RAGAS scores and suggests the next hyperparameter configuration
- **Multi-source data ingestion** -- load documents from local files (PDF, TXT, CSV), Google Drive, AWS S3, Notion, web URLs, or HuggingFace datasets
- **RAGAS evaluation** -- faithfulness, answer relevancy, context precision, and context recall metrics with composite scoring
- **MLflow experiment tracking** -- every iteration is logged with parameters, metrics, and artifacts
- **Cost-aware** -- tracks API costs per iteration and stops when the budget is exceeded
- **Resumable** -- interrupted experiments can be continued from the last checkpoint
- **Git checkpoints** -- auto-commits the best configuration when scores improve

## Architecture

```
program.md (config)
      |
      v
+--------------+     +---------------+     +--------------+
| Data Sources |---->| RAG Pipeline  |---->|  Evaluator   |
|  (loaders)   |     | chunk / embed |     |   (RAGAS)    |
+--------------+     |   / retrieve  |     +------+-------+
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

**Loop flow:** Load data from all enabled sources --> run RAG pipeline with current config --> evaluate with RAGAS --> log to MLflow --> if score improved, git checkpoint --> LLM agent suggests next config --> repeat until threshold met, budget exceeded, or max iterations reached.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (required for LLM generation and RAGAS evaluation)

### Installation

```bash
git clone https://github.com/shivamshinde123/Auto-RAG-Research.git
cd Auto-RAG-Research
uv sync
```

### Set API Keys

Create a `.env` file in the project root or export environment variables directly:

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional -- only needed for specific data sources
export NOTION_API_KEY="ntn_..."           # Notion connector
export AWS_ACCESS_KEY_ID="..."            # S3 connector
export AWS_SECRET_ACCESS_KEY="..."        # S3 connector
```

### Configure

Edit `program.md` to define your experiment. This file uses a simple Markdown-based config format:

```markdown
## Data Sources

[[data_sources]]
type: huggingface
dataset_name: rajpurkar/squad
split: validation
sample_size: 50
enabled: true

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
# Validate config and credentials without running the experiment
uv run python main.py --dry-run

# Run the full experiment loop
uv run python main.py --config program.md

# Resume an interrupted experiment from where it left off
uv run python main.py --config program.md --resume

# Enable detailed debug logging
uv run python main.py --config program.md --verbose
```

## Data Source Setup

AutoRAGResearch supports eight data source connectors. You can enable multiple sources simultaneously -- documents from all enabled sources are merged and deduplicated before building the vector store.

At least one source providing QA pairs (currently only `huggingface`) must be enabled for RAGAS evaluation to work.

### Local PDF

Reads all `.pdf` files from a directory. Uses PyMuPDF as the primary parser with pdfplumber as a fallback.

```markdown
[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Directory containing `.pdf` files |

No credentials needed.

### Local TXT

Reads all `.txt` files from a directory with automatic encoding detection via chardet.

```markdown
[[data_sources]]
type: local_txt
path: data/txt/
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Directory containing `.txt` files |

No credentials needed.

### Local CSV

Reads all `.csv` files from a directory, extracting text from a specified column. Each row becomes a separate document.

```markdown
[[data_sources]]
type: local_csv
path: data/csv/
text_column: content
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Directory containing `.csv` files |
| `text_column` | Yes | Name of the column containing document text |

No credentials needed.

### Google Drive

Reads PDF and DOCX files from a Google Drive folder using the Drive API v3 with OAuth2.

```markdown
[[data_sources]]
type: gdrive
folder_id: YOUR_GDRIVE_FOLDER_ID
credentials_path: .secrets/gdrive_credentials.json
file_types: [pdf, docx, txt]
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `folder_id` | Yes | Google Drive folder ID (from the folder URL) |
| `credentials_path` | No | Path to OAuth2 credentials JSON (default: `.secrets/credentials.json`) |

**Setup steps:**

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a project
2. Enable the Google Drive API
3. Create OAuth2 credentials (Desktop application) and download `credentials.json`
4. Place the file at `.secrets/gdrive_credentials.json` (or your configured path)
5. On first run, a browser window will open for OAuth consent. The token is cached at `.secrets/token.json`

### AWS S3

Reads PDF and TXT files from an S3 bucket with an optional key prefix.

```markdown
[[data_sources]]
type: s3
bucket: your-bucket-name
prefix: documents/
region: us-east-1
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `bucket` | Yes | S3 bucket name |
| `prefix` | No | Key prefix to filter objects (default: empty string) |
| `region` | No | AWS region |

**Credentials:** Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables (or add them to your `.env` file). The IAM user/role needs `s3:GetObject` and `s3:ListBucket` permissions.

### Notion

Fetches pages from a Notion database, extracting text content from all block types recursively (paragraphs, headings, lists, code blocks, toggles, etc.).

```markdown
[[data_sources]]
type: notion
database_id: YOUR_NOTION_DATABASE_ID
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `database_id` | Yes | Notion database ID (from the database URL) |

**Setup steps:**

1. Go to [Notion Integrations](https://www.notion.so/my-integrations) and create a new integration
2. Copy the integration token
3. Set `export NOTION_API_KEY="ntn_..."` in your environment
4. In Notion, share the target database with your integration (click the three-dot menu on the database page and select "Connect to" your integration)

### Web URLs

Scrapes web pages using BeautifulSoup. Strips navigation, footer, and sidebar elements. Respects `robots.txt`.

```markdown
[[data_sources]]
type: web
urls:
  - https://example.com/docs
  - https://example.com/blog
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `urls` | Yes | List of URLs to scrape (use `- url` syntax for each) |

No credentials needed. Note: some sites may block automated scraping.

### HuggingFace Datasets

Loads datasets from HuggingFace Hub. Currently supports SQuAD and HotpotQA format datasets. This is the only connector that provides QA pairs for RAGAS evaluation, so at least one HuggingFace source should be enabled.

```markdown
[[data_sources]]
type: huggingface
dataset_name: rajpurkar/squad
split: validation
sample_size: 20
enabled: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `dataset_name` | Yes | HuggingFace dataset identifier (e.g., `rajpurkar/squad`, `hotpot_qa`) |
| `split` | Yes | Dataset split to use (e.g., `train`, `validation`, `test`) |
| `sample_size` | No | Limit number of examples loaded (useful for testing and cost control) |

No credentials needed for public datasets.

### Using Multiple Sources

You can combine any number of data sources. Documents from all enabled sources are merged and deduplicated by content hash:

```markdown
[[data_sources]]
type: local_pdf
path: ./data/papers
enabled: true

[[data_sources]]
type: web
urls:
  - https://docs.example.com/guide
enabled: true

[[data_sources]]
type: huggingface
dataset_name: rajpurkar/squad
split: validation
sample_size: 100
enabled: true
```

## Viewing Results in MLflow

Every experiment iteration is logged to MLflow with hyperparameters, RAGAS scores, cost metrics, and config artifacts.

```bash
# Start the MLflow UI
uv run mlflow ui

# Then open http://localhost:5000 in your browser
```

In the MLflow UI you can:

- Compare metrics across iterations (faithfulness, answer relevancy, context precision, context recall, composite score)
- View hyperparameter configurations for each run
- Track cost per iteration and cumulative cost
- Filter runs tagged with `is_best=True` to find the top-performing configurations
- Download config artifacts for any run

MLflow data is stored locally in the `mlruns/` directory by default.

## Configuration Reference

The `program.md` file uses a Markdown-based format with `##` section headers and `key: value` pairs. Lists use bracket syntax `[a, b, c]` or YAML-style `- item` for multiline lists.

### Data Sources

Each data source is defined as a `[[data_sources]]` block. See the [Data Source Setup](#data-source-setup) section for details on each connector.

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Connector type: `local_pdf`, `local_txt`, `local_csv`, `gdrive`, `s3`, `notion`, `web`, `huggingface` |
| `enabled` | Yes | Whether this source is active (`true` / `false`) |

Additional fields vary by connector type.

### Search Space

Defines the hyperparameter space the agent explores. Each parameter accepts a list of values.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `chunk_size` | Document chunk sizes in characters | `[256, 512, 1024]` |
| `chunk_overlap` | Overlap between consecutive chunks in characters | `[0, 50, 100]` |
| `top_k` | Number of chunks retrieved per query | `[3, 5, 8, 10]` |
| `embedding_model` | Embedding model for vector store | `[all-MiniLM-L6-v2, BGE-large, text-embedding-ada-002]` |
| `llm_model` | LLM for answer generation | `[gpt-4o-mini, gpt-3.5-turbo]` |

**Supported embedding models:**

| Name | Provider | Notes |
|------|----------|-------|
| `all-MiniLM-L6-v2` | HuggingFace (local) | Fast, no API cost |
| `BGE-large` | HuggingFace (local) | Higher quality, no API cost |
| `text-embedding-ada-002` | OpenAI API | Requires `OPENAI_API_KEY` |

**Supported LLM models:**

| Name | Provider | Notes |
|------|----------|-------|
| `gpt-4o-mini` | OpenAI API | Good quality/cost balance |
| `gpt-3.5-turbo` | OpenAI API | Lower cost |

### Optimization Target

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `primary_metric` | Main metric to optimize | `context_recall` | `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` |
| `secondary_metric` | Secondary metric | `faithfulness` | Same as above |
| `min_threshold` | Stop when composite score >= this value | `0.80` | `0.0` to `1.0` |

The composite score is the weighted average of all four RAGAS metrics (equal weights by default).

### Constraints

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iterations` | Maximum number of experiment iterations | `20` |
| `max_cost_usd` | Maximum cumulative API cost in USD | `5.0` |

The experiment stops when any of these conditions is met: composite score reaches `min_threshold`, cost exceeds `max_cost_usd`, or iteration count reaches `max_iterations`.

### Experiment

| Parameter | Description | Default |
|-----------|-------------|---------|
| `experiment_name` | MLflow experiment name | `autoragresearch_run_1` |
| `git_checkpoints` | Auto-commit best configs to git | `true` |

## Output Files

| File | Description |
|------|-------------|
| `experiment_config.json` | Current iteration's hyperparameters (written by the agent) |
| `experiment_history.jsonl` | Append-only log of all runs with configs, scores, and timestamps |
| `agent_notes.md` | Agent's analysis and reasoning chain for each iteration |
| `best_config.json` | Best configuration found across all iterations |
| `mlruns/` | MLflow tracking data (parameters, metrics, artifacts) |

## Project Structure

```
Auto-RAG-Research/
  main.py                  # CLI entry point
  program.md               # Experiment configuration
  pyproject.toml            # Dependencies and project metadata
  src/
    config_loader.py        # Parses program.md into Python dataclasses
    rag_pipeline.py         # Chunking, embedding, retrieval, generation
    evaluator.py            # RAGAS evaluation and composite scoring
    agent.py                # LLM agent for config suggestion
    cost_tracker.py         # API cost tracking with budget enforcement
    experiment_logger.py    # MLflow experiment logging
    dataset_loader.py       # Orchestrates all data source loading
    git_checkpoint.py       # Auto-commits on score improvement
    data_sources/
      __init__.py           # Connector registry and factory
      base.py               # Abstract base class for connectors
      local_pdf.py          # Local PDF files (PyMuPDF + pdfplumber)
      local_txt.py          # Local TXT files (chardet encoding detection)
      local_csv.py          # Local CSV files
      gdrive.py             # Google Drive API v3
      s3.py                 # AWS S3
      notion.py             # Notion API
      web.py                # Web scraping (BeautifulSoup)
      huggingface.py        # HuggingFace datasets (SQuAD, HotpotQA)
  tests/                    # Unit tests
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies: `uv sync --extra dev`
4. Make your changes and add tests
5. Run the test suite: `uv run pytest --cov=src --cov-report=term-missing`
6. Commit and push your branch
7. Open a pull request targeting `main`

## License

MIT
