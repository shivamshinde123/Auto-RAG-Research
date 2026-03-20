## IMPORTANT
# This is the ONLY file that should be edited to run experiments.
# Do NOT modify main.py, src/, or any other source files.
# Just update this config and run: uv run python main.py

## Data Sources
# Place your PDF files in the data/pdfs/ folder.
# QA pairs for evaluation are auto-generated from PDF content.

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

## QA Generation
# Number of question-answer pairs to generate from PDF content for evaluation.
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
max_iterations: 1
max_cost_usd: 5.0

## Experiment
experiment_name: autoragresearch_run_1
git_checkpoints: true
