## Data Sources
# You can enable multiple sources at once. The system will load and merge
# documents from all enabled sources before building the vector store.

[[data_sources]]
type: local_pdf
path: data/pdfs/
enabled: true

[[data_sources]]
type: local_txt
path: data/txt/
enabled: false

[[data_sources]]
type: local_csv
path: data/csv/
text_column: content
enabled: false

[[data_sources]]
type: gdrive
folder_id: YOUR_GDRIVE_FOLDER_ID
credentials_path: .secrets/gdrive_credentials.json
file_types: [pdf, docx, txt]
enabled: false

[[data_sources]]
type: s3
bucket: your-bucket-name
prefix: documents/
region: us-east-1
enabled: false

[[data_sources]]
type: notion
database_id: YOUR_NOTION_DATABASE_ID
enabled: false

[[data_sources]]
type: web
urls:
  - https://example.com/docs
  - https://example.com/blog
enabled: false

[[data_sources]]
type: huggingface
dataset_name: hotpotqa
split: train
sample_size: 50
enabled: false

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
experiment_name: autoragresearch_run_1
git_checkpoints: true
