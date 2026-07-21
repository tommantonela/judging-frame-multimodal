# Who Judges the Frame?

**Companion repository for the paper _“Who Judges the Frame? Auditing Multimodal LLM Judges for News Framing Across Event-Level Perspectives.”_**

## Overview

This repository contains the code used to study news framing across matched, event-level perspectives. The workflow combines:

1. **Multimodal article embeddings** to compare how left-, center-, and right-oriented coverage of the same event is represented in text, images, and combined text-image inputs.
2. **LLM-as-a-judge experiments** to assess framing constructs under different input and metadata conditions.
3. **A reproducible analysis notebook** for similarity analysis, score comparisons, agreement statistics, metadata-sensitivity tests, and publication-ready charts.

The repository is designed around matched news articles covering the same event. It supports primary left, center, and right articles as well as optional fallback candidates stored in `more_left`, `more_center`, and `more_right`.

## Research workflow

### Experiment 1 — Event-level viewpoint divergence

The first experiment examines how similarly or differently left-, center-, and right-oriented articles represent the same news event.

The analysis:

- Generates embeddings for headline-and-text, image-only, and combined headline-text-image inputs.
- Visualizes embedding spaces with UMAP and PCA.
- Calculates within-modality cosine similarity for:
  - Left ↔ Right
  - Left ↔ Center
  - Right ↔ Center
- Examines event-level asymmetry and topic-level differences.
- Calculates cross-modality similarity for the same articles.

A higher cosine similarity indicates that two article representations are closer in the embedding space.

### Experiment 2 — Multimodal LLM framing assessment

The second experiment audits how LLM judges assess individual news articles across input modalities and metadata conditions.

The configured framing constructs are:

- Emotional intensity
- Conflict framing
- Polarization framing
- Sensationalism

The main individual-article experiment uses a 1–5 Likert scale. The analysis compares:

- Political-leaning groups
- Input modalities
- Topic categories
- Rank-order consistency using Spearman correlation
- Absolute agreement using intraclass correlation coefficients (ICC)
- Sensitivity to source and political-leaning metadata
- Paired score changes using the Wilcoxon signed-rank test

The judge script also supports pairwise, triadic, and ideology-inference experiments.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── run-embeddings-user-friendly.py
├── run-judges-frame-commented-images-required.py
├── rqs-analysis-commented.ipynb
└── data/
    ├── allsides_Jan-May_2025.jsonl
    ├── allsides_Jun2025_May2026.jsonl
    ├── full_articles/
    │   ├── apnews.com.json
    │   ├── reuters.com.json
    │   └── ...
    ├── embeddings_*.pickle
    ├── results_*.pickle
└── outputs/
    └── charts/
```

The commands below assume that the scripts retain the filenames shown above. Adjust the commands if you rename them before release.

## Installation

### 1. Clone the repository

```bash
git clone TODO_REPOSITORY_URL
cd TODO_REPOSITORY_NAME
```

### 2. Create a Python environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch GPU builds depend on the local CUDA version. For GPU use, install the appropriate `torch` and `torchvision` wheels for the system before installing the remaining requirements.

The Nemotron embedding model can optionally use `flash-attn` on a compatible NVIDIA/CUDA system:

```bash
python -m pip install "flash-attn>=2.6.3,<2.8" --no-build-isolation
```

### 4. Configure the local judge model

The judge experiments use Ollama through `langchain-ollama`. Install and start Ollama separately, then make the desired model available locally.

The current default model configured in the script is:

```text
gemma4:e2b
```

A different local model can be supplied with `--llm`.

## Data layout

The scripts expect either:

```text
data/
├── allsides_Jan-May_2025.jsonl
├── allsides_Jun2025_May2026.jsonl
└── full_articles/
```

or, for the judge script:

```text
data/
├── allsides_Jan2025_May2026_combined.jsonl
└── full_articles/
```

Each event-level dataset row is expected to contain a shared event headline and article candidates for the `left`, `center`, and `right` groups. Optional fallback candidates may be provided in `more_left`, `more_center`, and `more_right`.

The flattened full-article records are expected to provide fields used by the scripts, including:

- Article URL
- Extracted headline
- Extracted body text
- Source/domain
- Political-leaning label
- Local image metadata and paths

### Comparable-sample rule

Every selected article must have at least one readable, non-GIF local image, **including articles used in text-only conditions**.

Images are passed to the model only when the selected input condition includes `image`. The universal image-eligibility rule keeps the article sample comparable across text, image, and multimodal conditions.

## Quick start

### Inspect supported options

```bash
python run-embeddings-user-friendly.py --list-options
python run-judges-frame-commented-images-required.py --list-options
```

Full command-line help is available with:

```bash
python run-embeddings-user-friendly.py --help
python run-judges-frame-commented-images-required.py --help
```

### Validate the local data

Validate dataset files, full-article matching, and image coverage without loading the embedding model:

```bash
python run-embeddings-user-friendly.py \
    --data-dir ./data \
    --validate-only
```

## Step 1: Generate multimodal embeddings

The default embedding model is:

```text
nvidia/llama-nemotron-embed-vl-1b-v2
```

To create the three embedding conditions used in the main event-level analysis:

```bash
python run-embeddings-user-friendly.py \
    --data-dir ./data \
    --output-dir ./outputs \
    --input-condition headline-text image headline-text-image
```

Useful options include:

```text
--device cuda
--batch-size 1
--limit 100
--checkpoint-every 10
--overwrite
```

For example, a short GPU test run is:

```bash
python run-embeddings-user-friendly.py \
    --data-dir ./data \
    --output-dir ./outputs \
    --input-condition headline-text-image \
    --device cuda \
    --limit 20
```

Embedding runs are resumable. Existing checkpoints are loaded unless `--overwrite` is supplied.

The output uses the nested structure:

```python
results[event_headline][leaning_group][article_url] = embedding
```

Default-dataset output files follow this pattern:

```text
embeddings_<input-condition>__.pickle
```

## Step 2: Run LLM judge experiments

The local Ollama model must be running and available before starting these experiments.

### Individual framing assessment

Run one framing dimension and one input condition:

```bash
python run-judges-frame-commented-images-required.py \
    --dir-path ./data \
    --experiment individual \
    --dimension emotional-intensity \
    --input-condition headline-text \
    --llm YOUR_OLLAMA_MODEL
```

Run every configured framing dimension and every compatible non-metadata input condition:

```bash
python run-judges-frame-commented-images-required.py \
    --dir-path ./data \
    --experiment individual \
    --llm YOUR_OLLAMA_MODEL
```

### Metadata-sensitivity experiment

```bash
python run-judges-frame-commented-images-required.py \
    --dir-path ./data \
    --experiment individual-metadata \
    --llm YOUR_OLLAMA_MODEL
```

### Pairwise comparison

```bash
python run-judges-frame-commented-images-required.py \
    --dir-path ./data \
    --experiment pairs \
    --input-condition image \
    --llm YOUR_OLLAMA_MODEL \
    --include-more
```

### Triadic comparison

```bash
python run-judges-frame-commented-images-required.py \
    --dir-path ./data \
    --experiment triadic \
    --input-condition headline-text-image \
    --llm YOUR_OLLAMA_MODEL \
    --include-more
```

### Ideology-inference triads

```bash
python run-judges-frame-commented-images-required.py \
    --dir-path ./data \
    --experiment ideology_triadic \
    --input-condition headline-text-image \
    --llm YOUR_OLLAMA_MODEL \
    --include-more
```

The `--include-more` flag allows the script to use candidates from `more_left`, `more_center`, or `more_right` when a primary article is unavailable or ineligible.

Judge results are checkpointed after each event and resume from existing files. A backup is created before an existing result file is loaded.

Result files follow these patterns:

```text
results_<experiment>_<input-condition>_<dimension>_<model>.pickle
results_<experiment>_<input-condition>_<model>.pickle
```

The second pattern is used for experiments that do not require a framing dimension.

## Step 3: Run the analysis notebook

Start JupyterLab:

```bash
jupyter lab
```

Open:

```text
rqs-analysis-commented.ipynb
```

In the notebook's configuration cell, set `DATA_DIR` to the directory containing the datasets, embeddings, and judge-result files. For the repository layout above:

```python
DATA_DIR = Path("data")
```

Then run the notebook from top to bottom.

Charts are saved automatically under:

```text
outputs/charts/
```

The notebook ends with a chart manifest listing every figure generated during the current run.

To generate publication-ready vector graphics in addition to PNG files, update:

```python
CHART_FORMATS = ("png", "pdf")
```

## Supported input conditions

### Embedding script

- `headline`
- `headline-text`
- `image`
- `summary`
- `headline-summary`
- `headline-image`
- `headline-text-image`
- `headline-summary-image`

### Judge script

Without metadata:

- `headline`
- `headline-text`
- `image`
- `headline-text-image`

With source and/or leaning metadata:

- `headline-text-source`
- `image-source`
- `headline-text-source-label`
- `image-source-label`
- `headline-text-image-source`
- `headline-text-image-source-label`

## Reproducibility notes

- Judge-model temperature is set to `0`.
- The judge script uses a fixed random seed for article-order randomization.
- UMAP and PCA projections use fixed random states in the analysis notebook.
- Embedding and judge runs use resumable checkpoints.
- Judge-result files are backed up before resuming.
- Figures use shared plotting helpers and are saved at 300 DPI by default.
- Model outputs can still vary across model versions, quantizations, hardware backends, and dependency versions.
- For an archival release, record the exact Ollama model digest, embedding-model revision, Python version, package versions, operating system, and accelerator configuration.

To capture the installed Python environment:

```bash
python -m pip freeze > environment-lock.txt
```