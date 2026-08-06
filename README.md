# EcoSpec-KG

EcoSpec-KG turns the Chinese `HJ 1166-HJ 1176` ecological assessment
standards into a traceable, machine-readable knowledge graph. The project
supports a lightweight native graph-RAG pipeline and an adapter for Microsoft
GraphRAG 3.1.0.

The project does **not** claim to replace GIS, remote sensing, field surveys,
or ecosystem process models. It organizes standards knowledge and evaluates
retrieval, relation extraction, completion, and provenance tracing.

## Data policy

Raw standards are not redistributed. Point the project to a local folder:

```powershell
$env:ECOSPEC_CORPUS_DIR='D:\else\lky\生态论文重写版本\全国生态状况调查评估技术规范'
```

Generated annotation records may be released separately. Each record retains
the standard code, page, section, evidence text, and source file hash.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[ui,dev]"

eco-spec-kg corpus --source "$env:ECOSPEC_CORPUS_DIR" --out data/processed
eco-spec-kg annotate --chunks data/processed/chunks.jsonl --out data/annotations/annotation_template.csv
eco-spec-kg predict --chunks data/processed/chunks.jsonl --out results/rule_predictions.jsonl
eco-spec-kg index --adapter native --chunks data/processed/chunks.jsonl --out data/index/native
eco-spec-kg serve --data data --host 127.0.0.1 --port 7860
```

All unrun experiment metrics remain JSON `null`. The reporting command renders
them as `待实验补充`; it never invents values.

## Remote GPU

The primary completion model is `Qwen/Qwen3-0.6B`, with thinking disabled for
structured extraction. The embedding model is
`Qwen/Qwen3-Embedding-0.6B`. A remote vLLM endpoint can be configured with:

```powershell
$env:ECOSPEC_LLM_BASE_URL='http://gpu-server:8000/v1'
$env:ECOSPEC_LLM_API_KEY='local-token'
```

LoRA defaults are rank 8, alpha 16, dropout 0.05, learning rate `2e-4`,
five epochs maximum, and seeds 42, 43, and 44.

## Public commands

- `corpus`: deduplicate official PDFs and extract traceable chunks.
- `annotate`: create a two-reviewer annotation sheet.
- `index`: build a native or Microsoft GraphRAG index.
- `train`: prepare or run LoRA training.
- `predict`: run the deterministic rule baseline.
- `evaluate`: calculate extraction and provenance metrics.
- `ablate`: create the fixed ablation experiment matrix.
- `report`: render machine-readable results into Markdown.
- `serve`: launch the Gradio knowledge service.

## Repository status

The included fixture is for software tests only. It is not an experimental
result and must not be counted in the paper dataset.

## V2 experiment chain

The leakage-controlled V2 workflow separates blind extraction, prediction
validation, and strict evaluation. See
[`docs/V2正式实验命令链.md`](docs/V2正式实验命令链.md) for the frozen dataset
layout, commands, output manifests, and publication boundary.

For the local Qwen3.5-9B service, V2-aligned LoRA data, and server validation
steps, see
[`docs/Qwen3.5本地部署与V2验证手册.md`](docs/Qwen3.5本地部署与V2验证手册.md).
