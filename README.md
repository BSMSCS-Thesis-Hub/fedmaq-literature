# fedmaq-literature

PDF library, markdown conversions, RAG index, and per-paper summaries for the FedMAQ thesis.

## Structure

```text
papers/               # PDFs (pipeline reads; agents do not parse in chat)
markdown/{slug}/      # converted text + meta.yaml
summaries/drafts/     # LLM drafts (human approve)
summaries/            # approved summaries
src/fedmaq_literature/  # convert, ingest, workflows, CLI
storage/              # Chroma + cache (gitignored)
.cursor/              # rules, skills, paper_registry
```

## Stack

- PDF: Docling primary, Marker GPU fallback
- Embeddings: `Qwen/Qwen3-Embedding-4B` (local GPU, serialized jobs)
- Index: LlamaIndex + ChromaDB
- LLM: OpenRouter (DeepSeek V4)

## Setup

```bash
uv sync --extra dev --extra convert   # Docling primary converter
# Optional GPU fallback:
# uv sync --extra marker

uv run fedmaq-lit list-slugs
uv run fedmaq-lit convert --slug he-2025-dynfed
uv run fedmaq-lit ingest --slug he-2025-dynfed --convert-only
```

Conversion writes `markdown/{slug}/paper.md` and `meta.yaml`, updates `paper_registry.md`.
Docling runs first; Marker is used when Docling QA fails (serialize GPU jobs with embedding).

On Windows, if HuggingFace model download fails on symlinks, the converter sets
`HF_HUB_DISABLE_SYMLINKS=1` automatically; enable Developer Mode for faster caching.

## Agent onboarding

1. Read [../fedmaq-experiments/HANDOFF.md](../fedmaq-experiments/HANDOFF.md).
2. Read [AGENTS.md](AGENTS.md). Domain rules: `../fedmaq-experiments/.cursor/rules/`.
