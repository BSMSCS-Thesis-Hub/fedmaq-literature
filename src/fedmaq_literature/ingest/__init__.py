"""LlamaIndex ingestion with Qwen/Qwen3-Embedding-4B (scaffold)."""

import os

DEFAULT_EMBED_MODEL = os.environ.get(
    "FEDMAQ_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"
)
FALLBACK_EMBED_MODEL = os.environ.get(
    "FEDMAQ_EMBED_FALLBACK_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)
EMBED_BATCH_SIZE = int(os.environ.get("FEDMAQ_EMBED_BATCH_SIZE", "4"))

RETRIEVAL_INSTRUCT = (
    "Given a research question about federated learning, quantization, "
    "or knowledge distillation, retrieve relevant passages from academic papers."
)
