"""Tests for LlamaIndex + ChromaDB ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml

from fedmaq_literature.ingest.pipeline import run_ingest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding


def test_run_ingest(tmp_path: Path) -> None:
    # 1. Setup mock directories
    (tmp_path / "papers").mkdir()
    (tmp_path / "markdown").mkdir()
    cursor_dir = tmp_path / ".cursor" / "project"
    cursor_dir.mkdir(parents=True)

    # 2. Setup mock registry
    registry_file = cursor_dir / "paper_registry.md"
    registry_content = (
        "# Paper Registry\n\n"
        "| Slug | PDF | Baseline | Conversion | Summary | Tags |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| test-paper | Test Paper | FedAvg | ready | none | test |\n"
    )
    registry_file.write_text(registry_content, encoding="utf-8")

    # 3. Setup mock converted markdown
    paper_dir = tmp_path / "markdown" / "test-paper"
    paper_dir.mkdir()
    (paper_dir / "paper.md").write_text(
        "# Test Title\n\nThis is a test paper.", encoding="utf-8"
    )

    meta_content = {"page_count": 5, "char_count": 22, "confidence": 0.99}
    (paper_dir / "meta.yaml").write_text(yaml.safe_dump(meta_content), encoding="utf-8")

    # 4. Mock the Chroma db persistent client and collection
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    # Patch chromadb and HuggingFaceEmbedding
    with (
        patch("chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("fedmaq_literature.ingest.pipeline.HuggingFaceEmbedding") as mock_hfe,
    ):

        # Set HuggingFaceEmbedding to return MockEmbedding
        mock_hfe.return_value = MockEmbedding(embed_dim=384)

        # Run the ingest pipeline for a specific slug
        ret = run_ingest(slug="test-paper", root=tmp_path, device="cpu")
        assert ret == 0

        # Verify collection delete was called to clear duplicate entries
        mock_collection.delete.assert_called_once_with(where={"slug": "test-paper"})

        # Verify collection reset and run with slug=None
        mock_collection.reset_mock()
        ret = run_ingest(slug=None, root=tmp_path, device="cpu")
        assert ret == 0
        mock_collection.delete.assert_called_once_with(where={"slug": "test-paper"})
