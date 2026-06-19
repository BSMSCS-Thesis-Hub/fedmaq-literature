"""Marker PDF-to-markdown adapter (GPU fallback)."""

from __future__ import annotations

import os
from pathlib import Path

from fedmaq_literature.convert.models import ConvertOutput, QAReport
from fedmaq_literature.convert.qa import assess_markdown


def _marker_device() -> str:
    forced = os.environ.get("FEDMAQ_MARKER_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def convert_pdf(pdf_path: Path) -> ConvertOutput:
    """Convert a PDF with Marker and run content-only QA."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise RuntimeError(
            "Marker is not installed. Run: uv sync --extra marker"
        ) from exc

    device = _marker_device()
    artifact_dict = create_model_dict(device=device)
    converter = PdfConverter(
        artifact_dict,
        renderer="marker.renderers.markdown.MarkdownRenderer",
    )
    output = converter(str(pdf_path))
    markdown = output.markdown or ""

    qa: QAReport = assess_markdown(
        markdown,
        mean_grade=None,
        low_grade=None,
        confidence=None,
        page_count=None,
    )

    return ConvertOutput(
        markdown=markdown,
        converter="marker",
        pdf_path=pdf_path,
        page_count=None,
        qa=qa,
        raw_confidence={"device": device},
    )
