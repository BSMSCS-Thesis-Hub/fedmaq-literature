"""Shared types for PDF conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class QAReport:
    passed: bool
    confidence: float | None
    mean_grade: str | None
    low_grade: str | None
    char_count: int
    page_count: int | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConvertOutput:
    markdown: str
    converter: str
    pdf_path: Path
    page_count: int | None = None
    qa: QAReport | None = None
    raw_confidence: dict[str, str | float | None] = field(default_factory=dict)
