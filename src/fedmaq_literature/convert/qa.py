"""Quality checks for converted markdown."""

from __future__ import annotations

import os
import re

from fedmaq_literature.convert.models import QAReport

GRADE_ORDER = ("poor", "fair", "good", "excellent")
HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)

DEFAULT_MIN_MEAN_GRADE = os.environ.get("FEDMAQ_QA_MIN_MEAN_GRADE", "good").lower()
DEFAULT_MIN_LOW_GRADE = os.environ.get("FEDMAQ_QA_MIN_LOW_GRADE", "fair").lower()
MIN_CHAR_COUNT = int(os.environ.get("FEDMAQ_QA_MIN_CHARS", "1500"))


def _grade_index(grade: str | None) -> int:
    if grade is None:
        return -1
    normalized = grade.strip().lower()
    try:
        return GRADE_ORDER.index(normalized)
    except ValueError:
        return -1


def grade_at_least(grade: str | None, minimum: str) -> bool:
    return _grade_index(grade) >= _grade_index(minimum)


def assess_markdown(
    markdown: str,
    *,
    mean_grade: str | None = None,
    low_grade: str | None = None,
    confidence: float | None = None,
    page_count: int | None = None,
    min_mean_grade: str = DEFAULT_MIN_MEAN_GRADE,
    min_low_grade: str = DEFAULT_MIN_LOW_GRADE,
) -> QAReport:
    """Return QA verdict combining Docling grades and content heuristics."""
    reasons: list[str] = []
    stripped = markdown.strip()
    char_count = len(stripped)

    if char_count < MIN_CHAR_COUNT:
        reasons.append(f"char_count {char_count} < {MIN_CHAR_COUNT}")

    alpha = sum(ch.isalnum() for ch in stripped)
    if char_count and alpha / char_count < 0.45:
        reasons.append("low alphanumeric ratio")

    if not HEADING_RE.search(stripped):
        reasons.append("no markdown headings detected")

    if mean_grade is not None and not grade_at_least(mean_grade, min_mean_grade):
        reasons.append(f"mean_grade {mean_grade} below {min_mean_grade}")

    if low_grade is not None and not grade_at_least(low_grade, min_low_grade):
        reasons.append(f"low_grade {low_grade} below {min_low_grade}")

    passed = len(reasons) == 0
    return QAReport(
        passed=passed,
        confidence=confidence,
        mean_grade=mean_grade,
        low_grade=low_grade,
        char_count=char_count,
        page_count=page_count,
        reasons=tuple(reasons),
    )
