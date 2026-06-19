"""PDF conversion orchestration: Docling primary, Marker fallback."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from fedmaq_literature.convert.models import ConvertOutput
from fedmaq_literature.paths import markdown_dir, repo_root
from fedmaq_literature.registry import resolve_pdf_path, update_registry_conversion


def _qa_passed(output: ConvertOutput | None) -> bool:
    return bool(output and output.qa and output.qa.passed)


def convert_paper(
    slug: str,
    *,
    pdf_path: Path | None = None,
    force_marker: bool = False,
    skip_marker_fallback: bool = False,
    root: Path | None = None,
) -> ConvertOutput:
    """Convert a paper PDF to markdown/{slug}/paper.md with QA metadata."""
    root = root or repo_root()
    source = pdf_path or resolve_pdf_path(slug, root=root)
    docling_output: ConvertOutput | None = None
    marker_output: ConvertOutput | None = None

    if not force_marker:
        from fedmaq_literature.convert import docling as docling_backend

        try:
            docling_output = docling_backend.convert_pdf(source)
        except RuntimeError:
            docling_output = None

        if _qa_passed(docling_output):
            return _finalize(slug, docling_output, root=root)

        if skip_marker_fallback:
            if docling_output is None:
                update_registry_conversion(slug, "failed", root=root)
                raise RuntimeError(f"Docling conversion failed for slug '{slug}'")
            return _finalize(slug, docling_output, root=root, failed=True)

    if not force_marker and skip_marker_fallback:
        assert docling_output is not None
        return _finalize(slug, docling_output, root=root, failed=True)

    from fedmaq_literature.convert import marker as marker_backend

    try:
        marker_output = marker_backend.convert_pdf(source)
    except RuntimeError:
        marker_output = None

    if _qa_passed(marker_output):
        return _finalize(slug, marker_output, root=root)

    if _qa_passed(docling_output):
        return _finalize(slug, docling_output, root=root)

    chosen = _pick_best(docling_output, marker_output)
    if chosen is None:
        update_registry_conversion(slug, "failed", root=root)
        raise RuntimeError(f"All converters failed for slug '{slug}'")

    return _finalize(slug, chosen, root=root, failed=True)


def _pick_best(
    docling_output: ConvertOutput | None,
    marker_output: ConvertOutput | None,
) -> ConvertOutput | None:
    candidates = [
        output for output in (docling_output, marker_output) if output is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: item.qa.char_count if item.qa else len(item.markdown),
    )


def _finalize(
    slug: str,
    output: ConvertOutput,
    *,
    root: Path,
    failed: bool = False,
) -> ConvertOutput:
    write_conversion(slug, output, root=root)
    if failed or not _qa_passed(output):
        update_registry_conversion(slug, "failed", root=root)
    else:
        update_registry_conversion(slug, "ready", root=root)
    return output


def write_conversion(
    slug: str, output: ConvertOutput, *, root: Path | None = None
) -> Path:
    """Write paper.md and meta.yaml for a conversion result."""
    root = root or repo_root()
    out_dir = markdown_dir(root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_path = out_dir / "paper.md"
    paper_path.write_text(output.markdown, encoding="utf-8")

    qa_dict = asdict(output.qa) if output.qa else None
    meta = {
        "slug": slug,
        "pdf": output.pdf_path.relative_to(root).as_posix(),
        "converter": output.converter,
        "converted_at": datetime.now(UTC).isoformat(),
        "page_count": output.page_count,
        "char_count": output.qa.char_count if output.qa else len(output.markdown),
        "qa": qa_dict,
        "confidence": output.raw_confidence or None,
    }
    meta_path = out_dir / "meta.yaml"
    meta_path.write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_dir
