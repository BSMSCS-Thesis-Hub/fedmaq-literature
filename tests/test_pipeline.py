"""Tests for conversion output writing."""

from pathlib import Path

from fedmaq_literature.convert.models import ConvertOutput, QAReport
from fedmaq_literature.convert.pipeline import write_conversion


def test_write_conversion(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    pdf = tmp_path / "papers" / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "markdown").mkdir()

    output = ConvertOutput(
        markdown="# Sample\n\nBody text.",
        converter="docling",
        pdf_path=pdf,
        page_count=3,
        qa=QAReport(
            passed=True,
            confidence=0.95,
            mean_grade="good",
            low_grade="fair",
            char_count=20,
            page_count=3,
        ),
        raw_confidence={"mean_grade": "good"},
    )

    out_dir = write_conversion("sample-slug", output, root=tmp_path)
    assert (out_dir / "paper.md").read_text(encoding="utf-8") == output.markdown
    meta = (out_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "slug: sample-slug" in meta
    assert "converter: docling" in meta


def test_post_process_markdown_math() -> None:
    from fedmaq_literature.convert.pipeline import _post_process_markdown

    # 1. Block math wrapped in aligned
    input_text = "Some text.\n$$\n\\frac{\\partial C}{\\partial z_i} & \\approx \\frac{1}{T} (q_i - p_i)\n$$\nMore text."
    expected = "Some text.\n$$\n\\begin{aligned}\n\\frac{\\partial C}{\\partial z_i} & \\approx \\frac{1}{T} (q_i - p_i)\n\\end{aligned}\n$$\nMore text."
    assert _post_process_markdown(input_text) == expected

    # 2. Block math wrapped in equation and has alignment -> strip equation, wrap in aligned
    input_text = "$$\n\\begin{equation}\n\\frac{\\partial C}{\\partial z_i} & \\approx \\frac{1}{T} (q_i - p_i)\n\\end{equation}\n$$"
    expected = "$$\n\\begin{aligned}\n\\frac{\\partial C}{\\partial z_i} & \\approx \\frac{1}{T} (q_i - p_i)\n\\end{aligned}\n$$"
    assert _post_process_markdown(input_text) == expected

    # 3. Inline math containing alignment -> convert to block math and wrap in aligned
    input_text = "We have $\\frac{\\partial C}{\\partial z_i} & \\approx \\frac{1}{T} (q_i - p_i)$."
    expected = "We have \n$$\n\\begin{aligned}\n\\frac{\\partial C}{\\partial z_i} & \\approx \\frac{1}{T} (q_i - p_i)\n\\end{aligned}\n$$\n."
    assert _post_process_markdown(input_text) == expected
