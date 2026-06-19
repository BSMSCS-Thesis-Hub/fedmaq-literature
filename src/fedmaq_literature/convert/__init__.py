"""PDF conversion: Docling primary, Marker GPU fallback."""

from fedmaq_literature.convert.models import ConvertOutput, QAReport
from fedmaq_literature.convert.pipeline import convert_paper, write_conversion

__all__ = ["ConvertOutput", "QAReport", "convert_paper", "write_conversion"]
