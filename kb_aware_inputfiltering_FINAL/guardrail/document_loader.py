from pathlib import Path
from typing import Union


def load_document(path_or_file: Union[str, Path, object]) -> str:
    """Load TXT, PDF, or DOCX content. Supports Streamlit uploaded files."""
    name = getattr(path_or_file, "name", None) or str(path_or_file)
    suffix = Path(name).suffix.lower()

    if hasattr(path_or_file, "read"):
        data = path_or_file.read()
        if isinstance(data, str):
            raw_bytes = data.encode("utf-8", errors="ignore")
        else:
            raw_bytes = data
    else:
        raw_bytes = Path(path_or_file).read_bytes()

    if suffix == ".txt" or suffix == "":
        return raw_bytes.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF: {exc}") from exc

    if suffix == ".docx":
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(raw_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            raise RuntimeError(f"Could not read DOCX: {exc}") from exc

    return raw_bytes.decode("utf-8", errors="ignore")
