from pathlib import Path
from pypdf import PdfReader
import io

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

def candidate_name_from_filename(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()
