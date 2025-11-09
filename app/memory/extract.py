# memory/extract.py
from __future__ import annotations

import csv
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# ---------- Optional deps (import lazily/optionally) ----------
try:
    import magic  # python-magic (libmagic)
except Exception:  # pragma: no cover
    magic = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    import docx  # python-docx
except Exception:  # pragma: no cover
    docx = None

try:
    import openpyxl
except Exception:  # pragma: no cover
    openpyxl = None

try:
    import xlrd
except Exception:  # pragma: no cover
    xlrd = None

try:
    from bs4 import BeautifulSoup  # for HTML
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    import chardet  # optional charset detection
except Exception:  # pragma: no cover
    chardet = None


# ---------- Config ----------
DEFAULT_TXT_ENCODING = "utf-8"
READ_CHUNK_SIZE = 1 << 14  # 16 KiB
SUBPROCESS_TIMEOUT = 45  # seconds
MAX_SOFT_BYTES_TXT = None  # set to int to cap reads of giant .txt files


# ---------- Utilities ----------
@dataclass(frozen=True)
class ExtractionError(Exception):
    path: str
    reason: str
    def __str__(self) -> str:  # pragma: no cover
        return f"ExtractionError({self.path}): {self.reason}"


_EXT_TO_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".html": "text/html",
    ".htm": "text/html",
    ".rtf": "application/rtf",
}

def _run(cmd: list[str], *, timeout: int = SUBPROCESS_TIMEOUT) -> str:
    """Run a command and return stdout as UTF-8 (lossy)."""
    logger.debug("Running command: %s", " ".join(cmd))
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
    return out.decode("utf-8", "ignore")

def _detect_encoding(path: str, default: str = DEFAULT_TXT_ENCODING) -> str:
    if not chardet:
        return default
    try:
        with open(path, "rb") as fh:
            sample = fh.read(8192)
        guess = chardet.detect(sample)
        enc = guess.get("encoding") or default
        return enc
    except Exception:
        return default

def detect_mime(path: str) -> str:
    """Best-effort MIME detection using libmagic, then extension, then mimetypes."""
    # libmagic is best (detects real content)
    if magic:
        try:
            with open(path, "rb") as fh:
                return magic.from_buffer(fh.read(4096), mime=True)
        except Exception:
            pass
    # extension mapping (curated)
    ext = Path(path).suffix.lower()
    if ext in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext]
    # fallback to Python's mimetypes
    guess, _ = mimetypes.guess_type(path)
    return guess or "application/octet-stream"


# ---------- Sanitization ----------
# Remove NUL and other C0 control chars except \t, \n; also drop DEL (\x7F)
_CTRL_RE = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]')

def _sanitize_text(s: Optional[str], *, keep_newlines: bool = True) -> str:
    """
    Make text safe for MySQL TEXT/JSON:
      - normalize CRLF → LF
      - strip NULs and disallowed control characters
      - optionally collapse newlines
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_RE.sub("", s)
    if not keep_newlines:
        s = s.replace("\n", " ")
    return s


# ---------- Readers ----------
def _read_txt(path: str, encoding: Optional[str] = None) -> str:
    enc = encoding or _detect_encoding(path)
    size = os.path.getsize(path)
    if MAX_SOFT_BYTES_TXT and size > MAX_SOFT_BYTES_TXT:
        logger.warning("Truncating text read for %s to %d bytes", path, MAX_SOFT_BYTES_TXT)
        with open(path, "rb") as fh:
            data = fh.read(MAX_SOFT_BYTES_TXT)
        return data.decode(enc, "ignore")
    with open(path, "r", encoding=enc, errors="ignore") as fh:
        return fh.read()

def _read_pdf(path: str) -> str:
    if not fitz:
        raise ExtractionError(path, "PDF support requires PyMuPDF (fitz)")
    doc = fitz.open(path)
    try:
        lines: list[str] = []
        for page in doc:
            lines.append(page.get_text("text"))
        return "\n".join(lines)
    finally:
        doc.close()

def _read_docx(path: str) -> str:
    if not docx:
        raise ExtractionError(path, "DOCX support requires python-docx")
    d = docx.Document(path)
    parts: list[str] = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    # tables (optional but helpful)
    for t in d.tables:
        for row in t.rows:
            parts.append("\t".join(c.text or "" for c in row.cells))
    return "\n".join(parts)

def _read_doc(path: str) -> str:
    """
    Read legacy .doc using progressive fallbacks:
      1) antiword
      2) catdoc
      3) LibreOffice headless conversion to TXT
    """
    # 1) antiword
    if shutil.which("antiword"):
        try:
            return _run(["antiword", path])
        except Exception as e:
            logger.info("antiword failed on %s: %s", path, e)

    # 2) catdoc
    if shutil.which("catdoc"):
        try:
            return _run(["catdoc", "-w", path])
        except Exception as e:
            logger.info("catdoc failed on %s: %s", path, e)

    # 3) LibreOffice headless (best all-around fallback)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        tmp_dir = Path(path).parent
        out_txt = tmp_dir / (Path(path).stem + ".txt")
        # Convert directly to text. Filter name varies by build; "Text" usually works.
        try:
            _run([soffice, "--headless", "--convert-to", "txt:Text", "--outdir", str(tmp_dir), path])
            if out_txt.exists():
                try:
                    return _read_txt(str(out_txt))
                finally:
                    # cleanup best-effort
                    try:
                        out_txt.unlink(missing_ok=True)  # py3.8+: ignore via try/except
                    except Exception:
                        pass
        except Exception as e:
            logger.info("LibreOffice conversion failed on %s: %s", path, e)

    raise ExtractionError(path, "Cannot read .doc; install antiword/catdoc or LibreOffice")

def _read_xlsx(path: str) -> str:
    if not openpyxl:
        raise ExtractionError(path, "XLSX support requires openpyxl")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    lines: list[str] = []
    try:
        for ws in wb.worksheets:
            lines.append(f"# Sheet: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                lines.append("\t".join(cells))
        return "\n".join(lines)
    finally:
        wb.close()

def _read_xls(path: str) -> str:
    if not xlrd:
        raise ExtractionError(path, "XLS support requires xlrd<2.0 (legacy .xls)")
    book = xlrd.open_workbook(path)
    lines: list[str] = []
    for si in range(book.nsheets):
        sh = book.sheet_by_index(si)
        lines.append(f"# Sheet: {sh.name}")
        for r in range(sh.nrows):
            cells = [str(sh.cell_value(r, c)) for c in range(sh.ncols)]
            lines.append("\t".join(cells))
    return "\n".join(lines)

def _read_csv_tsv_stream(path: str, delimiter: Optional[str] = None) -> str:
    # Stream to avoid loading giant files into memory for sniffing
    with open(path, "r", encoding=_detect_encoding(path), errors="ignore", newline="") as fh:
        head = fh.read(4096)
        fh.seek(0)
        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(head)
                delimiter = dialect.delimiter
            except Exception:
                delimiter = "\t" if path.lower().endswith(".tsv") else ","
        reader = csv.reader(fh, delimiter=delimiter)
        # Build incrementally into a StringIO (fast enough and avoids quadratic string concat)
        out = io.StringIO()
        write = out.write
        join = delimiter.join
        for row in reader:
            write(join(row))
            write("\n")
        return out.getvalue()

def _read_json(path: str) -> str:
    with open(path, "r", encoding=_detect_encoding(path), errors="ignore") as fh:
        data = json.load(fh)
    # Pretty printed, ASCII preserved
    return json.dumps(data, ensure_ascii=False, indent=2)

def _read_jsonl(path: str) -> str:
    out_lines: list[str] = []
    with open(path, "r", encoding=_detect_encoding(path), errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                out_lines.append(json.dumps(json.loads(line), ensure_ascii=False))
            except Exception:
                out_lines.append(line)
    return "\n".join(out_lines)

def _read_html(path: str) -> str:
    if not BeautifulSoup:
        raise ExtractionError(path, "HTML support requires beautifulsoup4")
    with open(path, "r", encoding=_detect_encoding(path), errors="ignore") as fh:
        soup = BeautifulSoup(fh, "html.parser")
    # Remove scripts/styles and get visible text
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    # normalize blank lines
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def _read_rtf(path: str) -> str:
    """Best-effort RTF using unrtf if available, otherwise fallback to plain read."""
    if shutil.which("unrtf"):
        try:
            return _run(["unrtf", "--text", path])
        except Exception as e:
            logger.info("unrtf failed on %s: %s", path, e)
    # last resort: naive read (won't strip RTF control words)
    return _read_txt(path)


# ---------- Public API ----------
def extract_text(path: str, mime: Optional[str] = None) -> str:
    """
    Extract text from many common document types.

    Supported (via optional deps / CLIs):
      - .pdf (PyMuPDF)
      - .docx (python-docx; includes tables)
      - .doc (antiword/catdoc/LibreOffice headless)
      - .xlsx (openpyxl, read-only)
      - .xls (xlrd<2.0)
      - .csv/.tsv (streaming, dialect sniffing)
      - .json/.jsonl
      - .html (beautifulsoup4)
      - .rtf (unrtf if present; otherwise naive)
      - everything else treated as text

    Fallbacks:
      - If a specialized reader fails, we log and fall back to `_read_txt`.

    Notes:
      - Subprocess timeouts are enforced.
      - MIME detection prefers libmagic, then extension, then mimetypes.
      - For best .doc support, install one of: antiword, catdoc, or LibreOffice.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mime = mime or detect_mime(path)
    ext = Path(path).suffix.lower()
    logger.debug("Extracting from %s (mime=%s, ext=%s)", path, mime, ext)

    def _sanitize_return(s: str) -> str:
        return _sanitize_text(s, keep_newlines=True)

    try:
        # Strong type checks first by MIME, then extension as a guardrail.
        if mime == "application/pdf" or ext == ".pdf":
            return _sanitize_return(_read_pdf(path))

        if mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ) or ext == ".docx":
            return _sanitize_return(_read_docx(path))

        if mime in ("application/msword", "application/x-msword") or ext == ".doc":
            return _sanitize_return(_read_doc(path))

        if mime in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ) or ext == ".xlsx":
            return _sanitize_return(_read_xlsx(path))

        if mime in ("application/vnd.ms-excel", "application/xls") or ext == ".xls":
            return _sanitize_return(_read_xls(path))

        if mime in ("text/csv",) or ext == ".csv":
            return _sanitize_return(_read_csv_tsv_stream(path, ","))

        if mime in ("text/tab-separated-values",) or ext == ".tsv":
            return _sanitize_return(_read_csv_tsv_stream(path, "\t"))

        if mime in ("application/json",) or ext == ".json":
            return _sanitize_return(_read_json(path))

        if mime in ("application/x-ndjson", "application/jsonlines") or ext in (".jsonl", ".ndjson"):
            return _sanitize_return(_read_jsonl(path))

        if mime in ("text/html", "application/xhtml+xml") or ext in (".html", ".htm"):
            return _sanitize_return(_read_html(path))

        if mime in ("application/rtf",) or ext == ".rtf":
            return _sanitize_return(_read_rtf(path))

        # Default: treat as text
        return _sanitize_return(_read_txt(path))

    except ExtractionError as ee:
        logger.exception("[extract] specific failure on %s (%s): %s", path, mime or ext, ee)
        # Best-effort fallback to plain text
        try:
            return _sanitize_return(_read_txt(path))
        except Exception as e2:
            raise ExtractionError(path, f"Failed specialized and text fallback: {e2}") from ee
    except Exception as e:
        logger.exception("[extract] generic failure on %s (%s): %s", path, mime or ext, e)
        try:
            return _sanitize_return(_read_txt(path))
        except Exception:
            # If even plain read fails, bubble up a clear error
            raise ExtractionError(path, f"Failed with fallback: {e}") from e
