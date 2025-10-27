# core/file_utils.py
import os
import hashlib
from pathlib import Path

def read_head(path: Path, max_chars=4000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""

def is_text_file(path: Path) -> bool:
    try:
        import magic
        return magic.from_file(str(path), mime=True).startswith("text")
    except Exception:
        return path.suffix.lower() in {
            ".cbl", ".cob", ".cpy", ".txt", ".json", ".yaml", ".yml", ".xml", ".sql"
        }

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def secure_unzip(zip_path: Path, dest_dir: Path):
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        for m in z.infolist():
            target = (dest_dir / m.filename).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise ValueError(f"Unsafe path: {m.filename}")
            if not m.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                z.extract(m, dest_dir)
