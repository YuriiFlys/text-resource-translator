"""Content-addressed сховище оригінальних артефактів на диску.

Шлях: RAW_DIR/<source>/<sha256[:2]>/<sha256>.<ext> + сайдкар <sha256>.meta.json.
Однаковий вміст не дублюється; повторний put повертає той самий шлях.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredFile:
    path: Path
    sha256: str
    size_bytes: int
    created: bool  # False, якщо файл з таким вмістом уже був


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, source: str, data: bytes, ext: str, meta: dict | None = None) -> StoredFile:
        sha = hashlib.sha256(data).hexdigest()
        ext = ext.lstrip(".") or "bin"
        directory = self.root / source / sha[:2]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sha}.{ext}"
        created = not path.exists()
        if created:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        sidecar = directory / f"{sha}.meta.json"
        payload = {
            "sha256": sha,
            "size_bytes": len(data),
            "stored_at": datetime.now(UTC).isoformat(),
            **(meta or {}),
        }
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return StoredFile(path=path, sha256=sha, size_bytes=len(data), created=created)

    def read(self, path: str | Path) -> bytes:
        return Path(path).read_bytes()
