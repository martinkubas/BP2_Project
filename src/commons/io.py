from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_json_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.glob("*.json")):
        if path.is_file():
            yield path


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
