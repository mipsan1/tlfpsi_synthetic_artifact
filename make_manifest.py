"""Compute SHA-256 of every file in this directory and write MANIFEST.json."""
import hashlib
import json
from pathlib import Path

EXCLUDE = {"MANIFEST.json", "__pycache__", ".DS_Store"}


def main() -> None:
    here = Path(__file__).resolve().parent
    files = sorted(
        p for p in here.iterdir()
        if p.is_file() and p.name not in EXCLUDE and not p.name.endswith(".pyc")
    )
    manifest = {}
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        manifest[p.name] = {
            "size_bytes": p.stat().st_size,
            "sha256": h.hexdigest(),
        }
    Path(here / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote MANIFEST.json with {len(manifest)} entries")


if __name__ == "__main__":
    main()
