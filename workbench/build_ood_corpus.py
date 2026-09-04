"""Build the offline OOD evaluation corpus ``data/ood-suite-v1``.

Three domains, each written as one parquet with a single ``text`` column
(same schema the evaluator's ``_load_rows`` expects):

* ``code`` — Python source files from installed site-packages (offline).
* ``news`` — ag_news train rows from hf-mirror (English news).
* ``zh``   — XLSum v2 chinese_simplified articles from hf-mirror (Chinese news).

The script is deterministic: row selection uses sorted/hash order, never
random.  Re-running it must reproduce byte-identical parquet content for the
offline domains and stable content for the downloaded ones (rows are taken in
dataset order from a fixed offset).

Usage:
    python workbench/build_ood_corpus.py            # build all domains
    python workbench/build_ood_corpus.py --list     # show current files
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "ood-suite-v1"

AG_NEWS_URL = (
    "https://hf-mirror.com/datasets/fancyzhx/ag_news/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
XLSUM_URL = (
    "https://hf-mirror.com/datasets/csebuetnlp/xlsum/resolve/main/"
    "data/chinese_simplified_XLSum_v2.0.tar.bz2"
)

CODE_FILES = 60
CODE_MAX_CHARS = 8000
CODE_MIN_LINES = 120
NEWS_ROWS = 400
ZH_ROWS = 400
ZH_MIN_CHARS = 200

# Package roots scanned for the code domain.  Ordered for determinism.
CODE_PACKAGES = ("torch", "transformers", "numpy", "pyarrow", "sympy")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _site_packages() -> list[Path]:
    found: list[Path] = []
    for entry in sys.path:
        path = Path(entry) if entry else Path.cwd()
        if path.name in {"site-packages", "dist-packages"} and path.is_dir():
            found.append(path)
    return found


def _collect_code_rows() -> list[str]:
    packages = _site_packages()
    if not packages:
        raise RuntimeError("no site-packages directory found on sys.path")
    candidates: list[Path] = []
    for site in packages:
        for package in CODE_PACKAGES:
            root = site / package
            if not root.is_dir():
                continue
            candidates.extend(root.rglob("*.py"))
    # Deterministic order: hash of the path string.
    candidates.sort(key=lambda item: _sha256(str(item).encode()) + str(item))
    rows: list[str] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.count("\n") < CODE_MIN_LINES:
            continue
        rows.append(text[:CODE_MAX_CHARS])
        if len(rows) >= CODE_FILES:
            break
    if len(rows) < 8:
        raise RuntimeError(f"only found {len(rows)} usable code files")
    return rows


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ood-corpus-builder/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _collect_news_rows(cache: Path) -> list[str]:
    import pyarrow.parquet as parquet

    if not cache.is_file():
        print(f"downloading ag_news train parquet -> {cache}", flush=True)
        cache.write_bytes(_download(AG_NEWS_URL))
    table = parquet.read_table(cache, columns=["text"])
    rows = [str(value) for value in table["text"].to_pylist()[:NEWS_ROWS]]
    rows = [row.strip().replace("\n", " ") for row in rows if row and row.strip()]
    return rows


def _collect_zh_rows(cache: Path) -> list[str]:
    if not cache.is_file():
        print(f"downloading XLSum chinese_simplified -> {cache}", flush=True)
        cache.write_bytes(_download(XLSUM_URL))
    rows: list[str] = []
    with tarfile.open(cache, "r:bz2") as archive:
        # Prefer the small val/test JSONL members over the large train file.
        members = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".jsonl")
        ]
        members.sort(key=lambda member: (not member.name.endswith("val.jsonl"),
                                         not member.name.endswith("test.jsonl"),
                                         member.size))
        for member in members:
            if len(rows) >= ZH_ROWS:
                break
            with io.TextIOWrapper(archive.extractfile(member), encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    article = json.loads(line)
                    text = str(article.get("text", "")).strip()
                    if len(text) >= ZH_MIN_CHARS:
                        rows.append(text.replace("\n", " "))
                    if len(rows) >= ZH_ROWS:
                        break
    if len(rows) < 16:
        raise RuntimeError(f"only extracted {len(rows)} chinese articles")
    return rows


def _write_parquet(name: str, rows: list[str]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    out = OUT_DIR / f"{name}-00000-of-00001.parquet"
    table = pa.table({"text": rows})
    parquet.write_table(table, out)
    print(f"wrote {out} ({len(rows)} rows, {out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list existing suite files and exit")
    parser.add_argument(
        "--domains", default="code,news,zh",
        help="comma-separated domains to (re)build; existing files are overwritten",
    )
    args = parser.parse_args()
    if args.list:
        for path in sorted(OUT_DIR.glob("*-00000-of-00001.parquet")):
            print(f"{path.name}  {path.stat().st_size / 1e6:.1f} MB")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_cache = OUT_DIR / "_downloads"
    download_cache.mkdir(exist_ok=True)
    requested = [item.strip() for item in args.domains.split(",") if item.strip()]

    if "code" in requested:
        _write_parquet("code", _collect_code_rows())
    if "news" in requested:
        try:
            _write_parquet("news", _collect_news_rows(download_cache / "ag_news-train.parquet"))
        except Exception as exc:  # network optional; keep the rest of the suite
            print(f"[warn] news domain skipped: {type(exc).__name__}: {exc}", flush=True)
    if "zh" in requested:
        try:
            _write_parquet("zh", _collect_zh_rows(download_cache / "xlsum-zh.tar.bz2"))
        except Exception as exc:
            print(f"[warn] zh domain skipped: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
