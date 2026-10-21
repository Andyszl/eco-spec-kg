"""Copy a checksummed, prediction-free review packet; never run inference."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil


def _safe_relative(value, prefix=None):
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError("unsafe manifest path")
    path = PurePosixPath(value)
    if (path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/"))
            or (prefix and path.parts[0] != prefix)):
        raise ValueError("unsafe manifest path")
    return path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    lock = json.loads((root / "protocol/packet_lock.json").read_text(encoding="utf-8"))
    files = lock["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("empty packet inventory")
    for rel, expected in files.items():
        path = root / _safe_relative(rel, "packet")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid checksum format")
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("symlink in packet path")
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"packet checksum mismatch: {rel}")
    inventory = set()
    for path in (root / "packet").rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in packet inventory")
        if path.is_file():
            inventory.add(path.relative_to(root).as_posix())
    if inventory != set(files):
        raise ValueError("packet inventory differs from lock")
    meta = json.loads((root / "packet/source_manifest.json").read_text(encoding="utf-8"))
    if (meta["gold_available"] is not False or meta["inference_ready"] is not False
            or meta["project_exposure_human_confirmation"] != "pending"):
        raise ValueError("this is an unreviewed material packet, not approved gold")
    sources = meta["sources"]
    if meta["source_count"] != 2 or len(sources) != 2 or len({s["id"] for s in sources}) != 2:
        raise ValueError("expected two distinct source documents")
    if len({s["sha256"] for s in sources}) != 2:
        raise ValueError("duplicate sources")
    for source in sources:
        source_path = root / "packet" / _safe_relative(source["file"], "sources")
        if source["sha256"] != _digest(source_path):
            raise ValueError("source manifest checksum mismatch")
        if not source_path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("not a PDF source")
    return root, files, meta


def prepare(root, out):
    root, files, meta = verify(root)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists, preserved: {out}")
    if out.resolve().is_relative_to(root):
        raise ValueError("review output must be outside the immutable checkout")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "packet", out)
    for rel, expected in files.items():
        copied = out / PurePosixPath(rel).relative_to("packet")
        if _digest(copied) != expected:
            raise ValueError("copied packet checksum mismatch")
    status = {
        "status": "material_copy_verified_human_review_required",
        "source_count": len(meta["sources"]),
        "source_count_means": "documents, not independent experiments",
        "model_calls": 0,
        "gold_available": False,
        "inference_ready": False,
        "source_exposure_human_confirmation": "pending",
        "required_next_action": "human_blind_annotation_and_review",
        "original_packet_lock_sha256": _digest(root / "protocol/packet_lock.json"),
    }
    (out / "preparation_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = prepare(args.root, args.out)
    print("BLIND_PACKET_OK=true")
    print(f"SOURCE_COUNT={status['source_count']}")
    print("MODEL_CALLS=0")
    print("INFERENCE_READY=false")
    print("HUMAN_REVIEW_REQUIRED=true")
    print(f"PACKET_DIRECTORY={args.out.absolute()}")


if __name__ == "__main__":
    main()
