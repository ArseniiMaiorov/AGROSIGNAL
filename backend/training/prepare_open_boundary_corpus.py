#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_registry(registry_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(registry_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("manifest_file", str(path.resolve()))
            items.append(payload)
    return items


def _normalize_source_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare open/public boundary corpus manifest")
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("backend/training/dataset_registry"),
        help="Directory with open/public dataset registry manifests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/debug/runs/open_boundary_corpus_manifest.json"),
        help="Output manifest path",
    )
    parser.add_argument(
        "--years",
        default="2023,2024,2025",
        help="Comma-separated Sentinel observation years",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for open/public corpus fetch stages",
    )
    args = parser.parse_args()

    registry_dir = args.registry_dir.resolve()
    if not registry_dir.exists():
        raise SystemExit(f"Registry dir not found: {registry_dir}")

    registry = _load_registry(registry_dir)
    if not registry:
        raise SystemExit(f"No dataset manifests found in {registry_dir}")

    sources = []
    for item in registry:
        source_name = str(item.get("source") or "unknown")
        sources.append(
            {
                "source": source_name,
                "source_id": _normalize_source_name(source_name),
                "artifact_uri": item.get("artifact_uri"),
                "license": item.get("license"),
                "country": item.get("country") or [],
                "year": item.get("year"),
                "label_type": item.get("label_type"),
                "prepare_recipe": item.get("prepare_recipe"),
                "manifest_file": item.get("manifest_file"),
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "open_public_only",
        "years": [token.strip() for token in str(args.years).split(",") if token.strip()],
        "tile_size": int(args.tile_size),
        "source_count": len(sources),
        "sources": sources,
        "training_target": "boundary_unet_v3_cpu",
        "notes": [
            "This manifest intentionally references only open/public corpora.",
            "Sentinel Hub failover is resolved at runtime by the provider client.",
            "Manual/private GT is excluded from this production-safe path.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
