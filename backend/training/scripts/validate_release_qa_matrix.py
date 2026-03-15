#!/usr/bin/env python3
"""Validate release QA matrix for Russian autodetect regression runs."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


EXPECTED_REGIONS = {
    "Kaliningrad",
    "Belgorod",
    "Voronezh",
    "Krasnodar",
    "Stavropol",
    "Tatarstan",
    "Bashkortostan",
    "Orenburg",
    "Altai Krai",
    "Novosibirsk",
    "Amur",
    "Primorsky Krai",
}


def main(path_str: str) -> int:
    path = Path(path_str).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = list(payload.get("items") or [])
    if len(items) != 36:
        raise SystemExit(f"Expected 36 QA AOI items, got {len(items)}")
    modes = list(payload.get("modes") or [])
    if modes != ["standard", "quality"]:
        raise SystemExit(f"Expected modes ['standard', 'quality'], got {modes}")

    region_counts = Counter()
    seen_ids = set()
    for item in items:
        item_id = str(item.get("id") or "")
        region = str(item.get("region") or "")
        lat = float(item.get("lat"))
        lon = float(item.get("lon"))
        if not item_id:
            raise SystemExit("Each QA item must have non-empty id")
        if item_id in seen_ids:
            raise SystemExit(f"Duplicate QA item id: {item_id}")
        seen_ids.add(item_id)
        if region not in EXPECTED_REGIONS:
            raise SystemExit(f"Unexpected region: {region}")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise SystemExit(f"Invalid coordinates for {item_id}: {lat}, {lon}")
        region_counts[region] += 1

    missing = EXPECTED_REGIONS - set(region_counts)
    if missing:
        raise SystemExit(f"Missing regions: {sorted(missing)}")
    for region, count in region_counts.items():
        if count != 3:
            raise SystemExit(f"Region {region} must contain exactly 3 AOI, got {count}")

    print(f"OK: {path} validated, {len(items)} AOI and {len(items) * len(modes)} total runs")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validate_release_qa_matrix.py <matrix.json>")
    raise SystemExit(main(sys.argv[1]))
