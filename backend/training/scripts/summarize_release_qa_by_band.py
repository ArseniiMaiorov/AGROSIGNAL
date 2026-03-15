#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REGION_BANDS = {
    "Kaliningrad": "central",
    "Belgorod": "south",
    "Voronezh": "south",
    "Krasnodar": "south",
    "Stavropol": "south",
    "Tatarstan": "central",
    "Bashkortostan": "central",
    "Orenburg": "south",
    "Altai Krai": "central",
    "Novosibirsk": "north",
    "Amur": "north",
    "Primorsky Krai": "north",
}


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize release QA matrix by north/central/south bands")
    parser.add_argument("results", type=Path, nargs="?", default=Path("backend/training/release_russia_qa_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("backend/debug/runs/release_qa_band_summary.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[REGION_BANDS.get(str(row.get("region") or ""), "unknown")].append(row)

    summary: dict[str, Any] = {}
    for band, items in grouped.items():
        queue_wall = [float(item["queue_wall_s"]) for item in items if item.get("queue_wall_s") is not None]
        field_counts = [float(item["field_count"]) for item in items if item.get("field_count") is not None]
        fail_count = sum(1 for item in items if str(item.get("status")) != "done")
        stale_count = sum(1 for item in items if bool(item.get("stale_running")))
        empty_count = sum(1 for item in items if bool(item.get("empty_output")))
        summary[band] = {
            "runs": len(items),
            "fail_count": fail_count,
            "stale_count": stale_count,
            "empty_count": empty_count,
            "success_rate": round((len(items) - fail_count) / max(len(items), 1), 4),
            "avg_queue_wall_s": _avg(queue_wall),
            "avg_field_count": _avg(field_counts),
        }

    payload = {
        "region_bands": summary,
        "source_results": str(args.results.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
