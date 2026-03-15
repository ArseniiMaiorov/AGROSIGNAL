#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def _parse_major_minor(version: str) -> tuple[int, int]:
    parts = (version or "0.0").split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def _check_version_ranges() -> list[str]:
    errors: list[str] = []
    import numpy as np
    import scipy
    import sklearn

    np_major, np_minor = _parse_major_minor(np.__version__)
    if (np_major, np_minor) >= (2, 3):
        errors.append(f"numpy must be <2.3, got {np.__version__}")

    sp_major, sp_minor = _parse_major_minor(scipy.__version__)
    if (sp_major, sp_minor) != (1, 14):
        errors.append(f"scipy must be 1.14.x, got {scipy.__version__}")

    sk_major, sk_minor = _parse_major_minor(sklearn.__version__)
    if (sk_major, sk_minor) < (1, 5):
        errors.append(f"scikit-learn must be >=1.5, got {sklearn.__version__}")

    return errors


def _check_pickle_compat(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"pickle file not found: {path}"]
    try:
        with open(path, "rb") as fh:
            _ = pickle.load(fh)
    except Exception as exc:
        errors.append(f"failed to load pickle {path}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate training runtime compatibility")
    parser.add_argument(
        "--pickle",
        type=Path,
        default=Path("backend/models/object_classifier.pkl"),
        help="Pickle artifact to verify sklearn compatibility",
    )
    parser.add_argument(
        "--skip-pickle",
        action="store_true",
        help="Skip pickle load check (versions only)",
    )
    args = parser.parse_args()

    errors = _check_version_ranges()
    if not args.skip_pickle:
        errors.extend(_check_pickle_compat(args.pickle))

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print("OK: training environment compatibility checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
