"""Merge benchmark rows from fill-in run roots into one main run root.

Fill-in jobs write to their own roots so that concurrent jobs never rewrite the
same result file. A row from a fill-in root replaces the main row for the same
model, target and mode unless the main row succeeded and the new one failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for source in args.sources:
        for path in sorted(source.glob("e2e-*.json")):
            target_path = args.into / path.name
            rows = json.loads(target_path.read_text()) if target_path.exists() else []
            for new in json.loads(path.read_text()):
                key = (new["model"], new["target"], new["mode"])
                old = next(
                    (r for r in rows if (r["model"], r["target"], r["mode"]) == key),
                    None,
                )
                if old is not None and old["status"] == 0 and new["status"] != 0:
                    continue
                if old is not None:
                    rows.remove(old)
                rows.append(new)
                print("MERGED", *key, new["status"], new["error"])  # noqa: T201 - CLI output contract
            target_path.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
