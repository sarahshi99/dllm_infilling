from __future__ import annotations

import json

from .manifests import read_jsonl, write_json, write_jsonl
from .protocol import EXPERIMENT_DIR, FIXED_FULL_ROWS, sha256_file
from .runner import FIXED_RESULTS_PATH


def main() -> None:
    rows = read_jsonl(FIXED_RESULTS_PATH)
    widths = ["1", "4", "8", "16", "inf"]
    index = {
        "source_path": "results/fixed_full_1000_results.jsonl",
        "source_rows": len(rows),
        "source_sha256": sha256_file(FIXED_RESULTS_PATH),
        "shards": {},
    }
    for width in widths:
        selected = [row for row in rows if row["w"] == width]
        if not selected:
            continue
        if len(selected) != FIXED_FULL_ROWS:
            raise AssertionError(f"w={width} expected {FIXED_FULL_ROWS}, found {len(selected)}")
        path = EXPERIMENT_DIR / f"results/fixed_full_1000_w{width}_results.jsonl"
        write_jsonl(path, selected)
        index["shards"][width] = {
            "path": f"results/{path.name}",
            "rows": len(selected),
            "sha256": sha256_file(path),
        }
    if sum(item["rows"] for item in index["shards"].values()) != len(rows):
        raise AssertionError("result shards do not cover the combined JSONL")
    write_json(EXPERIMENT_DIR / "results/fixed_full_1000_results.index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
