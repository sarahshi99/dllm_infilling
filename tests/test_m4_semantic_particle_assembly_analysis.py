from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.m4_semantic_particle_assembly import run_analysis


def row(index: int, method: str, passed: bool, forwards: int) -> dict[str, object]:
    middle_hash = f"{method}-{index}" if method != "m4_best_single_particle" else f"best-{index}"
    return {
        "row_key": f"row-{index}", "task_group": f"HumanEval/{index}", "length_bucket": "medium", "status": "ok", "passed": passed,
        "candidate_middle_sha256": middle_hash,
        "connector_indices": list(range(index % 3)),
        "assembly_metadata": {"fragment_count": 1 + index % 3, "provider_candidate_count": 1, "assembly_fallback_to_best": False},
        "metrics": {"standalone_actual_forward_count": forwards, "actual_forward_count": forwards, "standalone_token_budget": 30720 if forwards == 512 else 34816, "total_sec_including_probe": 1.0},
    }


class M4SemanticParticleAssemblyAnalysisTest(unittest.TestCase):
    def test_grouped_output_keeps_primary_equal_compute_comparison_and_activation(self) -> None:
        methods = ("m4_best_single_particle", "m4_assembly_without_repair", "m4_assembly_with_repair")
        payloads = {method: [row(index, method, bool((index + offset) % 3), 576 if method == methods[2] else 512) for index in range(148)] for offset, method in enumerate(methods)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for method in methods:
                path = root / f"{method}.jsonl"
                path.write_text("".join(json.dumps(item) + "\n" for item in payloads[method]), encoding="utf-8")
                paths.append(path)
            summary = run_analysis(*paths, root / "out", bootstrap_replicates=20)
            activation = json.loads((root / "out" / "activation_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["primary_comparison"], [methods[1], methods[0]])
        self.assertEqual(summary["cost_contract"][methods[1]], 512)
        self.assertEqual(activation["row_count"], 148)


if __name__ == "__main__":
    unittest.main()
