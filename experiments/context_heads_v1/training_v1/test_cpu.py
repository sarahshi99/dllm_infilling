"""CPU tests of data isolation, actual optimization, aggregation and resume."""
import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..contract import HeadBatch
from ..heads import ContextHead
from ..loss import teacher_kl
from ..test_cpu import cfg, fixture
from .data import FIELDS, PackedBank, load_arrays, metadata, sha
from .run import evaluate, save_checkpoint, update


class TrainingCPU(unittest.TestCase):
    def test_metadata_split_leakage_duplicates_and_only_allowed_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"states.jsonl.gz"
            def rows(values):
                with gzip.open(path, "wt") as f:
                    for sid, split, group in values:
                        row = dict(state_id=sid, raw_path="s.npz", raw_sha256="s", teacher_path="t.npz",
                                   teacher_sha256="t", group=group, split=split, span="single", order="global",
                                   arm="Z4", history_length=1, passed="must not be retained")
                        f.write(json.dumps(row)+"\n")
            rows([("b", "validation", "g2"), ("a", "train", "g1")])
            result = metadata(path)
            self.assertEqual([r["state_id"] for r in result], ["a", "b"])
            self.assertNotIn("passed", result[0])
            for invalid in ([("a", "train", "g"), ("b", "validation", "g")],
                            [("a", "train", "g1"), ("a", "validation", "g2")],
                            [("a", "external_test", "g1"), ("b", "validation", "g2")]):
                rows(invalid)
                with self.assertRaises(ValueError):
                    metadata(path)

    def test_only_contract_arrays_read_teacher_full_support_and_prefix_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            batch = fixture(b=1)
            one = batch.queries(batch.query_indices[:, 3:4], batch.history_mask[:, 3:4], batch.old_logits[:, 3:4])
            def save(current):
                arrays = {key: value.numpy() for key, value in vars(current).items()}
                # Reading this unrelated object array would fail allow_pickle=False.
                arrays["unrelated_object"] = np.array([None], dtype=object)
                np.savez(path/"state.npz", **arrays)
                np.savez(path/"teacher.npz", fresh_logits=np.zeros(11, dtype=np.float32))
                return dict(state_id="synthetic", raw_path="state.npz", raw_sha256=sha(path/"state.npz"),
                            teacher_path="teacher.npz", teacher_sha256=sha(path/"teacher.npz"), history_length=int(current.history_mask.sum()))
            row = save(one)
            arrays = load_arrays(row, path, cfg())
            self.assertEqual(set(arrays), {*FIELDS, "teacher_logits"})
            self.assertEqual(arrays["teacher_logits"].shape, (1, 1, 11))
            bad_mask = one.history_mask.clone()
            bad_mask[..., 1] = False
            row = save(replace(one, history_mask=bad_mask))
            with self.assertRaises(ValueError):
                load_arrays(row, path, cfg())
            row = save(one)
            with (path/"teacher.npz").open("ab") as f:
                f.write(b"changed")
            with self.assertRaises(ValueError):
                load_arrays(row, path, cfg())

    def test_actual_kl_optimization_and_exact_cpu_optimizer_resume(self):
        batch = fixture(b=2)
        batch = batch.queries(batch.query_indices[:, 1:2], batch.history_mask[:, 1:2], batch.old_logits[:, 1:2])
        teacher = torch.zeros(2, 1, 11)
        teacher[..., 2] = 3
        head = ContextHead("H0-static", cfg())
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=0.)
        def loss(current):
            return teacher_kl(batch.old_logits, current(batch).delta_logits, teacher, torch.ones(2, 1, dtype=torch.bool)).item()
        initial = loss(head)
        for _ in range(2):
            update(head, optimizer, batch, teacher)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"last.pt"
            save_checkpoint(path, dict(head=head.state_dict(), optimizer=optimizer.state_dict()))
            for _ in range(2):
                update(head, optimizer, batch, teacher)
            resumed = ContextHead("H0-static", cfg(seed=9))
            state = torch.load(path, map_location="cpu", weights_only=False)
            resumed.load_state_dict(state["head"])
            opt2 = torch.optim.AdamW(resumed.parameters(), lr=.01, weight_decay=0.)
            opt2.load_state_dict(state["optimizer"])
            for _ in range(2):
                update(resumed, opt2, batch, teacher)
            for first, second in zip(head.parameters(), resumed.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertLess(loss(head), initial)

    def test_memmap_batches_readonly_source_and_validation_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            batch = fixture(b=2)
            one = batch.queries(batch.query_indices[:, 1:2], batch.history_mask[:, 1:2], batch.old_logits[:, 1:2])
            values = {key: value.numpy() for key, value in vars(one).items()}
            values["teacher_logits"] = values["old_logits"].copy()
            for key, value in values.items():
                np.save(path/(key+".npy"), value)
            (path/"manifest.json").write_text(json.dumps(dict(fields=list(values))))
            rows = [dict(split="validation", group="g"+str(i), history_length=1, order="global", span="single", arm="Z4") for i in range(2)]
            (path/"records.json").write_text(json.dumps(rows))
            bank = PackedBank(path)
            idx = bank.indices("validation")
            loaded, teacher = bank.batch(idx, "cpu")
            loaded.token_ids.fill_(9999)
            self.assertFalse(np.any(bank.arrays["token_ids"] == 9999))
            result = evaluate(None, bank, idx, torch.device("cpu"), 1)
            self.assertEqual(result["by_bucket"]["all"]["count"], 2)
            self.assertEqual(result["by_bucket"]["all"]["kl_T1"], 0.)
            self.assertEqual(result["group_mean_kl_T1"], 0.)


if __name__ == "__main__":
    unittest.main()
