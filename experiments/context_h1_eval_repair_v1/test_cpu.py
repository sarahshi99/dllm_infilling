import unittest
from dataclasses import replace
import torch
from experiments.context_heads_v1.test_cpu import cfg,fixture
from experiments.context_h1_training_v1.train import initialize
from experiments.context_heads_v1.integration_v1.adapter import KeywordHeadAdapter

class RepairTests(unittest.TestCase):
    def test_h1_live_bf16_matches_trained_fp32_definition(self):
        head=initialize(cfg(seed=20261107),'H1-context-prev')
        head.output.weight.data.normal_(std=.01)
        head.gamma[-1].weight.data.normal_(std=.01)
        batch=fixture();batch=replace(batch,old_hidden=batch.old_hidden.bfloat16())
        expected=head(replace(batch,old_hidden=batch.old_hidden.float())).delta_logits
        actual=KeywordHeadAdapter(head)(**vars(batch))['delta_logits']
        self.assertTrue(torch.equal(actual,expected))
        self.assertGreater(float(actual.abs().sum()),0)
