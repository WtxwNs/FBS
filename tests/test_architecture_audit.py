import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from fbs.audit import build_report, verify_meta, parameter_inventory, cost_estimate
from fbs.model import PAW


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / 'configs/qwen3_4b_public.config.json').read_text())
        self.run = json.loads((ROOT / 'configs/qwen3_reconstruction.example.json').read_text())

    def test_full_public_shapes_and_unique_counts_without_weights(self):
        try:
            import transformers
        except ImportError:
            self.skipTest('Install requirements-hf.txt')
        report = build_report(self.config, self.run['fbs'])
        inventory = verify_meta(self.config, self.run['fbs'], report)
        # Independent public safetensors metadata: total_size / two BF16 bytes.
        self.assertEqual(inventory['backbone'], 8044936192 // 2)
        self.assertEqual(inventory['auxiliary'], 30160728)
        shapes = {row['name']: row['shape'] for row in inventory['tensors']}
        self.assertEqual(shapes['layers.0.attn.q_proj.weight'], [4096, 2560])
        self.assertEqual(shapes['layers.0.attn.k_proj.weight'], [1024, 2560])
        self.assertNotIn('lm_head.weight', shapes)  # Tied, counted once.

    def test_shape_conflict_and_mutable_revision_fail_before_weight_download(self):
        try:
            from transformers import Qwen3Config
        except ImportError:
            self.skipTest('Install requirements-hf.txt')
        from fbs.hf import QwenFBSModel, validate_backbone_config
        cfg = Qwen3Config(**self.config)
        validate_backbone_config(cfg, self.run['pretrained']['expected_config'])
        for key, wrong in [('hidden_size', 4096), ('num_hidden_layers', 32), ('vocab_size', 32000), ('head_dim', 80)]:
            with self.assertRaisesRegex(ValueError, key):
                validate_backbone_config(cfg, {key: wrong})
        with patch('transformers.AutoConfig.from_pretrained', side_effect=AssertionError('download')):
            with self.assertRaisesRegex(ValueError, 'immutable'):
                QwenFBSModel.from_pretrained('Qwen/Qwen3-4B', 'main')

    def test_inventory_deduplicates_shared_parameters_and_counts_frozen_weights(self):
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(7, 3)
        model.head = torch.nn.Linear(3, 7, bias=False)
        model.head.weight = model.embedding.weight
        model.embedding.weight.requires_grad_(False)
        report = parameter_inventory(model)
        self.assertEqual(report['total'], 21)
        self.assertEqual(report['trainable_by_component']['backbone'], 0)

    def test_paw_matmul_cost_includes_full_vocabulary_despite_top_k(self):
        cfg = dict(self.config, hidden_size=16, vocab_size=32, num_hidden_layers=1)
        fbs = dict(k_max=3, preview_rank=2, chunk_rank=2, gate_hidden=2)
        paw = PAW(16, 32, k_max=3, rank=2, top_k=1).train()
        # Use actual CPU operator FLOPs to check the analytic PAW projection cost.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            paw(torch.randn(1, 1, 16), torch.randn(32, 16))
        matrix_flops = sum(event.flops for event in prof.key_averages() if event.key in ('aten::mm', 'aten::addmm'))
        expected = cost_estimate(cfg, fbs, 3, 1, 1)['paw_all_layers_per_token']
        self.assertEqual(matrix_flops, expected)


if __name__ == '__main__':
    unittest.main()
