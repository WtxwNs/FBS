import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from fbs.model import FBSModel, PAW
from fbs.inference import new_state, decode_step, prefill, generate
from fbs.checkpoint import save_checkpoint, load_checkpoint
from fbs.data import bios_from_offsets, resolve_spans, assert_disjoint


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def model(self, **kw):
        return FBSModel(16, d_model=8, n_layers=2, n_heads=2, d_ff=16, k_max=2, **kw).eval()

    def test_hard_skip_executes_no_qkv_or_ffn_and_preserves_position(self):
        model = self.model()
        state = new_state(model)
        with patch.object(model.layers[0].attn.q_proj, 'forward', side_effect=AssertionError('Q executed')), patch.object(model.layers[0].ffn, 'forward', side_effect=AssertionError('FFN executed')):
            decode_step(model, torch.tensor([1]), state, gate_actions=[True, False])
        self.assertEqual(state.layers[0].positions, [])
        self.assertEqual(state.layers[1].positions, [0])
        decode_step(model, torch.tensor([2]), state, gate_actions=[False, True])
        self.assertEqual(state.layers[0].positions, [1])
        self.assertEqual(state.layers[1].positions, [0])
        self.assertEqual((state.executed, state.skipped, state.position), (2, 2, 2))

    def test_dense_and_cached_no_skip_match(self):
        model = self.model()
        ids = torch.tensor([[1,2,3,4]])
        cached, _ = prefill(model, ids, threshold=1.)
        dense, _ = model(ids, targets=torch.tensor([[2,3,4,5]]), threshold=1.)
        torch.testing.assert_close(cached, dense, atol=2e-6, rtol=2e-6)

    def test_zero_preview_is_really_zero_including_compressor_bias(self):
        paw = PAW(8, 16, k_max=4).eval()
        with torch.no_grad():
            paw.window_pred.weight.zero_()
            paw.window_pred.bias.fill_(-50)
        with patch.object(paw.preview_heads[0], 'forward', side_effect=AssertionError('unused horizon')):
            z, _ = paw(torch.randn(1, 2, 8), torch.randn(16,8))
        self.assertEqual(z.count_nonzero().item(), 0)

    def test_fixed_length_generation_and_checkpoint_roundtrip(self):
        model = self.model(preview_rank=2)
        ids = torch.tensor([[1,2,3]])
        expected, _ = generate(model, ids, 5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            actual, state = generate(restored.eval(), ids, 5)
        torch.testing.assert_close(expected, actual)
        self.assertEqual(actual.shape[1], 8)
        self.assertEqual(state.position, 7)

    def test_bios_alignment_and_split_overlap(self):
        spans = resolve_spans([(0,2),(2,4),(4,6)], [(0,4)])
        self.assertEqual(spans, [(0,4),(4,6)])
        self.assertEqual(bios_from_offsets([(0,2),(2,4),(4,6),(0,0)], spans), [0,1,3,-100])
        with self.assertRaises(ValueError):
            assert_disjoint(['What is A?'], ['what  is a !'])

    def test_disabled_gate_neither_executes_nor_changes_training_gradients(self):
        model = self.model(use_paw=False, use_ch=False, use_sg=False)
        ids = torch.tensor([[1,2,3]])
        with patch.object(model.layers[0].sg, 'forward', side_effect=AssertionError('disabled gate ran')):
            model(ids)
            model.train()
            _, loss = model(ids, targets=torch.tensor([[2,3,4]]))
            loss.backward()
        self.assertTrue(all(p.grad is None for layer in model.layers for p in layer.sg.parameters()))

    def test_calibration_updates_only_paw_and_sg(self):
        import copy
        from fbs.calibrate import freeze_backbone, update
        model = self.model()
        reference = copy.deepcopy(model).eval()
        freeze_backbone(model)
        before = {n: p.detach().clone() for n,p in model.named_parameters()}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        result = update(model, reference, optimizer, torch.tensor([1,2]), torch.tensor([3,4]), new_tokens=3, epochs=1)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        changed = []
        for name, parameter in model.named_parameters():
            if not torch.equal(before[name], parameter):
                changed.append(name)
                self.assertTrue('.paw.' in name or '.sg.' in name, name)
        self.assertTrue(any('.sg.' in name for name in changed))

    def test_qwen_adapter_recovers_original_when_modules_disabled(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest('Install requirements-hf.txt')
        from fbs.hf import QwenFBSModel
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             # Qwen3-4B has hidden_size != num_heads * head_dim.
                             head_dim=8, max_position_embeddings=64, attention_dropout=0.)
        backbone = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[1,2,3,4]])
        with torch.no_grad():
            original = backbone(ids).logits
            model = QwenFBSModel(backbone, k_max=2, preview_rank=2, chunk_rank=2, gate_hidden=4,
                                  use_paw=False, use_ch=False, use_sg=False).eval()
            cached, _ = model(ids)
            dense, _ = model(ids, targets=torch.tensor([[2,3,4,5]]), threshold=1.)
        torch.testing.assert_close(original, cached, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(original, dense, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
