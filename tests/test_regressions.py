"""Small CPU regression tests; no external data or checkpoints needed."""
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fbs.model import ChunkHead, FBSModel, PAW
from fbs.utils import batchify, generate_pseudo_labels, load_corpus


class FixedLabels(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def forward(self, h):
        labels = torch.tensor(self.labels[:h.size(1)], device=h.device)
        return F.one_hot(labels, 4).to(h.dtype).unsqueeze(0).expand(h.size(0), -1, -1)


class BatchingTests(unittest.TestCase):
    def test_preserves_every_pair_including_partial_batches(self):
        for length in (0, 1, 2, 4, 5, 12, 13, 14, 50):
            with self.subTest(length=length):
                batches = batchify([list(range(1, length + 1))], 3, 4)
                pairs = []
                for inp, target in batches:
                    self.assertEqual(tuple(inp.shape), (3, 4))
                    valid = target != -100
                    pairs.extend(zip(inp[valid].tolist(), target[valid].tolist()))
                    self.assertTrue(torch.all(inp[~valid] == 0))
                self.assertEqual(pairs, [(i, i + 1) for i in range(1, length)])

    def test_sample_corpus_has_training_targets(self):
        tokens = load_corpus(str(Path(__file__).resolve().parents[1] / 'data/sample.txt'))
        batches = batchify([list(range(1, len(tokens) + 1))], 4, 32)
        self.assertTrue(batches)
        self.assertEqual(sum((t != -100).sum().item() for _, t in batches), len(tokens) - 1)

    def test_invalid_dimensions_fail_clearly(self):
        for batch_size, seq_len in ((0, 4), (-1, 4), (4, 0), (4, -1)):
            with self.assertRaises(ValueError):
                batchify([[1, 2]], batch_size, seq_len)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_chunk_head_ignores_future_tokens_and_boundaries(self):
        ch = ChunkHead(4).eval()
        h = torch.randn(1, 6, 4)
        changed = h.clone()
        changed[:, 3:] += 10
        # A B/I chunk crosses the prefix boundary; its eventual end is
        # unknown to queries in the prefix and must not affect them.
        ch.label_head = FixedLabels([3, 0, 1, 1, 1, 3])
        original, _ = ch(h)
        ch.label_head = FixedLabels([3, 0, 1, 3, 0, 1])
        altered, _ = ch(changed)
        torch.testing.assert_close(original[:, :3], altered[:, :3])
        self.assertTrue(torch.isfinite(original).all())
        torch.testing.assert_close(original[:, 0], torch.zeros_like(original[:, 0]))

    def test_singleton_chunks_cannot_leak_future_states(self):
        ch = ChunkHead(4).eval()
        ch.label_head = FixedLabels([3] * 6)
        h = torch.randn(1, 6, 4)
        changed = h.clone()
        changed[:, 3:] += 10
        original, _ = ch(h)
        altered, _ = ch(changed)
        torch.testing.assert_close(original[:, :3], altered[:, :3])

    def test_model_is_prefix_invariant(self):
        model = FBSModel(16, d_model=8, n_layers=2, n_heads=2, d_ff=16, k_max=2).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 3, 9, 10]])
        with torch.no_grad():
            full, _ = model(tokens, threshold=1.0)
            altered, _ = model(changed, threshold=1.0)
            prefix, _ = model(tokens[:, :3], threshold=1.0)
        torch.testing.assert_close(full[:, :3], altered[:, :3])
        torch.testing.assert_close(full[:, :3], prefix)

    def test_paw_first_horizon_uses_already_shifted_targets(self):
        paw = PAW(4, 7, k_max=1)
        with torch.no_grad():
            paw.window_pred.weight.zero_()
            paw.window_pred.bias.zero_()
        h = torch.randn(1, 3, 4)
        target = torch.tensor([[2, 3, -100]])
        _, loss = paw(h, torch.randn(7, 4), target)
        logits = paw.preview_heads[0](h)
        expected = F.cross_entropy(logits.reshape(-1, 7), target.reshape(-1))
        torch.testing.assert_close(loss.squeeze(), expected, rtol=1e-5, atol=1e-5)

    def test_padded_batch_has_finite_loss_and_gradients(self):
        model = FBSModel(8, d_model=8, n_layers=1, n_heads=2, d_ff=16, k_max=8)
        inp, target = batchify([[1, 2, 3]], 4, 4)[0]
        pseudo = generate_pseudo_labels(inp).masked_fill(target == -100, -100)
        _, loss = model(inp, targets=target, pseudo_labels=pseudo, threshold=1.0)
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
