import sys
import os
import unittest
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.losses import CrossEntropyWithLogits
from src.core.tensor import Tensor
from src.utils.backend import xp


def to_numpy(array_like):
    if xp.__name__ == "cupy":
        return xp.asnumpy(array_like)
    return array_like


class TestCrossEntropy(unittest.TestCase):
    def test_value_and_grad_match_pytorch_3d(self):
        import torch

        rng = np.random.default_rng(42)
        batch_size, seq_len, num_classes = 4, 7, 5
        logits_np = rng.standard_normal((batch_size, seq_len, num_classes)).astype(np.float64)
        targets_np = rng.integers(0, num_classes, size=(batch_size, seq_len), dtype=np.int64)

        logits = Tensor(logits_np, requires_grad=True)
        targets = Tensor(targets_np, requires_grad=False)
        loss = CrossEntropyWithLogits(logits, targets, use_mask=False)
        loss.backward()

        logits_pt = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        targets_pt = torch.tensor(targets_np, dtype=torch.int64)
        ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss_pt = ce(logits_pt.view(-1, num_classes), targets_pt.view(-1))
        loss_pt.backward()

        self.assertTrue(np.allclose(loss.data, loss_pt.item(), atol=1e-4))

        grad_np = to_numpy(logits.grad.data)
        grad_pt = logits_pt.grad.detach().numpy().reshape(batch_size, seq_len, num_classes)
        self.assertTrue(np.allclose(grad_np, grad_pt, atol=1e-4))

    def test_2d_and_3d_equivalence(self):
        import torch

        rng = np.random.default_rng(0)
        batch_size, num_classes = 6, 11
        logits2d_np = rng.standard_normal((batch_size, num_classes)).astype(np.float64)
        targets1d_np = rng.integers(0, num_classes, size=(batch_size,), dtype=np.int64)

        # 2D path
        logits2d = Tensor(logits2d_np, requires_grad=True)
        targets1d = Tensor(targets1d_np, requires_grad=False)
        loss2d = CrossEntropyWithLogits(logits2d, targets1d, use_mask=False)
        loss2d.backward()

        # 3D singleton-sequence path
        logits3d_np = logits2d_np[:, None, :]
        targets2d_np = targets1d_np[:, None]
        logits3d = Tensor(logits3d_np, requires_grad=True)
        targets2d = Tensor(targets2d_np, requires_grad=False)
        loss3d = CrossEntropyWithLogits(logits3d, targets2d, use_mask=False)
        loss3d.backward()

        print(f"\n=== 2D vs 3D Loss Comparison ===")
        print(f"2D loss: {loss2d.data}")
        print(f"3D loss: {loss3d.data}")
        print(f"Difference: {abs(loss2d.data - loss3d.data)}")
        print(f"2D loss shape: {loss2d.data.shape if hasattr(loss2d.data, 'shape') else 'scalar'}")
        print(f"3D loss shape: {loss3d.data.shape if hasattr(loss3d.data, 'shape') else 'scalar'}")
        
        self.assertTrue(np.allclose(loss2d.data, loss3d.data, atol=1e-4))

        grad2d_np = to_numpy(logits2d.grad.data)
        grad3d_np = to_numpy(logits3d.grad.data)[:, 0, :]
        self.assertTrue(np.allclose(grad2d_np, grad3d_np, atol=1e-4))

        # Cross-check with PyTorch 2D path
        import torch
        logits2d_pt = torch.tensor(logits2d_np, dtype=torch.float64, requires_grad=True)
        targets1d_pt = torch.tensor(targets1d_np, dtype=torch.int64)
        ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss_pt = ce(logits2d_pt, targets1d_pt)
        loss_pt.backward()
        
        print(f"\n=== PyTorch Comparison ===")
        print(f"Our 2D loss: {loss2d.data}")
        print(f"PyTorch loss: {loss_pt.item()}")
        print(f"Difference: {abs(loss2d.data - loss_pt.item())}")
        print(f"PyTorch loss type: {type(loss_pt.item())}")
        print(f"Our loss type: {type(loss2d.data)}")
        
        self.assertTrue(np.allclose(loss2d.data, loss_pt.item(), atol=1e-4))
        self.assertTrue(np.allclose(grad2d_np, logits2d_pt.grad.detach().numpy(), atol=1e-4))

    def test_grad_sums_to_zero_per_position(self):
        rng = np.random.default_rng(123)
        batch_size, seq_len, num_classes = 3, 4, 7
        logits_np = rng.standard_normal((batch_size, seq_len, num_classes)).astype(np.float64)
        targets_np = rng.integers(0, num_classes, size=(batch_size, seq_len), dtype=np.int64)

        logits = Tensor(logits_np, requires_grad=True)
        targets = Tensor(targets_np, requires_grad=False)
        loss = CrossEntropyWithLogits(logits, targets, use_mask=False)
        loss.backward()

        grad_np = to_numpy(logits.grad.data)
        sums = grad_np.sum(axis=-1)
        
        print(f"\n=== Gradient Sums Debug ===")
        print(f"Gradient shape: {grad_np.shape}")
        print(f"Gradient sums shape: {sums.shape}")
        print(f"Gradient sums: {sums}")
        print(f"Max absolute sum: {np.max(np.abs(sums))}")
        print(f"Min absolute sum: {np.min(np.abs(sums))}")
        print(f"Mean absolute sum: {np.mean(np.abs(sums))}")
        print(f"All sums close to zero: {np.allclose(sums, 0.0, atol=1e-5)}")
        
        self.assertTrue(np.allclose(sums, 0.0, atol=1e-5))

    def test_invalid_shapes_raise(self):
        # 1D logits
        with self.assertRaises(ValueError):
            CrossEntropyWithLogits(Tensor(np.zeros((5,), dtype=np.float64), requires_grad=True),
                         Tensor(np.zeros((1,), dtype=np.int64), requires_grad=False))

        # 4D logits
        with self.assertRaises(ValueError):
            CrossEntropyWithLogits(Tensor(np.zeros((2, 3, 4, 5), dtype=np.float64), requires_grad=True),
                         Tensor(np.zeros((2, 3, 4), dtype=np.int64), requires_grad=False))


if __name__ == "__main__":
    unittest.main()


