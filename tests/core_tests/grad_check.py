import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
import unittest

from src.core.losses import BCE, CrossEntropy, MSE
from src.core.tensor import Tensor
from src.utils.backend import xp
from src.core.module import Module
from src.core.optim import Standard, AdamW



import torch
import torch.nn as nn
import torch.optim as optim

class GradCheckCrossEntropy(unittest.TestCase):
    def _pt_ce(self, logits, targets):
        logits_pt_flat = logits.view(-1, logits.shape[-1])       # (B*T, C)
        targets_pt_flat = targets.view(-1)                  # (B*T,)

        ce_pt = torch.nn.CrossEntropyLoss()
        loss_pt = ce_pt(logits_pt_flat, targets_pt_flat)
        loss_pt.backward()

        return loss_pt

    def _mine_ce(self, logits, targets):
        loss = CrossEntropy(logits, targets)
        loss.backward()

        return loss

    def test_cross_entropy(self):
        np.random.seed(42)
        batch_size = 4
        seq_len = 10
        num_classes = 5

        # Fake logits and targets
        logits_np = np.random.randn(batch_size, seq_len, num_classes).astype(np.float64)
        targets_np = np.random.randint(0, num_classes, size=(batch_size, seq_len))

        # PyTorch
        logits_pt = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        targets_pt = torch.tensor(targets_np, dtype=torch.int64)


        # Your custom Tensor setup
        logits = Tensor(logits_np, requires_grad=True)
        targets = Tensor(targets_np, requires_grad=False)  # still class indices

        loss_pt = self._pt_ce(logits_pt, targets_pt)
        loss_mine = self._mine_ce(logits, targets)


        # Compare loss
        self.assertTrue(np.allclose(loss_mine.data, loss_pt.item(), atol=1e-6))
        # Compare gradients
        self.assertTrue(np.allclose(logits.grad.data, logits_pt.grad.detach().numpy(), atol=1e-5))

    def test_log(self):
        x_np = np.array([[1, 2, 3], [4, 5, 6]])
        x_pt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

        x = Tensor(x_np, requires_grad=True)
        y = x.log().mean()  # Make it scalar for backward()
        y_pt = x_pt.log().mean()  # Make it scalar for backward()
        y.backward()
        y_pt.backward()
        self.assertTrue(np.allclose(x.grad.data, x_pt.grad.detach().numpy()))

    
    def test_linear(self):
        np.random.seed(42)
        batch_size = 4
        seq_len = 10
        in_features = 10
        out_features = 5
        
        x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float64)
        w_np = np.random.randn(in_features, out_features).astype(np.float64)
        b_np = np.random.randn(out_features).astype(np.float64)
        
        x_pt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        w_pt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)

        x = Tensor(x_np, requires_grad=True)
        w = Tensor(w_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        
        y = x @ w + b
        y_pt = x_pt @ w_pt + b_pt

        loss_y = y.mean()
        loss_y_pt = y_pt.mean()

        loss_y.backward()
        loss_y_pt.backward()

        # Debug: print shapes and 

        self.assertTrue(np.allclose(x.grad.data, x_pt.grad.detach().numpy(), atol=1e-8))
        self.assertTrue(np.allclose(w.grad.data, w_pt.grad.detach().numpy(), atol=1e-8))
        self.assertTrue(np.allclose(b.grad.data, b_pt.grad.detach().numpy(), atol=1e-8))

    def test_bce(self):
        np.random.seed(42)
        batch_size = 8

        logits_np = np.random.randn(batch_size, 1).astype(np.float64)
        targets_np = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float64)

        # --- PyTorch ---
        logits_pt = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        targets_pt = torch.tensor(targets_np, dtype=torch.float64)

        probs_pt = torch.sigmoid(logits_pt)
        probs_pt.retain_grad()  # 🌟 This is the fix
        loss_pt = torch.nn.functional.binary_cross_entropy(probs_pt, targets_pt)
        loss_pt.backward()

        # --- Your framework ---
        logits = Tensor(logits_np, requires_grad=True)
        targets = Tensor(targets_np, requires_grad=False)

        loss = BCE(logits, targets)
        loss.backward()

        self.assertTrue(np.allclose(loss.data, loss_pt.item(), atol=1e-6))
        self.assertTrue(np.allclose(logits.grad.data, logits_pt.grad.detach().numpy(), atol=1e-6))

    def test(self):
        # shape: (B, S, V) = (2, 2, 3)
        logits_arr = xp.array(
            [
                [[2.0, 1.0, 0.1], [0.5, 2.1, 1.3]],
                [[1.2, 0.7, 3.3], [0.3, 0.1, 0.9]],
            ]
        )
        targets_arr = xp.array([[0, 2], [2, 1]])  # shape (B, S)

        logits  = Tensor(logits_arr,  requires_grad=True)
        targets = Tensor(targets_arr, requires_grad=False)

        loss = CrossEntropy(logits, targets)

        # ----- value -----
        max_logits   = logits_arr.max(axis=-1, keepdims=True)
        shifted      = logits_arr - max_logits
        logsumexp    = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
        log_softmax  = shifted - logsumexp
        batch_idx, seq_idx = xp.indices(targets_arr.shape)
        expected_loss = -log_softmax[batch_idx, seq_idx, targets_arr].mean()
        self.assertTrue(np.allclose(loss.data, expected_loss))

        # ----- gradient -----
        loss.backward()
        softmax = xp.exp(shifted) / xp.exp(shifted).sum(axis=-1, keepdims=True)
        grad_expected = softmax.copy()
        grad_expected[batch_idx, seq_idx, targets_arr] -= 1.0
        grad_expected /= (targets_arr.shape[0] * targets_arr.shape[1])  # B * S
        self.assertTrue(np.allclose(logits.grad.data, grad_expected))



    def test_learn_3d(self):
        # Set seeds for reproducible results
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None

        class Net(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = self.linear(10, 32)
                self.fc2 = self.linear(32, 32)
                self.fc3 = self.linear(32, 32)
                # Fix: layer_norm expects axis index, not dimension size
                self.ln1 = self.layer_norm(axis=-1)  # Normalize over the last dimension
                self.ln2 = self.layer_norm(axis=-1)
                self.ln3 = self.layer_norm(axis=-1)
                self.fc4 = self.linear(32, 5)  # say 5 classes for variety

            def forward(self, x):
                # x shape: (B, S, input_dim)
                B, S, _ = x.data.shape
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                # x = self.ln1(x)
                x = self.relu(x)
                # x = self.ln2(x)
                x = self.fc3(x)
                x = self.relu(x)
                # x = self.ln3(x)
                x = self.fc4(x)                # logits shape: (B, S, 5)
                return x

        B, S, input_dim, num_classes = 128, 10, 10, 5
        net = Net()
        x = Tensor(np.random.randn(B, S, input_dim))
        net._build(x.shape)

        y = Tensor(np.random.randint(0, num_classes, size=(B, S)))  # shape (B, S)

        optimizer = AdamW(net.parameters(), lr=0.01)
        print("-" * 100)
        print("Custom Implementation:")
        for i in range(100):
            optimizer.zero_grad()
            y_hat = net.forward(x)              # shape (B, S, V)
            loss = CrossEntropy(y_hat, y)
            loss.backward()
            optimizer.step()
            if i % 10 == 0: 
                print(f"Step {i}: {loss.data}")
        print("-" * 100)

    def test_learn_3d_pytorch(self):
        # Set seeds for reproducible results (same as above)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
        
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 32)
                self.fc2 = nn.Linear(32, 32)
                self.fc3 = nn.Linear(32, 32)
                self.fc4 = nn.Linear(32, 5)  # 5 classes

            def forward(self, x):
                B, S, _ = x.shape
                x = x.view(B * S, -1)          # flatten sequence dim
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)                # logits (B*S, 5)
                x = x.view(B, S, -1)           # reshape back (B, S, classes)
                return x

        B, S, input_dim, num_classes = 128, 10, 10, 5
        net = Net()
        x = torch.randn(B, S, input_dim)
        y = torch.randint(0, num_classes, (B, S))

        optimizer = optim.AdamW(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        print("-" * 100)
        print("PyTorch Implementation:")
        for i in range(100):
            optimizer.zero_grad()
            y_hat = net(x)               # (B, S, classes)
            # reshape to (B*S, classes) and targets to (B*S)
            loss = criterion(y_hat.view(B * S, -1), y.view(-1))
            loss.backward()
            optimizer.step()
            if i % 10 == 0: 
                print(f"Step {i}: {loss.item()}")
        print("-" * 100)

    
    # def test_pt(self):
    #     class Net(nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.fc1 = nn.Linear(10, 32)
    #             self.fc2 = nn.Linear(32, 1)
    #             self.relu = nn.ReLU()
    #             self.sigmoid = nn.Sigmoid()

    #         def forward(self, x):
    #             x = self.fc1(x)
    #             x = self.relu(x)
    #             x = self.fc2(x)
    #             return x

    #     # Instantiate the model
    #     net = Net()

    #     # Print parameter shapes just like your custom one
    #     for param in net.parameters():
    #         print(param.shape)

    #     # Data — still random (you might wanna change y later to test real learning)
    #     x = torch.randn(128, 10)
    #     y = torch.randint(0, 2, size=(128, 1))

    #     # Optimizer
    #     optimizer = optim.AdamW(net.parameters(), lr=0.1)

    #     # Loss function
    #     loss_fn = nn.BCELoss()

    #     print("-" * 100)
    #     for i in range(1000):
    #         y_hat = net(x)
    #         loss = loss_fn(y_hat, y)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         print(loss.item())
    #     print("-" * 100)



if __name__ == "__main__":
    unittest.main()