import os
import sys
import unittest
import tempfile
import shutil
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from src.models.transformer import Transformer, Embedding
from src.models.torch_transformer import TransformerBlock
from src.core.tensor import Tensor
from src.core.optim import SGD, AdamW
from src.core.losses import CrossEntropyWithLogits
from core.nn import Module
from src.utils.backend import xp, set_seed


class TestAdamW(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_adamw_optimizer(self):
        """
        Make sure that a few optimisation steps with our AdamW produce the
        exact same weights as torch.optim.AdamW given the same
        hyper-parameters and random seed.
        """
        set_seed(42)

        # deterministic data + initial weights
        x_data = xp.random.randn(4, 8).astype(xp.float32)
        w_data = xp.random.randn(8, 8).astype(xp.float32)

        # ----- custom framework -----
        x_my = Tensor(x_data, requires_grad=False)
        w_my = Tensor(w_data.copy(), requires_grad=True)
        optim_my = AdamW(
            params={"w": w_my},
            lr=1e-3,
            weight_decay=0.01,        # non-zero weight-decay
            beta_1=0.9,
            beta_2=0.95,
            clip_norm=1.0,
        )

        # ----- PyTorch -----
        x_pt = torch.tensor(x_data, requires_grad=False)
        w_pt = torch.tensor(w_data.copy(), requires_grad=True)
        optim_pt = torch.optim.AdamW(
            [w_pt],
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,        # match custom optimiser
        )

        # run a handful of optimisation steps
        for _ in range(100):
            # forward pass
            y_my = (x_my @ w_my).mean()
            y_pt = (x_pt @ w_pt).mean()

            # zero-grad
            optim_my.zero_grad()
            optim_pt.zero_grad()

            # backward
            y_my.backward()
            y_pt.backward()

            # update
            optim_my.step()
            optim_pt.step()

            self.assert_close(w_my.data, w_pt.detach().numpy(), atol=1e-4)

# --------------------------------------------------------------------------- #
# Additional robustness tests                                                 #
# --------------------------------------------------------------------------- #

    def test_multiple_parameters(self):
        """
        Two trainable tensors (weight + bias) are updated simultaneously.
        Every optimisation step must match PyTorch exactly.
        """
        set_seed(123)

        # synthetic data ----------------------------------------------------------------
        x_data = xp.random.randn(6, 10).astype(xp.float32)
        w_data = xp.random.randn(10, 4).astype(xp.float32)
        b_data = xp.random.randn(4).astype(xp.float32)

        # ----- our framework -----------------------------------------------------------
        x_my  = Tensor(x_data, requires_grad=False)
        w_my  = Tensor(w_data.copy(), requires_grad=True)
        b_my  = Tensor(b_data.copy(), requires_grad=True)
        optim_my = AdamW(
            params={"w": w_my, "b": b_my},
            lr=2e-4,
            weight_decay=0.02,
            beta_1=0.9,
            beta_2=0.98,
            clip_norm=5.0,
        )

        # ----- torch -------------------------------------------------------------------
        x_pt  = torch.tensor(x_data, requires_grad=False)
        w_pt  = torch.tensor(w_data.copy(), requires_grad=True)
        b_pt  = torch.tensor(b_data.copy(), requires_grad=True)
        optim_pt = torch.optim.AdamW(
            [w_pt, b_pt],
            lr=2e-4,
            betas=(0.9, 0.98),
            weight_decay=0.02,
        )

        # training loop -----------------------------------------------------------------
        for _ in range(75):
            # forward pass: simple linear model
            y_my = (x_my @ w_my + b_my).mean()
            y_pt = (x_pt @ w_pt + b_pt).mean()

            # zero-grad
            optim_my.zero_grad()
            optim_pt.zero_grad()

            # backward
            y_my.backward()
            y_pt.backward()

            # update
            optim_my.step()
            optim_pt.step()

            # parameters must match
            self.assert_close(w_my.data, w_pt.detach().numpy(), atol=1e-4)
            self.assert_close(b_my.data, b_pt.detach().numpy(), atol=1e-4)

    def test_gradient_clipping(self):
        """
        A small clip_norm should lead to a smaller parameter update than a
        very large clip_norm (effectively 'no-clip').
        """
        set_seed(7)

        # exaggerated inputs to guarantee a large gradient
        x_data = (xp.random.randn(128, 64) * 1e8).astype(xp.float32)
        w_data = xp.random.randn(64, 64).astype(xp.float32)

        # tensors -----------------------------------------------------------------------
        x_my = Tensor(x_data, requires_grad=False)
        w_clip = Tensor(w_data.copy(), requires_grad=True)
        w_noclip = Tensor(w_data.copy(), requires_grad=True)

        # optimisers --------------------------------------------------------------------
        optim_clip = AdamW(params={"w": w_clip},
                           lr=1e-3, beta_1=0.9, beta_2=0.95,
                           weight_decay=0.0, clip_norm=1.0)
        optim_noclip = AdamW(params={"w": w_noclip},
                             lr=1e-3, beta_1=0.9, beta_2=0.95,
                             weight_decay=0.0, clip_norm=1e9)  # effectively no clipping

        # single optimisation step ------------------------------------------------------
        y_clip = (x_my @ w_clip).sum()
        y_noclip = (x_my @ w_noclip).sum()

        optim_clip.zero_grad()
        optim_noclip.zero_grad()

        y_clip.backward()
        y_noclip.backward()


        total_norm_clip = optim_clip._get_total_norm()
        total_norm_noclip = optim_noclip._get_total_norm()
        print(f"total_norm_clip: {total_norm_clip}")
        print(f"total_norm_noclip: {total_norm_noclip}")

        optim_clip.step()
        optim_noclip.step()

        # update magnitude with clipping must be smaller
        delta_clip   = xp.linalg.norm(w_clip.data - w_data)
        delta_noclip = xp.linalg.norm(w_noclip.data - w_data)
        self.assertTrue(delta_clip < delta_noclip * 0.9)  # at least 10 % smaller

    # def test_state_save_and_load(self):
    #     """
    #     After saving and reloading the optimiser state the training trajectory
    #     must continue exactly from the same point.
    #     """
    #     set_seed(99)

    #     # data & parameters -------------------------------------------------------------
    #     x_data = xp.random.randn(8, 8).astype(xp.float32)
    #     w_data = xp.random.randn(8, 8).astype(xp.float32)

    #     x1 = Tensor(x_data, requires_grad=False)
    #     w1 = Tensor(w_data.copy(), requires_grad=True)
    #     optim1 = AdamW(params={"w": w1},
    #                    lr=1e-3, beta_1=0.9, beta_2=0.95,
    #                    weight_decay=0.01, clip_norm=1.0)

    #     # train for a few steps ---------------------------------------------------------
    #     for _ in range(30):
    #         y = (x1 @ w1).mean()
    #         optim1.zero_grad()
    #         y.backward()
    #         optim1.step()

    #     # save optimiser state ----------------------------------------------------------
    #     tmp_dir = tempfile.mkdtemp()
    #     try:
    #         optim1._save_state(tmp_dir)

    #         # create a fresh optimiser + param tensor -----------------------------------
    #         x2 = Tensor(x_data, requires_grad=False)
    #         w2 = Tensor(w_data.copy(), requires_grad=True)
    #         optim2 = AdamW(params={"w": w2},
    #                        lr=1e-3, beta_1=0.9, beta_2=0.95,
    #                        weight_decay=0.01, clip_norm=1.0)
    #         optim2._load_state(tmp_dir)

    #         # weights and moments must match exactly after reload
    #         self.assert_close(w1.data, w2.data, atol=1e-6)
    #         self.assert_close(optim1.params['w']['m_t'], optim2.params['w']['m_t'], atol=1e-6)
    #         self.assert_close(optim1.params['w']['v_t'], optim2.params['w']['v_t'], atol=1e-6)

    #         # continue training for a few more steps ------------------------------------
    #         for _ in range(10):
    #             y1 = (x1 @ w1).mean()
    #             y2 = (x2 @ w2).mean()

    #             optim1.zero_grad()
    #             optim2.zero_grad()

    #             y1.backward()
    #             y2.backward()

    #             optim1.step()
    #             optim2.step()

    #             self.assert_close(w1.data, w2.data, atol=1e-6)
    #     finally:
    #         shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()