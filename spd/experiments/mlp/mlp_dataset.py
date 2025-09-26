from typing import Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
import numpy as np

from spd.utils.data_utils import SparseFeatureDataset
from torch.utils.data import DataLoader, Dataset


class ToySparseArithmeticDataset(Dataset[
    tuple[Float[Tensor, "batch 16"], Float[Tensor, "batch 20"]]
]):
    def __init__(
        self,
        n_features: int = 16,
        n_tasks: int = 20,
        feature_probability: float = 0.125,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """
        Toy dataset with sparse 16-dim inputs and 20 arithmetic targets.

        Args:
            n_features: Number of input features (fixed at 16 here).
            n_tasks: Number of output targets (fixed at 20 here).
            feature_probability: Probability each feature is nonzero.
            device: Device to place the tensors on.
            seed: Random seed for reproducibility.
        """
        assert n_features == 16, "This dataset is hardcoded for 16 input features."
        assert n_tasks == 20, "This dataset is hardcoded for 20 targets."

        self.n_features = n_features
        self.n_tasks = n_tasks
        self.feature_probability = feature_probability
        self.device = device
        self.seed = seed

    def __len__(self) -> int:
        # Effectively infinite dataset
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch 16"], Float[Tensor, "batch 20"]]:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        n_features = self.n_features
        X = torch.zeros((batch_size, n_features), device=self.device)

        # Bernoulli mask for sparsity
        mask = torch.bernoulli(
            torch.full((batch_size, n_features), self.feature_probability, device=self.device)
        )
        values = torch.randn((batch_size, n_features), device=self.device)
        X = mask * values # type: ignore

        x = [X[:, i] for i in range(n_features)]

        # Build 20 tasks
        Y = torch.stack(
            [
                x[0],                                # t0
                x[1],                                # t1
                x[0] + x[1],                         # t2
                x[0] * x[2],                         # t3
                x[3],                                # t4
                x[3] - x[4],                         # t5
                x[4] * x[5],                         # t6
                torch.max(x[3], x[5]),               # t7
                x[6] * x[7],                         # t8
                x[6] + 2 * x[7],                     # t9
                x[8] + x[9] + x[10],                 # t10
                (x[8] + x[9]) ** 2,                  # t11
                x[10] * (x[8] - x[9]),               # t12
                x[11],                               # t13
                x[12] * x[13],                       # t14
                x[11] + x[12] + x[13],               # t15
                x[14] * (x[11] + x[12]),             # t16
                (x[0] + x[1]) * x[5],                # t17
                (x[3] - x[4]) + x[7],                # t18
                x[9] * x[12],                        # t19
            ],
            dim=1,
        )

        return X.to(self.device), Y.to(self.device)