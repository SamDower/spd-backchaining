import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import einops
import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.experiments.resid_mlp.configs import (
    ResidMLPModelConfig,
    ResidMLPTaskConfig,
    ResidMLPTrainConfig,
)
from spd.interfaces import LoadableModule, RunInfo
from spd.log import logger
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.module_utils import init_param_
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


@dataclass
class ResidMLPTargetRunInfo(RunInfo[ResidMLPTrainConfig]):
    """Run info from training a ResidualMLPModel."""

    label_coeffs: Float[Tensor, " n_features"]

    @override
    @classmethod
    def from_path(cls, path: ModelPath) -> "ResidMLPTargetRunInfo":
        """Load the run info from a wandb run or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                resid_mlp_train_config_path = run_dir / "resid_mlp_train_config.yaml"
                label_coeffs_path = run_dir / "label_coeffs.json"
                checkpoint_path = run_dir / "resid_mlp.pth"
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                resid_mlp_train_config_path, label_coeffs_path, checkpoint_path = (
                    ResidMLP._download_wandb_files(wandb_path)
                )
        else:
            # `path` should be a local path to a checkpoint
            resid_mlp_train_config_path = Path(path).parent / "resid_mlp_train_config.yaml"
            label_coeffs_path = Path(path).parent / "label_coeffs.json"
            checkpoint_path = Path(path)

        with open(resid_mlp_train_config_path) as f:
            resid_mlp_train_config_dict = yaml.safe_load(f)

        with open(label_coeffs_path) as f:
            label_coeffs = torch.tensor(json.load(f))

        resid_mlp_train_config = ResidMLPTrainConfig(**resid_mlp_train_config_dict)
        return cls(
            checkpoint_path=checkpoint_path,
            config=resid_mlp_train_config,
            label_coeffs=label_coeffs,
        )


    
class SparseMLP(nn.Module):
    def __init__(self, input_dim=16, hidden1=20, hidden2=20, output_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1, bias=False)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc3 = nn.Linear(hidden2, output_dim, bias=False)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out