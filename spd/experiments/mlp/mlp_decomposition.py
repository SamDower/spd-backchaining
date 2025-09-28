"""Residual MLP decomposition script."""

import json
from pathlib import Path

import fire
import wandb

from spd.configs import Config
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.mlp.models import (
    SparseMLP
)
from spd.experiments.mlp.mlp_dataset import ToySparseArithmeticDataset
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    load_config,
    save_pre_run_info,
    set_seed,
)
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb
import torch



def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        tags = ["resid_mlp"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    out_dir = get_output_dir(use_wandb_id=config.wandb_project is not None)

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, ResidMLPTaskConfig)


    target_model = SparseMLP()
    # load target model from .pth file
    target_model.load_state_dict(torch.load("spd/experiments/mlp/arithmetic_model.pth", map_location=device))
    target_model = target_model.to(device)
    target_model.eval()


    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        if config.wandb_run_name:
            wandb.run.name = config.wandb_run_name

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=None,
        task_name=config.task_config.task_name,
    )
    #save_file(target_run_info.label_coeffs.detach().cpu().tolist(), out_dir / "label_coeffs.json")
    if config.wandb_project:
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir, policy="now")

    dataset = ToySparseArithmeticDataset(
        device=device,
    )

    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    # TODO: Below not needed when TMS supports config.n_eval_steps
    assert config.n_eval_steps is not None, "n_eval_steps must be set"
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
