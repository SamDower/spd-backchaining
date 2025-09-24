"""Residual MLP decomposition script."""

import json
from pathlib import Path

import fire
import wandb

from spd.configs import Config
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import (
    ResidMLP,
    ResidMLPTargetRunInfo,
    ResidMLPWithModules,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
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


def clone_resid_mlp_with_modules(model: ResidMLP):
    """Convert an existing ResidMLP with Parameter weights into one with nn.Linear modules."""
    new_model = ResidMLPWithModules(model.config)

    # Copy weights into Linear layers
    with torch.no_grad():
        new_model.W_E.weight.copy_(model.W_E.T)  # because einsum used (n_features, d_embed)
        new_model.W_U.weight.copy_(model.W_U.T)  # because einsum used (d_embed, n_features)

    # Copy over MLP layers
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        new_layer.load_state_dict(old_layer.state_dict())

    return new_model


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

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = ResidMLP.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    target_model = clone_resid_mlp_with_modules(target_model)
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
        train_config=target_run_info.config,
        task_name=config.task_config.task_name,
    )
    save_file(target_run_info.label_coeffs.detach().cpu().tolist(), out_dir / "label_coeffs.json")
    if config.wandb_project:
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir, policy="now")

    synced_inputs = target_run_info.config.synced_inputs
    dataset = ResidMLPDataset(
        n_features=target_model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=synced_inputs,
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
