# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import operator
import os
import sys

import torch
from lightning_fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import smlmshot
from smlmshot import utils

from .calibration import calibrate
from .validation import validate


def train(
    fabric: Fabric,
    model: nn.Module,
    training_module: nn.Module,
    opt: Optimizer,
    scheduler: LRScheduler,
    dl_train: DataLoader,
    dl_val: DataLoader,
    dl_calib: DataLoader,
    watched_metric: str,
    watched_metric_strategy: str,
    log_dir: str,
    cfg_str: str,
    begin_epoch: int = 0,
    begin_step: int = 0,
    n_epochs: int = -1,
    n_accum_steps: int = 1,
    patience: int = -1,
    enable_divergence_detection: bool = True,
    enable_profiling: bool = False,
):
    """Train a model for n epochs."""
    # init
    best_epoch = begin_epoch
    watched_metric_strategy = utils.format.format_string(watched_metric_strategy)
    if watched_metric_strategy not in ["min", "max"]:
        raise ValueError(
            "watched_metric_strategy must be 'min' or 'max',"
            f"found {watched_metric_strategy}."
        )
    best_metric = float("-inf") if watched_metric_strategy == "max" else float("inf")
    comp_operator = operator.ge if watched_metric_strategy == "max" else operator.le

    logger = None  # delay initialization
    loss_history = []
    n_epochs = n_epochs if n_epochs >= 0 else sys.maxsize**10
    path_last_ckpt, path_best_ckpt = utils.checkpoint.get_ckpts_path(log_dir)
    step = begin_step

    for epoch in range(begin_epoch, n_epochs):
        metrics = {}
        # train
        train_metrics, step = train_one_epoch(
            fabric=fabric,
            dl=dl_train,
            n_accum_steps=n_accum_steps,
            opt=opt,
            scheduler=scheduler,
            step=step,
            training_module=training_module,
            enable_profiling=enable_profiling,
        )
        metrics = metrics | train_metrics

        # calibrate
        calibrate(
            fabric=fabric, model=model, dl=dl_calib, watched_metric=watched_metric
        )
        metrics["threshold"] = model.threshold

        # validate
        metrics = metrics | validate(fabric=fabric, model=model, dl=dl_val)

        # log
        if fabric.is_global_zero:
            logs = {"epoch": epoch, "step": step} | metrics
            if logger is None:
                os.makedirs(log_dir, exist_ok=True)
                utils.logs.write_file(cfg_str, filename="config.yaml", log_dir=log_dir)
                logger = SummaryWriter(log_dir)
            for key, value in logs.items():
                if isinstance(value, Tensor) and value.nelement() > 1:
                    logger.add_tensor(tag=key, tensor=value, global_step=step)
                elif isinstance(value, Figure):
                    logger.add_figure(tag=key, figure=value, global_step=step)
                else:
                    logger.add_scalar(tag=key, scalar_value=value, global_step=step)
            print(utils.format.format_metrics(logs))

        # save
        utils.checkpoint.save_training(
            fabric=fabric,
            path=path_last_ckpt,
            epoch=epoch,
            step=step,
            optimizer_state_dict=opt.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            training_module_state_dict=training_module.state_dict(),
        )
        if comp_operator(metrics[watched_metric], best_metric):
            best_epoch = epoch
            best_metric = metrics[watched_metric]
            utils.checkpoint.save_training(
                fabric=fabric,
                path=path_best_ckpt,
                epoch=epoch,
                step=step,
                optimizer_state_dict=opt.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                training_module_state_dict=training_module.state_dict(),
            )

        # early stopping
        if patience > 0 and epoch > best_epoch + patience:
            if fabric.is_global_zero:
                print(f"Exceeded patience of {patience} epochs; early stopping.")
            break

        # divergence
        if enable_divergence_detection and len(dl_train) != 0:
            loss_history.append(metrics["loss"])
            if utils.divergence.detect_divergence(loss_history):
                if fabric.is_global_zero:
                    print("Divergence detected. Reverting to previous best weights.")
                utils.checkpoint.load_training(
                    fabric=fabric,
                    ckpt_path=path_best_ckpt,
                    optimizer=opt,
                    scheduler=scheduler,
                    training_module=training_module,
                )
                loss_history = []

    return best_metric


@torch.enable_grad()
def train_one_epoch(
    fabric: Fabric,
    dl: DataLoader,
    n_accum_steps: int,
    opt: Optimizer,
    scheduler: LRScheduler,
    step: int,
    training_module: nn.Module,
    enable_profiling: bool,
) -> tuple[dict, int]:
    """Train a model for one epoch."""
    if len(dl) == 0:
        return {"loss": torch.tensor(torch.nan, device=fabric.device)}, step

    training_module.train()
    opt.zero_grad()
    d = 1.0  # prodigy's multiplicative constant

    avg_metrics = smlmshot.metrics.MeanDictMetric(device=fabric.device)

    n_steps = len(dl) // n_accum_steps
    n_steps += 1 if len(dl) % n_accum_steps > 0 else 0
    with tqdm(total=n_steps, leave=False, disable=not fabric.is_global_zero) as pbar:
        with utils.profiler.Profiler(disable=not enable_profiling) as prof:
            for i, batch in enumerate(dl):
                # +1 so we start accumulating at the first step
                is_accumulating = (i + 1) % n_accum_steps != 0

                with fabric.no_backward_sync(training_module, enabled=is_accumulating):
                    metrics = training_module(batch)
                    fabric.backward(metrics["loss"] / n_accum_steps)

                if "d" in opt.param_groups[0]:
                    d = opt.param_groups[0]["d"]
                metrics["lr"] = d * scheduler.get_last_lr()[0]
                avg_metrics.update(metrics)
                pbar.desc = utils.format.format_metrics({"step": step} | metrics)

                if not is_accumulating:
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    step += 1
                    pbar.update(1)
                    prof.step()

    avg_metrics = avg_metrics.compute()
    return avg_metrics, step
