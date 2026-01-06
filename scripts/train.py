# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tempfile import TemporaryDirectory

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from smlmshot import datasets, engine, trainers, utils


@hydra.main(version_base=None, config_path="../configs")
@utils.oom.handle_oom()
def train(cfg: DictConfig) -> float:
    """Train a configuration."""
    # init
    cfgr = cfg.runtime
    utils.torch.initialize_torch(detect_anomaly=cfgr.detect_anomaly)
    fabric = utils.fabric.initialize_fabric(
        cfgr.seed, precision=cfgr.precision, devices=cfgr.devices
    )
    cfg = utils.config.initialize_config(cfg)

    cfg = utils.config.add_git_commit_hash(cfg)
    cfg = utils.config.add_eff_batch_size(cfg, world_size=fabric.world_size)
    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg, sort_keys=True))

    # psf and camera
    psf = instantiate(cfg.psf)
    camera = instantiate(cfg.camera)
    seed = utils.random.derive_new_seed(cfgr.seed)

    # create synthetic datasets
    seed = utils.random.derive_new_seed(seed)
    ds_train = datasets.MultipleSyntheticScenesDataset.init_from_config(
        length=cfg.ds_train.length, seed=seed, psf=psf, cfg=cfg
    )
    dl_train = utils.dataloader.build_dl(
        ds_train,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )

    seed = utils.random.derive_new_seed(seed)
    ds_val = datasets.SimulateSceneDataset(
        ds=datasets.OneSyntheticSceneDataset.init_from_config(
            length=cfg.ds_val.length, seed=seed, psf=psf, cfg=cfg
        ),
        camera=camera,
        jitter_std=cfg.ds_val.jitter_std,
        psf=psf,
        seed=seed,
    )
    dl_val = utils.dataloader.build_dl(
        ds_val,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )

    seed = utils.random.derive_new_seed(seed)
    ds_calib = datasets.SimulateSceneDataset(
        ds=datasets.OneSyntheticSceneDataset.init_from_config(
            length=cfg.ds_val.length, seed=seed, psf=psf, cfg=cfg
        ),
        camera=camera,
        jitter_std=cfg.ds_val.jitter_std,
        psf=psf,
        seed=seed,
    )
    dl_calib = utils.dataloader.build_dl(
        ds_calib,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )

    cfg = utils.config.add_total_steps(cfg, step_per_epoch=len(ds_train))
    dl_train, dl_val, dl_calib = fabric.setup_dataloaders(dl_train, dl_val, dl_calib)

    # model
    with fabric.init_module(empty_init="ckpt_path" in cfg):
        model = instantiate(cfg.model)
        seed = utils.random.derive_new_seed(seed)
        training_module = trainers.OptimalTransportTrainer.init_from_config(
            camera=camera, model=model, psf=psf, seed=seed, cfg=cfg
        )
        opt = instantiate(cfg.optimizer, params=training_module.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=opt)
    if fabric.is_global_zero:
        utils.model.present_model(model)
    if cfgr.compile:
        training_module.compile()
        if fabric.is_global_zero:
            print("Compilation enabled.")
    training_module, opt = fabric.setup(training_module, opt)

    epoch, step = 0, 0
    if "ckpt_path" in cfgr:
        epoch, step = utils.checkpoint.load_training(
            fabric=fabric,
            ckpt_path=cfgr.ckpt_path,
            optimizer=opt,
            scheduler=scheduler,
            training_module=training_module,
        )
        epoch += 1
        if fabric.is_global_zero:
            print(f"Training checkpoint loaded from {cfgr.ckpt_path}")
    elif "weights_path" in cfgr:
        utils.checkpoint.load_weights(
            fabric=fabric, ckpt_path=cfgr.weights_path, model=model
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {cfgr.weights_path}")

    # logs
    if "log_dir" not in cfgr:
        log_dir = utils.logs.get_log_dir(cfgr.name) if fabric.is_global_zero else None
        cfgr.log_dir = fabric.broadcast(log_dir, src=0)
    if cfgr.log_dir == "/tmp":
        log_dir = None
        if fabric.is_global_zero:
            tempdir = TemporaryDirectory()
            log_dir = tempdir.name
        cfgr.log_dir = fabric.broadcast(log_dir, src=0)
    elif os.path.exists(cfgr.log_dir):
        raise ValueError(f"log_dir '{cfgr.log_dir}' exists.")
    if fabric.is_global_zero:
        print(f"Log directory is {cfgr.log_dir}")

    # training options
    if not cfgr.detect_divergence and fabric.is_global_zero:
        print("Warning: divergence detection is disabled.")
    if cfgr.profile and fabric.is_global_zero:
        print("Profiling is enabled.")

    # train
    best_metric = engine.train(
        fabric=fabric,
        opt=opt,
        scheduler=scheduler,
        model=model,
        training_module=training_module,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_calib=dl_calib,
        log_dir=cfgr.log_dir,
        cfg_str=utils.config.dump_config(cfg),
        begin_epoch=epoch,
        begin_step=step,
        n_epochs=cfgr.n_epochs,
        n_accum_steps=cfgr.n_accum_steps,
        patience=cfgr.patience,
        watched_metric=cfgr.watched_metric,
        watched_metric_strategy=cfgr.watched_metric_strategy,
        enable_divergence_detection=cfgr.detect_divergence,
        enable_profiling=cfgr.profile,
    )
    return best_metric


if __name__ == "__main__":
    train()
