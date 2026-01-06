# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from smlmshot import engine, utils


@hydra.main(version_base=None, config_path="../configs", config_name="shot")
@torch.no_grad()
def test(cfg: DictConfig):
    """Perform a validation pass for a model with loaded weights."""
    cfgr = cfg.runtime
    utils.torch.initialize_torch(detect_anomaly=cfgr.detect_anomaly)
    fabric = utils.fabric.initialize_fabric(
        cfgr.seed, precision=cfgr.precision, devices=cfgr.devices
    )
    cfg = utils.config.initialize_config(cfg)

    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # load data
    ds_test = instantiate(cfg.ds_test, _convert_="partial")
    dl_test = utils.dataloader.build_dl(
        ds_test,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )
    dl_test = fabric.setup_dataloaders(dl_test)

    # model
    with fabric.init_module(empty_init="weights_path" in cfgr):
        model = instantiate(cfg.model)
    if fabric.is_global_zero:
        utils.model.present_model(model)
    model = fabric.setup(model)

    if "weights_path" in cfgr:
        utils.checkpoint.load_weights(
            fabric=fabric, ckpt_path=cfgr.weights_path, model=model
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {cfgr.weights_path}")
    else:
        if fabric.is_global_zero:
            print("WARNING: no weights loaded.")

    # evaluation
    metrics = engine.validate(fabric=fabric, model=model, dl=dl_test)

    # log
    if fabric.is_global_zero:
        print(utils.format.format_metrics(metrics))
    return metrics


if __name__ == "__main__":
    test()
