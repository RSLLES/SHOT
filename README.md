# SHOT: Optimal transport unlocks end-to-end learning for single-molecule localization
[ 📜 [`Paper`](https://arxiv.org/abs/2512.10683)] [ 📕 [`BibTeX`](#Citing)]

This repository contains the official implementation of the paper **Optimal transport unlocks end-to-end learning for single-molecule localization**.
It provides tools to train models, process new datasets, and assess performance on synthetic datasets.
We have named our method **SHOT**, an acronym of SMLM at High-density with Optimal Transport.

## Installation

This project has been tested and is known to work for **Python >=3.11**.
You can create a virtual environment with the tool of your choice and install the project using `pip`.
Here is a minimal working example with Python's `venv` and `pip` modules:

```bash
git pull https://github.com/RSLLES/SHOT.git
cd SHOT
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

*Edit 18-12-2025*: we recommend [uv](https://docs.astral.sh/uv/), a modern and fast Python project manager:
```bash
git pull https://github.com/RSLLES/SHOT.git
cd SHOT
uv sync
source .venv/bin/activate
```

## Configuration

This project uses `hydra` for configuration management and Lightning's `fabric` module to handle multi-GPU training.  
Scripts are located in the `scripts/` subdirectory and `hydra` configuration files are available in the `config/` subdirectory.
There is no default configuration; the user must provide a configuration name with the flag `--config-name` at runtime.

Two examples of configurations are provided: `configs/shot_EPFL_MT0N1HDAS.yaml` that is configured to work with the `MT0N1HDAS` dataset of EPFL's 2016 challenge (Sage et al., 2019), and `configs/shot_LiteLoc_NPC_U2OS.yaml` that processes the LiteLoc dataset (Fei et al., 2025).

```yaml
defaults:
  - psf: EPFL_AS
  - camera: EPFL
  - fluorescence: EPFL_N1HD
  - dataset@ds_test: EPFL_MT0N1HDAS
  - runtime: default
  - optimizer: adamw
  - scheduler: cosineannealing
  - model: shot
  - writer: csv
  - _self_
  - hydra_defaults

optimizer:
  lr: 0.0006

ds_train:
  length: ${eval:2**17}
  jitter_std: 0.03

ds_val:
  length: 256
  jitter_std: 0.03

trainer:
  reg: 1e-4
  n_sinkhorn_iters: 20

runtime:
  name: shot_EPFL_MT0N1HDAS

writer:
  filepath: "exports/shot_EPFL_MT0N1HDAS.csv"
```

### Model
We provide two models:
- **SHOT**: the original SHOT model of our paper, including both our optimal transport loss function and our recursive architecture.
- **SHOT_Lite**: for RAM-limited projects, disabling our recursive architecture yields small performance degradation with substantially faster training time and less RAM usage (differentiation through our simulator is extensive), as detailed in our ablation study.

To use **SHOT_Lite**, replace `model: shot` by `model: shot_lite`.

### Data
Datasets and PSFs paths are typically located in the `data/` subdirectory.

**Fluorescence parameters**:
Like DECODE and LiteLoc, our method needs some initial estimations of various constants to simulate similar training data alike the processed acquisition.
They are located in the `configs/fluorescence/` subdirectory.
Example from `configs/fluorescence/LiteLoc_NPC_U2OS.yaml`:

```yaml
bg_photon_mean: 200.0
bg_photon_std: 0.0
n_acts_per_frame: 100
photon_flux_mean: 12e3
photon_flux_std: 0.0
time_bleach: 1.5
time_off: 3.0
time_on: 2.5
```

**PSF**: 
PSF are assumed to be pre-calibrated using SMAP. They are located in the `configs/psf` subdirectory.
Example from `configs/psf/LiteLoc_NPC_U2OS.yaml`:

```yaml
_target_: smlmshot.simulation.psfs.CSplinesPSF.init_from_mat
filepath: "data/LiteLoc_NPC_U2OS/Astigmatism_Tetraspeck_beads_2um_50nm_256_1_MMStack_Pos0.ome_3dcal.mat"
pixel_size: 110.0
```

SMAP is available on Github: https://github.com/jries/SMAP .
Refer to SMAP official manual (https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf, section 5.4.1) for additionnal details about PSF calibration with SMAP.

**Camera**:
Parameters of the camera are required. They can usually be found in the manufacturer's documentation.
Example from `configs/camera/LiteLoc_NPC_U2OS.yaml`:

```yaml
_target_: smlmshot.simulation.Camera
adu_baseline: 100
em_gain: 1.0
inv_e_adu: "${eval: 1.0 / 0.7471}"
quantum_efficiency: 0.95
readout_noise: 1.535
spurious_charge: 0.002
type: "sCMOS"
```

**Acquisition**: 
Input data are loaded via a PyTorch dataset. 
They are only used at inference time, not during training.
Convinient preprocessing functions are available in `smlmshot.utils.image`. 
Examples of configuration are available in `config/dataset/`;
here is the content of `configs/dataset/LiteLoc_NPC_U2OS.yaml`: 

```yaml
_target_: smlmshot.datasets.ImagesDataset
y:
  _target_: smlmshot.utils.images.standardize_background
  y:
    _target_: torch.cat
    dim: 0
    tensors:
      - _target_: smlmshot.utils.images.read_tiff
        filepath: data/LiteLoc_NPC_U2OS/NPC_U2OS_640_40mw_20ms_256_2_MMStack_Pos0.ome.tif
      - _target_: smlmshot.utils.images.read_tiff
        filepath: data/LiteLoc_NPC_U2OS/NPC_U2OS_640_40mw_20ms_256_2_MMStack_Pos0_1.ome.tif
window: ${runtime.n_frames}
```

Note that our method loads the entire acquisition into RAM: we did not develop an incremental loading tool, but we will be glad to accept any pool request.

*Edit 18-12-2025*: The webpage of the EPFL's SMLM hub seems to be down and the dataset is unfortunately unavailable.
Interested can contact us directly.

### Runtime

Config files under `configs/runtime` contain all variables related to the execution environment and training hyperparameters.
Example of `configs/runtime/default.yaml`:

```yaml
batch_size: 64
compile: true
detect_divergence: true
devices: "auto"
eps: 1e-12
n_accum_steps: 1
n_epochs: 100
n_frames: 3
n_pixels: 64
n_workers: 4
patience: -1
precision: "32-true"
seed: 0
watched_metric_strategy: "max"
watched_metric: "E_3D"
```

`compile` tells PyTorch wether to compile the model or not. 
`detect_divergence` enables divergence detection, which resets to the previous best checkpoint if the loss value suddenly increases.
`patience` enables early stopping if the watched metric does not improve for n consecutive epochs; setting it to –1 disables this functionality.
`precision` refers to the hardware precision, see https://lightning.ai/docs/fabric/stable/fundamentals/precision.html. 
`watched_metric` defines which validation metric is used to determine the best checkpoint, and `watched_metric_strategy` specifies if it should be maximized or minimized.

### Export

To export data, you need to specify a `writer`.
The `csv` writer exports a `.csv` in ThunderSTORM format that can be read with multiple SMLM software like SMAP, ThunderSTORM, GDSC, etc.
We also provide a `ash` writer that creates a 2D image using Average Shifted Histogram, see Scott (1985).
You need to specify the output file path with `++writer.filepath=path/to/output/file.(csv/png)`.

## Training

Start training with the `scripts/train.py` script. Don't forget to specify a configuration (e.g., `shot_LiteLoc_NPC_U2OS`):

```bash
python scripts/train.py --config-name=shot_LiteLoc_NPC_U2OS
```

We provide support for multi-gpu training (automatic with `fabric`) and multi-node training with the `multinode.sh` script (note: you may need to modify it as it relies on `uv`).

Output files and Tensorboard logs will be saved in `outputs/<runtime.name>/v<automatic_version_number>/`.
This destination directory logic can be completely overwritten by specifying `++runtime.log_dir=custom/path/to/nonexistent/dir`.
`last.tar` is the checkpoint of the last epoch, and `best.tar` is the best performing model so far according to the watched metric defined in the runtime configuration.

Model compilation is enabled by default; if your machine does not support it, turn it off with `++runtime.compile=false`.

If an OOM error occurs, the training script will automatically catch it and retry using half the batch size and twice the number of gradient accumulation steps.

Backpropagation through the simulator is memory- and compute-intensive. 
If your hardware does not support it, or if you wish to have faster training at the cost of slight performance degradation, 
set the model to `shot_lite` (`model=shot_lite`): this lighter variant includes our optimal transport loss but not the iterative architecture.
You may further tweak training by adjusting runtime parameters like `runtime.batch_size`, `runtime.n_accum_steps`, `ds_train.length`, and `runtime.n_epochs` if needed.

```bash
python scripts/train.py --config-name=shot_LiteLoc_NPC_U2OS model=shot_lite ++runtime.name=shot_lite_LiteLoc_NPC_U2OS ++runtime.batch_size=32 ++runtime.n_accum_steps=2 ++runtime.n_epochs=50
```

### Export

Use `scripts/export.py` to export a `.csv` in ThunderSTORM format. 
It can be read with multiple software like SMAP, ThunderSTORM, GDSC, ShareLoc, etc.

Example with `shot_EPFL_MT0N1HDAS`:
```bash
python scripts/export.py --config-name=shot_EPFL_MT0N1HDAS ++runtime.weights_path=outputs/shot_EPFL_MT0N1HDAS/v0/last.tar ++writer.filepath=exports/shot_EPFL_MT0N1HDAS.csv
```

Note: `scripts/export.py` only works with one gpu: you can either specify `++runtime.devices=1` or add `CUDA_VISIBLE_DEVICES=<GPU_ID>` for fine-grained control.

### Evaluation

If the dataset includes ground truth coordinates, i.e. is an instance of `smlmshot.datasets.ImagesAndActivationsDataset` like `configs/dataset/EPFL_MT0N1HDAS.yaml`, the script `scripts/test.py` allows you to compute the set of metrics used in the EPFL 2016 challenge.


Example with `shot_EPFL_MT0N1HDAS`:
```bash
python scripts/test.py --config-name=shot_EPFL_MT0N1HDAS ++runtime.weights_path=outputs/shot_EPFL_MT0N1HDAS/v0/last.tar
```

## Citing

If you find this work useful, please consider giving a star ⭐ and citation:

```
@article{seailles2025optimal,
  title={{Optimal transport unlocks end-to-end learning for single-molecule localization}},
  author={Seailles, Romain and Masson, Jean-Baptiste and Ponce, Jean and Mairal, Julien},
  journal={arXiv preprint arXiv:2512.10683},
  url={https://arxiv.org/abs/2512.10683},
  year={2025}
}
```

## Appendix

### Code convention
If you intend to read the code, some conventional names include:
- `xyz`: Tensor of shape (**, 3) containing 3D coordinates. 
- `n_photons`: Tensor of shape (**, n_frames) containing the number of photons emitted during each frame.
- `significant`: Boolean tensor saying if a fluorophore is significant in the frame of interest, i.e. is above the 25% threshold defined by Sage et al. (2019) in the EPFL Challenge.