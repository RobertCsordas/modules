# Codebase for inspecting modularity of neural networks

The official repository for our paper "Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks".

## Installation

This project requires Python 3 and PyTorch 1.6.

```bash
pip3 install -r requirements.txt
```

Create a Weights and Biases account and run 
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

For plotting, LaTeX is required (to avoid Type 3 fonts and to render symbols). Installation is OS specific.

## Usage

The code makes use of Weights and Biases for experiment tracking. In the ```sweeps``` directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run multiple configurations and seeds.

To reproduce our results, start a sweep for each of the YAML files in the ```sweeps``` directory. Run wandb agent for each of them in the main directory. This will run all the experiments, and they will be displayed on the W&B dashboard.

The task ```sweeps/dm_math/dm_math_polycollect.yaml``` needs two 16Gb GPUs to run. In any of the experiments won't fit on a single GPU, it is possible to run them on multile (data parallel) by specifying multiple devices in CUDA_VISIBLE_DEVICE.
### Re-creating plots from the paper

Edit config file ```paper/config.json```. Enter your project name in the field "wandb_project" (e.g. "username/modules").

Run the scripts in the ```paper``` directory. For example:

```bash
cd paper
./run_all.sh
```

The output will be generated in the ```paper/out/``` directory.

If you want to reproduce individual plots, it can be done by running individial python files in the ```paper``` directory.