# Codebase for inspecting modularity of neural networks

The official repository for our paper "Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks".

Video presentation: https://www.youtube.com/watch?v=XX0DD-x0868  
Paper: https://arxiv.org/abs/2010.02066  
Poster: https://people.idsia.ch/~csordas/poster_iclr2021.pdf

## Setup

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

### Downloading data

All datasets are downloaded automatically except the Mathematics Dataset which is hosted in Google Cloud and one has to log in with his/her Google account to be able to access it. Download the .tar.gz file manually from here:

https://console.cloud.google.com/storage/browser/mathematics-dataset?pli=1

Copy it to the ``cache/dm_math/`` folder. You should have a ``cache/dm_math/mathematics_dataset-v1.0.tar.gz`` file in the project folder if you did everyhing correctly. 

## Usage

### Running the experiments from the paper on a cluster

The code makes use of Weights and Biases for experiment tracking. In the ```sweeps``` directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run multiple configurations and seeds.

To reproduce our results, start a sweep for each of the YAML files in the ```sweeps``` directory. Run wandb agent for each of them in the _root directory of the project_. This will run all the experiments, and they will be displayed on the W&B dashboard. More details on how to run W&B sweeps can be found at https://docs.wandb.com/sweeps/quickstart. If you want to use a Linux cluster to run the experiments, you might find https://github.com/robertcsordas/cluster_tool useful.

The task ```sweeps/dm_math/dm_math_polycollect.yaml``` needs two 16Gb GPUs to run. In any of the experiments won't fit on a single GPU, it is possible to run them on multile (data parallel) by specifying multiple devices in CUDA_VISIBLE_DEVICE.

#### Re-creating plots from the paper

Edit config file ```paper/config.json```. Enter your project name in the field "wandb_project" (e.g. "username/modules").

Run the scripts in the ```paper``` directory. For example:

```bash
cd paper
./run_all.sh
```

The output will be generated in the ```paper/out/``` directory.

If you want to reproduce individual plots, it can be done by running individial python files in the ```paper``` directory.

### Running experiments locally

It is possible to run single experiments with Tensorboard without using Weights and Biases. This was supposed to be used for debugging the code locally.

If you want to run experiments locally, you can use ```run.py```:

```bash
./run.py sweeps/tuple_rnn.yaml
```

If the sweep in question has multiple parameter choices, ```run.py``` will interactively prompt choices of each of them.

The experiment also starts a Tensorboard instance automatically on port 7000. If the port is already occupied, it will incrementally search for the next free port.

Note that the plotting scripts work only with Weights and Biases.

# Porting to new networks

Networks require porting to be able to be analyzed. This is because ordinary setups are designed to run with a single set of weights on a batch of data. However, the masking process runs with a batch of weights and a batch of data. Usually, the batch size for the weights is much smaller than the batch size for the data, so the current implementation divides the data batch size by the weight batch size and uses a different set of weights for each part of the batch.

The current implementation has numerous compatible layers in the ```layers``` directory and networks in the ```models``` directory.

Analyze networks without modification is possible by sacrificing speed and modifying the code. For simulating K mask samples and batch size B, one should run K forward-backward passes with a single mask sample and batch size of B/K. The optimizer should be called after the K passes. Using one mask sample on the unmodified network is already supported by ```masked_model.py```. Alternatively, one can start K processes and run the forward passes in parallel and accumulate the masks' gradients.
