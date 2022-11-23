# EDN

## Overview

PyTorch implementation of the EDN network from [Protein model quality assessment
using rotation-equivariant transformations on point clouds](https://arxiv.org/abs/2011.13557).
EDN is designed to predict the GDT-TS score of a protein model.

## Installation

### Create conda environment

```
conda create -n edn python=3.9 pip
conda activate edn
```
### Install pytorch

Install appropriate versions of torch and attendant libraries.  You need to figure out the appropriate version of cuda you have on your system, and set below.  Currently shown for cuda 11.7, if you want to install for CPU only, use CUDA="".

```
TORCH="1.13.0"
CUDA="cu117"
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```
### Install pytorch-lightning and other generic dependencies.

`pip install pytorch-lightning python-dotenv wandb`

### Install e3nn

In addition, we need to install the EDN-compatible version of the e3nn library (https://github.com/e3nn/e3nn). Please note that the specific version is only provided for compatability, further development should be done using the main e3nn branch.

`pip install git+ssh://git@github.com/drorlab/e3nn_edn.git`

### Install atom3d

Install of atom3d:

`pip install atom3d`

## Usage

### Training

To train the model on CPU, you can run the following command:

`python -m edn.train data/test data/test --batch_size=2 --accumulate_grad_batches=32 --learning_rate=0.001 --max_epochs=6 --output_dir out/model --num_workers=4`

Note this will run quite slowly. To run faster, consider using a GPU (see below).

### Inference

To make a prediction, the general format is as follows:

`python -m edn.predict input_dir checkpoint.ckpt output.csv [--nolabels]`

For example, to predict on the test lmdb file included in the repository, using dummy weights:

`python -m edn.predict data/test data/sample_weights.ckpt output.csv --nolabels`

The expected output in `output.csv` for the above command would be (with possible fluctuation in up to 7th decimal place):

```
id,target,pred
T0843-Alpha-Gelly-Server_TS3.pdb,0.0000000,0.4594232
T0843-BioSerf_TS1.pdb,0.0000000,0.5363037
```

### Using a GPU

You can enable a GPU with the `--gpus` flag.  It is also recommended to provision additional CPUs with the `--num_workers` flags (more is better). The GPU should have at least 12GB of memory.  For example to train:

`python -m edn.train data/test data/test --batch_size=2 --accumulate_grad_batches=32 --learning_rate=0.001 --max_epochs=6 --output_dir out/model --gpus=1 --num_workers=4`

And to predict:

`python -m edn.predict data/test data/sample_weights.ckpt output.csv --nolabels --gpus=1 --num_workers=4`


### Create input LMDB dataset

The LMDB data format from Atom3D (https://www.atom3d.ai/) allows for fast, random access to your data, which is useful for machine learning workloads. To convert a PDB dataset to the LMDB format, you can run:

`python -m atom3d.datasets data/pdbs test_lmdb -f pdb`

