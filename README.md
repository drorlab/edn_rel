# EDN

## Overview

Pytorch implementation of EDN network from [Protein model quality assessment
using rotation-equivariant transformations on point clouds](https://arxiv.org/abs/2011.13557).
EDN predicts the GDT-TS of a protein model.

## Installation

### Create conda environment

To create the conda environment and install the required libraries to run edn,
run the following:
```
conda env create -f environment.yml
conda activate edn
```

If you want to name your conda environment to something else, change name declared in the `environment.yml`.

### Install e3nn

In addition, we need to install the EDN-compatible version of e3nn (note this version is only provided
for compatability, further development should be done using the main e3nn branch):

`pip install git+ssh://git@github.com/drorlab/e3nn_edn.git`

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

You can enable a gpu with the `--gpus` flag.  It is also recommended to provision additional CPUs with the `--num_workers` flags (more is better). GPU should have at least 12GB of memory.  For example to train:

`python -m edn.train data/test data/test --batch_size=2 --accumulate_grad_batches=32 --learning_rate=0.001 --max_epochs=6 --output_dir out/model --gpus=1 --num_workers=4`

And to predict:

`python -m edn.predict data/test data/sample_weights.ckpt output.csv --nolabels --gpus=1 --num_workers=4`


### Create input LMDB dataset

The LMDB data format allows for fast, random access to your data, which is useful for machine learning workloads. To convert a PDB dataset to the LMDB format, you can run:

`python -m atom3d.datasets data/pdbs test_lmdb -f pdb`

