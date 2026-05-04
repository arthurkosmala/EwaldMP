# Ewald Message Passing

Reference implementation of the Ewald message passing scheme, proposed in the paper

Ewald-based Long-Range Message Passing for Molecular Graphs  
by Arthur Kosmala, Johannes Gasteiger, Nicholas Gao, Stephan Günnemann  
Accepted at ICML 2023   

  
![Ewald Concept Figure](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/_my_direct_uploads/ewald_concept_fig.png)

Models for which Ewald message passing is currently implemented:

- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/schnet.py)]
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/dimenet_plus_plus.py)]
- PaiNN [[`arXiv`](https://arxiv.org/abs/2102.03150)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/painn)]
- GemNet-(d)T [[`arXiv`](https://arxiv.org/abs/2106.08903)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/gemnet)]

Currently supported datasets:
 - OC20 [[`arXiv`](https://arxiv.org/abs/2010.09990)] [[`download`](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md)]
 - OE62 [[`arXiv`](https://arxiv.org/abs/2001.08954)] [[`download`](https://mediatum.ub.tum.de/1507656)]

This repository was forked from the [Open Catalyst 2020 (OC20) Project](https://github.com/Open-Catalyst-Project/ocp), which provides the codebase for training and inference on OC20, as well as on the OE62 dataset which we integrated into the OC20 pipeline.


## Installation

**Note**: this project was based off an early version of the `fairchem`-v1 repo, which has since been superseeded by `fairchem`-v2. Following the below installation instructions (adapted from `fairchem`-v1) should result in a consistent environment for Nvidia Ampere GPUs. Nvidia Hopper GPUs and newer require `cudatoolkit`>=11.8 (ideally >=12.0 for full Hopper features) and a suitable `pytorch` version. If supported by your hardware, we recommend using the preset environment files, since reproducibility was not tested in newer environments.

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html).

After installing [conda](http://conda.pydata.org/), run the following commands
to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ocp-models` and install dependencies.

### Pre-install step

Install `conda-merge`:
```bash
pip install conda-merge
```
If you're using system `pip`, then you may want to add the `--user` flag to avoid using `sudo`.
Check that you can invoke `conda-merge` by running `conda-merge -h`.

### GPU machines

Install the dependencies:
```bash
conda-merge env.common.yml env.gpu.yml > env.yml
conda env create -f env.yml
```
Activate the conda environment with `conda activate ocp-models`.

Install this package with `pip install -e .`.

Finally, install the pre-commit hooks:
```bash
pre-commit install
```

### CPU-only machines

Please skip the following if you completed the with-GPU installation from above.

```bash
conda-merge env.common.yml env.cpu.yml > env.yml
conda env create -f env.yml
conda activate ocp-models
pip install -e .
pre-commit install
```

We further recommend installing the `jupyter` package to access our example training and evaluation notebooks, as well as the `seml` package [[`github`](https://github.com/TUM-DAML/seml)] to run and log experiments from the CLI. To reproduce the long-range binning analyses from the Ewald message passing paper, please install the `simple-dftd3` package [[`installation instructions`](https://dftd3.readthedocs.io/en/latest/installation.html)] including the Python API.

## Data download and preprocessing

Dataset download links and instructions for __OC20__ can currently be found on the [[`fairchem website`](https://fair-chem.github.io/oc20/)].

__To replicate our experiments on OC20__, please consider downloading the following data splits for the _Structure to Energy and Forces_ (S2EF) task:
- train_2M
- val_id
- val_ood_ads
- val_ood_cat
- val_ood_both
- test

__To replicate our experiments on OE62__, please download the raw OE62 dataset [[`media server`](https://mediatum.ub.tum.de/1459255?show_id=1507656)]. Afterwards, run the `OE62_dataset_preprocessing.ipynb` notebook to deposit LMDB files containing the training, validation and test splits in a new `oe62` directory.

## Train and evaluate models from notebook

For interactive use, our notebook `train_and_evaluate.ipynb` allows training and evaluation of all studied model variants (baselines, Ewald versions, comparison studies) on OE62 and OC20. We also provide the notebook `evaluate_from_checkpoint.ipynb` to evaluate previously trained models.

We only recommend notebook-based training for OE62, as OC20 training may take days on a single `Nvidia A100` graphics card even for the fastest models.

## Train and evaluate models from CLI

Alternatively, experiments can be started from the CLI using
```bash
seml [your_experiment_name] add configs_[oe62/oc20]/[model_variant].yml start
```
For example, to train the SchNet model variant with added Ewald message passing on OE62, type
```bash
seml schnet_oe62_ewald add configs_oe62/schnet_oe62_ewald.yml start
```
Experiments beyond the studied variants can be easily defined by adding or modifying config YAML files. The log files deposited in `logs_oe62` or `logs_oc20` specify paths to the model checkpoint file as well as a tensorboard file to track progress. `Tensorboard` files can be found in the `logs` directory.

## Contact

Please reach out to [a.kosmala@tum.de](mailto:a.kosmala@tum.de) if you have any questions.

## Cite

Please cite our paper if you use our method or code in your own works:
```
@inproceedings{kosmala_ewaldbased_2023,
  title = {Ewald-based Long-Range Message Passing for Molecular Graphs},
  author = {Kosmala, Arthur and Gasteiger, Johannes and Gao, Nicholas and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Machine Learning (ICML)},
  year = {2023} 
}
```

## License

This project is released under the Hippocratic License 3.0. This concerns only the files added to the OC20 repository, which was itself released under the MIT license. 
