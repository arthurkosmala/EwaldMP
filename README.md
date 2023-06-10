# Ewald Message Passing

Reference implementation of the Ewald message passing scheme, proposed in the paper

Ewald-based Long-Range Message Passing for Molecular Graphs  
by Arthur Kosmala, Johannes Gasteiger, Nicholas Gao, Stephan Günnemann  
Accepted at ICML 2023

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

To setup a `conda` environment with the required dependencies, please follow the [OCP installation instructions](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md). They should work identically in this repository. We further recommend installing the `jupyter` package to access our example training and evaluation notebooks, as well as the `seml` package [[`github`](https://github.com/TUM-DAML/seml)] to run and manage (especially longer) experiments from the CLI. To reproduce the long-range binning analyses from the Ewald message passing paper, please install the `simple-dftd3` package [[`installation instructions`](https://dftd3.readthedocs.io/en/latest/installation.html)] including the Python API.

## Data download and preprocessing

Dataset download links and instructions for __OC20__ are in [DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md) in the original OC20 repository.

__To replicate our experiments on OC20__, please consider downloading the following data splits for the _Structure to Energy and Forces_ (S2EF) task:
- train_2M
- val_id
- val_ood_ads
- val_ood_cat
- val_ood_both
- test

__To replicate our experiments on OE62__, please download the raw OE62 dataset [[`media server`](https://mediatum.ub.tum.de/1459255?show_id=1507656)]. Afterwards, run the `OE62_dataset_preprocessing.ipynb` notebook to deposit LMDB files containing the training, validation and test splits in a new `oe62` directory.

## Train and evaluate models from notebook

For an interactive introduction, consider using our notebook `train_and_evaluate.ipynb` that allows training and evaluation of all studied model variants (baselines, Ewald versions, comparison studies) on OE62 and OC20. We also provide the notebook `evaluate_from_checkpoint.ipynb` to evaluate previously trained models.

Approximate OE62 and OC20 training times for all model variants are provided as guidance in the training notebook. We only recommend notebook-based training for OE62, as OC20 training takes over a day using a single `Nvidia A100` graphics card for our cheapest models.

## Train and evaluate models from CLI

Alternatively, experiments can be started from the CLI using
```bash
seml [your_experiment_name] add configs_[oe62/oc20]/[model_variant].yml start
```
For example, to train the SchNet model variant with added Ewald message passing on OE62, type
```bash
seml schnet_oe62_ewald add configs_oe62/schnet_oe62_ewald.yml start
```
Experiments beyond our studied variants can be easily defined by adding or modifying config YAML files. Once an experiment is started, a log file identified by the associated `SLURM jobid` is generated in `logs_oe62` or `logs_oc20`. Among other information, it contains the path to the model checkpoint file, as well as the path to a tensorboard file that may be used to track progress. `Tensorboard` files can be found in the `logs` directory.

## Contact

Please reach out to [arthur.kosmala@tum.de](mailto:arthur.kosmala@tum.de) if you have any questions.

## Cite

Please cite our paper if you use our method or code in your own works:
```
@misc{kosmala2023ewaldbased,
      title={Ewald-based Long-Range Message Passing for Molecular Graphs}, 
      author={Arthur Kosmala and Johannes Gasteiger and Nicholas Gao and Stephan Günnemann},
      year={2023},
      eprint={2303.04791},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This project is released under the Hippocratic License 3.0.
