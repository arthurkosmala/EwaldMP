import logging
import os

import seml
import torch
from sacred import Experiment

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_logging
from ocpmodels.trainers import EnergyTrainer

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
    dataset,
    task,
    model,
    optimizer,
    logger,
    name,
):
    setup_logging()

    # checkpoint_path_train = [checkpoint path if resuming from previous run]

    trainer = EnergyTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier=name,
        run_dir="./",
        # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=False,  # if True, do not save checkpoint, logs, or results
        print_every=5000,
        seed=0,  # random seed to use
        logger=logger,  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    # trainer.load_checkpoint(checkpoint_path=checkpoint_path_train)
    trainer.train()

    results_memory = {
        "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "max_reserved": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }

    #### Validation part ####
    checkpoint_path = os.path.join(
        trainer.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt"
    )

    trainer = EnergyTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="val",
        run_dir="./",
        # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=True,  # if True, do not save checkpoint, logs, or results
        print_every=5000,
        seed=0,  # random seed to use
        logger=logger,  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    metrics = trainer.validate()
    results = {key: val["metric"] for key, val in metrics.items()}

    results = {
        "performance": results,
        "memory": results_memory,
    }

    # the returned result will be written into the database
    return results
