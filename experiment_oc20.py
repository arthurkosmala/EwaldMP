import logging
import os

import seml
import torch
from sacred import Experiment

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_logging
from ocpmodels.trainers import ForcesTrainer

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
    dataset_train,
    dataset_id,
    dataset_ood_ads,
    dataset_ood_cat,
    dataset_ood_both,
    task,
    model,
    optimizer,
    logger,
    name,
):
    setup_logging()

    # ************************************************************************************************************************************************
    # Comment out the part enclosed in stars if you only want to validate or test on OC20
    # ************************************************************************************************************************************************
    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset_train,
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

    trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    trainer.train()

    checkpoint_path = os.path.join(
        trainer.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt"
    )
    # ************************************************************************************************************************************************
    # ************************************************************************************************************************************************
    # ************************************************************************************************************************************************

    #### Validation part ####

    # checkpoint_path = [your checkpoint path if you only want to validate or test]
    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset_id,
        optimizer=optimizer,
        identifier="val_id",
        run_dir="./",
        # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=True,  # if True, do not save checkpoint, logs, or results, set to False if you want to test not validate (results file needed)
        print_every=5000,
        seed=0,  # random seed to use
        logger=logger,  # logger of choice (tensorboard and wsqueueandb supported)
        local_rank=0,
        amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    metrics = trainer.validate()
    results_id = {key: val["metric"] for key, val in metrics.items()}

    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset_ood_ads,
        optimizer=optimizer,
        identifier="val_ood_ads",
        run_dir="./",
        # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=True,  # if True, do not save checkpoint, logs, or results, set to False if you want to test not validate (results file needed)
        print_every=5000,
        seed=0,  # random seed to use
        logger=logger,  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    metrics = trainer.validate()
    results_ood_ads = {key: val["metric"] for key, val in metrics.items()}

    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset_ood_cat,
        optimizer=optimizer,
        identifier="val_ood_cat",
        run_dir="./",
        # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=True,  # if True, do not save checkpoint, logs, or results, set to False if you want to test not validate (results file needed)
        print_every=5000,
        seed=0,  # random seed to use
        logger=logger,  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    metrics = trainer.validate()
    results_ood_cat = {key: val["metric"] for key, val in metrics.items()}

    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset_ood_both,
        optimizer=optimizer,
        identifier="val_ood_both",
        run_dir="./",
        # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=True,  # if True, do not save checkpoint, logs, or results, set to False if you want to test not validate (results file needed)
        print_every=5000,
        seed=0,  # random seed to use
        logger=logger,  # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    metrics = trainer.validate()
    results_ood_both = {key: val["metric"] for key, val in metrics.items()}

    results = {
        "id": results_id,
        "ood_ads": results_ood_ads,
        "ood_cat": results_ood_cat,
        "ood_both": results_ood_both,
    }

    # the returned result will be written into the database
    return results
