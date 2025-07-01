"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib
import argparse
import os
import sys
from typing import Any, Mapping, Type, TypeVar

from omegaconf import OmegaConf
from loguru import logger
from pathlib import Path

from nndet.utils.check import env_guard
from nndet.planning import PLANNER_REGISTRY
from nndet.io import get_task, get_training_dir
from nndet.io.load import load_pickle
from nndet.inference.loading import load_all_models, load_final_model
from nndet.inference.helper import predict_dir
from nndet.inference.onnx_helper import onnx_predict_dir
from nndet.utils.check import check_data_and_label_splitted


def run(cfg: dict,
        training_dir: Path,
        process: bool = True,
        num_models: int = None,
        num_tta_transforms: int = None,
        test_split: bool = False,
        num_processes: int = 3,
        model_fn=load_all_models,
        ):
    """
    Run inference pipeline

    Args:
        cfg: configurations
        training_dir: path to model directory
        process: preprocess test data
        num_models: number of models to use for ensemble; if None all Models
            are used
        num_tta_transforms: number of tta transformation; if None the maximum
            number of transformation is used
        test_split: Typical usage of nnDetection will never require
            this option! Predict an already preprocessed split of the original
            training data. The 'test' split needs to be located in fold 0 
            of a manually created split file.
    """
    plan = load_pickle(training_dir / "plan_inference.pkl")

    preprocessed_output_dir = Path(cfg["host"]["preprocessed_output_dir"])
    # Isolate torch predictions from others
    prediction_dir = training_dir / "test_predictions" / "torch"

    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
        )
    logger.add(Path(training_dir) / "inference.log", level="INFO")

    if process:
        planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])
        planner_cls.run_preprocessing_test(
            preprocessed_output_dir=preprocessed_output_dir,
            splitted_4d_output_dir=cfg["host"]["splitted_4d_output_dir"],
            plan=plan,
            num_processes=num_processes,
        )

    prediction_dir.mkdir(parents=True, exist_ok=True)
    if test_split:
        source_dir = preprocessed_output_dir / plan["data_identifier"] / "imagesTr"
        case_ids = load_pickle(training_dir / "splits.pkl")[0]["test"]
    else:
        source_dir = preprocessed_output_dir / plan["data_identifier"] / "imagesTs"
        case_ids = None

    predict_dir(source_dir=source_dir,
                target_dir=prediction_dir,
                cfg=cfg,
                plan=plan,
                source_models=training_dir,
                num_models=num_models,
                num_tta_transforms=num_tta_transforms,
                model_fn=model_fn,
                restore=True,
                case_ids=case_ids,
                **cfg.get("inference_kwargs", {}),
                )


def set_arg(cfg: Mapping, key: str, val: Any, force_args: bool) -> Mapping:
    """
    Check if value of config and given key match and handle approriately:
    If values match no action will be performend.
    If the values do not match and force_args is activated the value
    in the config will be overwritten.
    if the values do not match and force args is deactivatd a ValueError
    will be raised.

    Args:
        cfg: config to check and write values to
        key: key to check.
        val: Potentially new value.
        force_args: Enable if config value should be overwritten if values do
            not match.

    Returns:
        Type[dict]: config with potentially changed key
    """
    if key not in cfg:
        raise ValueError(f"{key} is not in config.")

    if cfg[key] != val:
        if force_args:
            logger.warning(f"Found different values for {key}, will overwrite {cfg[key]} with {val}")
            cfg[key] = val
        else:
            raise ValueError(f"Found different values for {key} and overwrite disabled."
                             f"Found {cfg[key]} but expected {val}.")
    return cfg

def predict_for_fold(fold, model_fn, task_name, model, task_model_dir, num_models, num_tta_transforms,
                     test_split, process, force_args, ov, check, num_processes):

    tdir = get_training_dir(task_model_dir / task_name / model, fold)
    if test_split and process:
        raise ValueError("When using the test split option raw data is not supported. Need to add --no_preprocess flag!")
    cfg = OmegaConf.load(str(tdir / "config.yaml"))
    cfg = set_arg(cfg, "task", task_name, force_args=force_args)
    cfg["exp"] = set_arg(cfg["exp"], "fold", fold, force_args=True if fold == -1 else force_args)
    cfg["exp"] = set_arg(cfg["exp"], "id", model, force_args=force_args)
    (cfg.merge_with_dotlist(
        (ov or []) + [
            "host.parent_data=${oc.env:det_data}",
            "host.parent_results=${oc.env:det_models}"
        ]
    ))
    [importlib.import_module(imp) for imp in cfg.get("additional_imports", [])]
    if check:
        if test_split:
            raise ValueError("Check is not supported for test split option.")
        check_data_and_label_splitted(
            task_name=cfg["task"], test=True, labels=False, full_check=True
        )
    run(
        OmegaConf.to_container(cfg, resolve=True), tdir, process=process,
        num_models=num_models, num_tta_transforms=num_tta_transforms,
        test_split=test_split, num_processes=num_processes, model_fn=model_fn,
    )

def onnx_predict_for_fold(
    fold, model_fn, task_name, model, task_model_dir, num_models, num_tta_transforms,
    test_split, process, force_args, ov, check, num_processes
):

    tdir = get_training_dir(task_model_dir / task_name / model, fold)
    cfg = OmegaConf.load(tdir / "config.yaml")
    plan = load_pickle(tdir / "plan_inference.pkl")
    pre_dir = Path(cfg["host"]["preprocessed_output_dir"])

    # Isolate ONNX predictions from torch one
    prediction_dir = tdir / "test_predictions" / "onnx"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Data source selection:
    # - Test split: use predefined cases from split file
    # - Validation: scan entire directory (case_ids=None)
    source_dir, case_ids = (
        (pre_dir / plan["data_identifier"] / "imagesTr", load_pickle(tdir / "splits.pkl")[0]["test"])
        if test_split else
        (pre_dir / plan["data_identifier"] / "imagesTs", None)
    )

    # Find ONNX model paths for ensemble
    if fold == -1:
        # Consolidated: use all ONNX models in the folder
        candidates = sorted(tdir.glob(f"{model}_fold*.onnx"))
        if not candidates:
            # fallback to single model.onnx if no fold models found
            candidates = [tdir / "model.onnx"] if (tdir / "model.onnx").exists() else []
    else:
        candidates = [tdir / f"{model}_fold{fold}.onnx"]
        if not candidates[0].exists():
            candidates = [tdir / "model.onnx"] if (tdir / "model.onnx").exists() else []

    onnx_model_paths = [c for c in candidates if c.exists()]
    if not onnx_model_paths:
        raise FileNotFoundError(f"ONNX model(s) not found in {tdir}. Please export your model(s) to ONNX first.")

    logger.info(f"Starting ONNX prediction using models: {onnx_model_paths}")

    if len(onnx_model_paths) == 1:
        # Single model, use standard prediction
        onnx_predict_dir(source_dir=source_dir, target_dir=prediction_dir,
            onnx_model_path=onnx_model_paths[0], plan=plan,
            case_ids=case_ids, restore=True,
        )
    else:
        # Ensemble prediction
        from nndet.inference.onnx_helper import onnx_predict_ensemble_dir
        onnx_predict_ensemble_dir(
            source_dir=source_dir, target_dir=prediction_dir,
            onnx_model_paths=onnx_model_paths, plan=plan,
            case_ids=case_ids, restore=True,
        )


@env_guard
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument('-f', '--fold', type=int, nargs='*', required=False, default=None,
                        help="Folds to use for prediction. If omitted, uses all models in the consolidated dir. "
                             "If one or more folds are specified, uses those folds from their respective training dirs.")
    parser.add_argument('-nmodels', '--num_models', type=int, default=None,
                        required=False,
                        help="number of models for ensemble(per default all models will be used)."
                             "NOT usable by default -- will use all models inside the folder!",
                        )
    parser.add_argument('-ntta', '--num_tta', type=int, default=None,
                        help="number of tta transforms (per default most tta are chosen)",
                        required=False,
                        )
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        default=None,
                        required=False,
                        help=("overwrites for config file. "
                              "inference_kwargs can be used to add additional "
                              "keyword arguments to inference."),
                        )
    parser.add_argument('--no_preprocess', action='store_false', help="Preprocess test data")
    parser.add_argument('--force_args', action='store_true',
                        help=("When transferring models betweens tasks the name "
                        "and fold might differ from the original one. "
                        "This forces an overwrite to the passed in arguments of"
                        " this function. This can be dangerous!"),
                        )
    parser.add_argument('--test_split', action='store_true',
                        help=("Typical usage of nnDetection will never require "
                              "this option! Predict an already preprocessed "
                              "split of the original training data. "
                              "The 'test' split needs to be located in fold 0 "
                              "of a manually created split file."),
                        )
    parser.add_argument('--check',
                    help="Run check of the test data before predicting",
                    action='store_true',
                    )   
    parser.add_argument('-npp', '--num_processes_preprocessing',
                        type=int, default=3, required=False,
                        help="Number of processes to use for resampling.",
                        )
    parser.add_argument('--onnx', action='store_true', help="Use ONNX model for prediction instead of PyTorch.")


    args = parser.parse_args()
    model = args.model
    folds = args.fold
    task = args.task
    num_models = args.num_models
    num_tta_transforms = args.num_tta
    ov = args.overwrites
    force_args = args.force_args
    test_split = args.test_split
    check = args.check
    num_processes = args.num_processes_preprocessing
    use_onnx = args.onnx

    task_name = get_task(task, name=True)
    task_model_dir = Path(os.getenv("det_models"))
    process = args.no_preprocess

    if use_onnx:
        predict_func = onnx_predict_for_fold
    else:
        predict_func = predict_for_fold

    if not folds:
        predict_func(
            -1, load_all_models, task_name, model, task_model_dir,
            num_models, num_tta_transforms, test_split, process, force_args,
            ov, check, num_processes
        )
    else:
        for fold in folds:
            predict_func(
                fold, lambda *a, **kw: load_final_model(*a, identifier='best', **kw),
                task_name, model, task_model_dir, 1, num_tta_transforms,
                test_split, process, force_args, ov, check, num_processes
            )


if __name__ == '__main__':
    main()
