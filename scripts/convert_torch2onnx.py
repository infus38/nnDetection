#!/usr/bin/env python3

import os
import argparse
import re
from pathlib import Path
from typing import Tuple, Any, List
from types import SimpleNamespace

import torch
import onnx
from loguru import logger
from omegaconf import OmegaConf
from onnxsim import simplify

from nndet.ptmodule import MODULE_REGISTRY
from nndet.io import get_task, get_training_dir
from nndet.io.load import load_pickle


class ONNXModelConverter:
    """Handles the conversion of a PyTorch model to ONNX format."""

    def __init__(self, config):
        self.config = SimpleNamespace(**config)
        self.model_root = (
            Path(os.getenv("det_models", "/"))
            / get_task(self.config.task_id, name=True)
            / self.config.model_name
        )

    def get_model_paths(self) -> List[Tuple[Path, Path, Path, int]]:
        """Get paths for models based on fold specification"""

        if self.config.fold is None: # Consolidated mode - all folds
            logger.info("Using models from consolidated folder")
            return self._get_consolidated_paths()
        elif isinstance(self.config.fold, list): # Specific folds from their own directories
            return self._get_multi_fold_paths(self.config.fold)
        else: # Single fold in fold directory
            return self._get_fold_dir_paths(self.config.fold)

    def _get_consolidated_paths(self) -> List[Tuple[Path, Path, Path, int]]:
        """Get all paths from consolidated folder"""
        consolidated_dir = self.model_root / "consolidated"

        if not consolidated_dir.exists():
            raise FileNotFoundError(
                f"Consolidated folder not found at {consolidated_dir}. "
                "Please run 'nndet_consolidate' first."
            )

        model_files = list(consolidated_dir.glob("model_fold*.ckpt"))
        if not model_files:
            raise FileNotFoundError(
                f"No model files found in consolidated folder: {consolidated_dir}"
            )

        # Extract fold numbers from filenames
        fold_to_path = {}
        for model_path in model_files:
            match = re.search(r"fold(\d+)", model_path.stem)
            if match:
                fold = int(match.group(1))
                fold_to_path[fold] = model_path

        paths = []
        for fold, model_path in fold_to_path.items():
            paths.append((
                consolidated_dir / "config.yaml",   # Shared config
                consolidated_dir / "plan.pkl",      # Shared plan
                model_path,                         # Specific checkpoint for the fold
                fold                                # Fold number
            ))

        if not paths:
            raise FileNotFoundError("No matching models found in consolidated folder")
        return paths

    def _get_multi_fold_paths(self, folds: List[int]) -> List[Tuple[Path, Path, Path, int]]:
        """Get paths for multiple folds from their individual directories"""
        paths = []
        for fold in folds:
            try:
                paths.extend(self._get_fold_dir_paths(fold))
            except FileNotFoundError as e:
                logger.warning(f"Skipping fold {fold}: {str(e)}")

        if not paths:
            raise FileNotFoundError("No valid folds found among specified fold directories")
        return paths

    def _get_fold_dir_paths(self, fold: int) -> List[Tuple[Path, Path, Path, int]]:
        """Get paths for a single fold from fold directory"""
        training_dir = get_training_dir(self.model_root, fold)

        required_files = {
            "config": training_dir / "config.yaml",
            "plan": training_dir / "plan.pkl",
            "checkpoint": training_dir / "model_best.ckpt",
        }

        missing_files = [name for name, path in required_files.items() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing files in {training_dir}: {', '.join(missing_files)}")

        return [(
            required_files["config"],
            required_files["plan"],
            required_files["checkpoint"],
            fold
        )]

    @staticmethod
    def initialize_model(config_path: Path, plan_path: Path) -> Tuple[Any, dict, dict]:
        """Initialize the model architecture."""
        model_config = OmegaConf.load(config_path)
        data_plan = load_pickle(plan_path)

        # Ensure  the  model  configuration  contains  a   module   name
        # otherwise we cannot get the model class from the registry
        if (module_name := model_config.get("module")) is None:
            raise ValueError("Model configuration missing module specification|field.")

        if (model_class := MODULE_REGISTRY.get(module_name)) is None:
            raise ValueError(
                f"Unregistered model: {module_name}. Available: {sorted(MODULE_REGISTRY.mapping.keys())}"
            )

        return model_class, OmegaConf.to_container(model_config), data_plan

    @staticmethod
    def get_input_spec(data_plan: dict) -> Tuple[int, Tuple[int, int, int]]:
        """Extract input specifications from the data plan."""

        # Determines the number of input channels (modalities) based  on
        # the dataset properties.
        # If `modalities` is a dictionary, the  number  of  channels  is
        # inferred from its length, where each key-value pair represents
        # a unique modality. For non-dictionary values, `modalities`  is
        # treated as a single integer indicating  the  total  number  of
        # channels.
        modalities = data_plan.get("dataset_properties", {}).get("modalities", 1)
        num_channels = (
            len(modalities) if isinstance(modalities, dict) else int(modalities)
        )

        # Extracts the patch size from the plan, which  defines  the  3D
        # input dimensions for the model
        patch_size = data_plan.get("patch_size", [128, 128, 128])
        if len(patch_size) != 3:
            raise ValueError("Patch size must have exactly 3 dimensions.")

        return num_channels, tuple(map(int, patch_size))

    def convert(self, model: torch.nn.Module, input_shape: Tuple[int, ...],
               output_path: Path, fold: int):
        """Convert the PyTorch model to ONNX."""

        # Skip the channel dimension in dynamic axes because the  number
        # of input channels, is determined by the model architecture and
        # can't vary without changing model weights or layers
        dynamic_axes = {
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch"},
        } if self.config.dynamic_axes else None

        torch.onnx.export(model=model, args=torch.randn(input_shape), 
                          f=str(output_path), input_names=["input"],
                          output_names=["output"], dynamic_axes=dynamic_axes,
                          opset_version=self.config.opset_version,
                          dynamo=self.config.dynamo, verify=self.config.dynamo)

        # Optionally simplify the ONNX IR to reduce complexity and size
        if self.config.simplify:
            try:
                model_onnx = onnx.load(str(output_path))
                simplified, check = simplify(model_onnx)
                if check:
                    onnx.save(simplified, str(output_path))
            except Exception as e:
                logger.warning(f"Simplification failed: {e}")

        # Perform a structural check on the generated ONNX IR to  ensure
        # validity
        onnx.checker.check_model(onnx.load(str(output_path)))
        logger.success(f"ONNX model for fold {fold} saved at: {output_path}")

    def execute(self):
        """Execute the model conversion pipeline."""
        mode_desc = ("consolidated folder" if self.config.fold is None
                     else f"folds {self.config.fold}" if isinstance(self.config.fold, list)
                     else f"fold {self.config.fold} directory")

        logger.info(f"Starting conversion of {self.config.model_name} for task {self.config.task_id}")
        logger.info(f"Processing mode: {mode_desc}")

        model_paths = self.get_model_paths()
        for config_path, plan_path, checkpoint_path, fold in model_paths:
            # Initialize and load model architecture
            model_class, model_config, data_plan = self.initialize_model(config_path, plan_path)

            # Create model instance
            model_instance = model_class(
                model_cfg=model_config.get("model_cfg", {}),
                trainer_cfg=model_config.get("trainer_cfg", {}),
                plan=data_plan
            )

            # Load weights from checkpoint
            model_instance.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"])
            model_instance.eval()

            # Get input specifications from the data plan
            num_channels, input_dims = self.get_input_spec(data_plan)
            logger.info(f"Fold {fold}: Input channels={num_channels}, Dimensions={input_dims}")
            input_shape = (1, num_channels, *input_dims)

            # Determine output path
            if self.config.output_path:
                output_path = self.config.output_path / f"{self.config.model_name}_fold{fold}.onnx"
            else:
                output_path = checkpoint_path.parent / f"{self.config.model_name}_fold{fold}.onnx"

            self.convert(model_instance, input_shape, output_path, fold)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX.")
    parser.add_argument("task_id", help="Task ID (e.g., 000)")
    parser.add_argument("model_name", help="Model name (e.g., RetinaUNetV001)")

    parser.add_argument(
        "--fold", nargs="*", type=int, default=None,
        help=("Convert specific folds from their own directories (default: all in consolidated). "
              "For single fold in fold directory, specify one fold number.")
    )

    parser.add_argument("--output", type=Path, help="Output directory for ONNX models")
    parser.add_argument("--opset_version", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", dest="simplify", help="Simplify the produced ONNX IR")
    parser.add_argument("--static_axes", action="store_false", dest="dynamic_axes", help="Use static axes")
    parser.add_argument("--dynamo", action="store_true", dest="dynamo", help="use torch dynamo for export")

    return parser.parse_args()


def main():
    """Main function to run the ONNX conversion."""
    args = parse_args()

    # Process fold argument
    if args.fold is None:
        folds = None  # All folds in consolidated
    elif len(args.fold) == 1:
        folds = args.fold[0]  # Single fold in their own directory
    else:
        folds = args.fold  # Specific folds in their own directories

    config = {"task_id": args.task_id, "model_name": args.model_name, "fold": folds,
              "output_path": args.output, "opset_version": args.opset_version, 
              "simplify": args.simplify, "dynamic_axes": args.dynamic_axes,
              "dynamo": args.dynamo
    }

    converter = ONNXModelConverter(config)
    converter.execute()

if __name__ == "__main__":
    main()