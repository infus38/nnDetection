#!/usr/bin/env python3

import os
import argparse
import warnings
from pathlib import Path
from typing import Tuple, Any
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

    def validate_paths(self) -> Tuple[Path, Path, Path, Path]:
        """Validate paths required for conversion and return them."""
        model_root = (
            Path(os.getenv("det_models", "/"))
            / get_task(self.config.task_id, name=True)
            / self.config.model_name
        )

        training_dir = get_training_dir(model_root, self.config.fold)

        required_files = {
            "config": training_dir / "config.yaml",
            "plan": training_dir / "plan.pkl",
            "checkpoint": training_dir / "model_best.ckpt",
        }

        missing_files = [name for name, path in required_files.items() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing files in {training_dir}")

        return (required_files["config"],required_files["plan"],
                required_files["checkpoint"], training_dir)

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


    def convert(self, model: torch.nn.Module, input_shape: Tuple[int, ...], output_path: Path):
        """Convert the PyTorch model to ONNX."""

        # Skip the channel dimension in dynamic axes because the  number
        # of input channels, is determined by the model architecture and
        # can't vary without changing model weights or layers
        dynamic_axes = {
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch"},
        } if self.config.dynamic_axes else None

        torch.onnx.export(model=model, args=torch.randn(input_shape), 
                          f=str(output_path),input_names=["input"], 
                          output_names=["output"], dynamic_axes=dynamic_axes,
                          opset_version=self.config.opset_version)

        # Optionally simplify the ONNX IR to reduce complexity and size
        print(self.config.simplify)
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
        logger.success(f"ONNX model saved at: {output_path}")


    def execute(self):
        """Execute the model conversion pipeline."""
        logger.info(f"Starting conversion of {self.config.model_name} for task {self.config.task_id}")

        config_path, plan_path, checkpoint_path, training_dir = self.validate_paths()
        model_class, model_config, data_plan = self.initialize_model(config_path, plan_path)

        model_instance = model_class(
            model_cfg=model_config.get("model_cfg", {}),
            trainer_cfg=model_config.get("trainer_cfg", {}),
            plan=data_plan
        )
        model_instance.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"])
        model_instance.eval()

        num_channels, input_dims = self.get_input_spec(data_plan)
        logger.info(f"Model input channels: {num_channels}, input dimensions: {input_dims}")
        input_shape = (1, num_channels, *input_dims)

        output_path = self.config.output_path or training_dir / f"{self.config.model_name}_fold{self.config.fold}.onnx"
        self.convert(model_instance, input_shape, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX.")
    parser.add_argument("task_id", help="Task ID (e.g., 000)")
    parser.add_argument("model_name", help="Model name (e.g., RetinaUNetV001)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--output", type=Path, help="Output path for ONNX model")
    parser.add_argument("--opset_version", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", dest="simplify", help="Simplify the produced ONNX IR")
    parser.add_argument("--static_axes", action="store_false", dest="dynamic_axes", help="Use static axes")

    return parser.parse_args()

def main():
    """Main function to run the ONNX conversion."""
    args = parse_args()
    config = {"task_id": args.task_id, "model_name": args.model_name, "fold": args.fold,
              "output_path": args.output, "opset_version": args.opset_version, 
              "simplify": args.simplify, "dynamic_axes": args.dynamic_axes,
    }
    converter = ONNXModelConverter(config)
    converter.execute()


if __name__ == "__main__":
    main()