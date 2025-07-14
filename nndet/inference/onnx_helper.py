import numpy as np
import onnxruntime as ort
from pathlib import Path
from loguru import logger
from nndet.io.load import load_pickle, save_pickle
from nndet.io.paths import get_case_id_from_path
from nndet.inference.onnx_postprocess import ONNXPostProcessor, create_postprocessor_from_config


def onnx_predict_ensemble_dir(source_dir, target_dir, onnx_model_paths, plan, case_ids=None,
                              restore=True, **kwargs):
    """
    Entry  point  for  ensemble  prediction  -   delegates   to   shared
    implementation
    """
    return _onnx_predict_common(source_dir, target_dir, onnx_model_paths, plan, case_ids=case_ids,
                                restore=restore, ensemble=True, **kwargs)

def onnx_predict_dir(source_dir, target_dir, onnx_model_path, plan, case_ids=None,
                     restore=True, **kwargs):
    """
    Entry point for single model prediction - wraps  path  in  list  for
    shared impl
    """
    return _onnx_predict_common(source_dir, target_dir, [onnx_model_path], plan, case_ids=case_ids,
                                restore=restore, ensemble=False, **kwargs)



def _create_postprocessor_from_plan(plan: dict) -> ONNXPostProcessor:
    """
    Create ONNX post-processor with exact PyTorch model parameters.
    """

    # Extract post-processing configuration from the  plan  architecture
    # to ensure exact compatibility with PyTorch model behavior.
    plan_arch = plan.get("architecture")
    if plan_arch is None:
        raise ValueError("Plan is missing 'architecture' section required for ONNX post-processing configuration")

    # Core model configuration from dataset and architecture planning
    num_classes = plan.get("num_classes")
    if num_classes is None:
        raise ValueError("Plan is missing 'num_classes' field required for ONNX post-processing")

    # Box coder weights must match the model's  spatial  dimensions.  In
    # torch, these are computed as (1.0,) *  (dim  *  2)  where  dim  is
    # spatial dimensions. For 3D models: (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    # for 2D: (1.0, 1.0, 1.0, 1.0)
    dim = plan_arch.get("dim")
    if dim is None:
        raise ValueError("Plan architecture is missing 'dim' field required for box coder configuration")
    coder_weights = (1.0,) * (dim * 2)  # Same logic as torch model

    classifier_type = "sigmoid"  # Standard for all current nnDetection models

    # Post-processing parameters from PyTorch model configuration. These
    # values may not be stored in the plan during training,  so  we  use
    # the same defaults as the PyTorch  model.  This  ensures  identical
    # behavior between ONNX and PyTorch. Default values are  taken  from
    # nndet/ptmodule/retinaunet/base.py
    param_defaults = {
        "topk_candidates": 10000,   # Pre-NMS candidate limit (PyTorch default)
        "score_thresh": 0,          # Minimal score threshold (PyTorch default)
        "remove_small_boxes": 0.01, # Small box removal threshold (PyTorch default)
        "nms_thresh": 0.6,          # NMS overlap threshold (PyTorch default)
        "detections_per_img": 100   # Final detection limit (PyTorch default)
    }

    # Extract parameters with defaults and log each parameter source
    param_values = {k: plan_arch.get(k, v) for k, v in param_defaults.items()}
    topk_candidates, score_thresh, remove_small_boxes, nms_thresh, detections_per_img = param_values.values()

    # Log each parameter individually with appropriate log level based on source
    for param_name, _ in param_defaults.items():
        actual_value = param_values[param_name]
        if param_name in plan_arch:
            logger.debug(f"Post-processing config - {param_name}: {actual_value} (plan)")
        else:
            logger.warning(f"Post-processing config - {param_name}: {actual_value} (default)")
    if missing_params := [k for k in param_defaults.keys() if k not in plan_arch]:
        logger.warning(f"Using PyTorch model defaults for missing plan parameters: {', '.join(missing_params)}")
        logger.warning("Consider updating your model plan to include explicit post-processing parameters")

    config = {
        "num_classes": num_classes,
        "classifier_type": classifier_type,
        "coder_weights": coder_weights,
        "topk_candidates": topk_candidates,
        "score_thresh": score_thresh,
        "remove_small_boxes": remove_small_boxes,
        "nms_thresh": nms_thresh,
        "detections_per_img": detections_per_img,
    }

    logger.debug(f"Post-processing config - num_classes: {num_classes}")
    logger.debug(f"Post-processing config - classifier_type: {classifier_type}")

    return create_postprocessor_from_config(config)


def _apply_onnx_postprocessing(onnx_outputs: list, image_shape: tuple,
                               postprocessor: ONNXPostProcessor) -> dict:
    """Apply post-processing to ONNX model outputs with error handling."""
    try:
        # Debug info for torch vs onnx comparison
        logger.debug(f"ONNX output count: {len(onnx_outputs)}")
        logger.debug(f"ONNX output shapes: {[out.shape for out in onnx_outputs]}")
        logger.debug(f"ONNX output dtypes: {[out.dtype for out in onnx_outputs]}")

        if len(onnx_outputs) >= 2:
            # Deltas and classification logits ranges can indicate model
            # health and help diagnose issues like  gradient  explosion,
            # poor training, or export problems.
            logger.debug(f"Box deltas range: [{onnx_outputs[0].min():.4f}, {onnx_outputs[0].max():.4f}]")
            logger.debug(f"Logits range: [{onnx_outputs[1].min():.4f}, {onnx_outputs[1].max():.4f}]")

            # Anchor count mismatches are a common source of ONNX export
            # issues. This happens when the ONNX export process  doesn't
            # perfectly replicate the anchor generation  logic,  leading
            # to dimension mismatches that cause inference  failures  or
            # incorrect results.
            if len(onnx_outputs) >= 3:
                anchor_count = onnx_outputs[2].shape[0]
                prediction_count = onnx_outputs[0].shape[0]
                logger.debug(f"Anchor count vs prediction count: {anchor_count} anchors, {prediction_count} predictions")

                if prediction_count != anchor_count:
                    logger.debug(f"Prediction/anchor count mismatch detected - using top-N selection")

        result = postprocessor.postprocess_onnx_output(onnx_outputs, image_shape)
        num_detections = len(result['pred_boxes'])

        # Provide insights into  model  performance  and  help  identify
        # potential issues with detection  quality,  score  calibration,
        # and spatial accuracy.
        if num_detections > 0:
            logger.debug(f"Final detections: {num_detections}")
            logger.debug(f"Score range: [{result['pred_scores'].min():.4f}, {result['pred_scores'].max():.4f}]")
            logger.debug(f"Box coordinate range: [{result['pred_boxes'].min():.1f}, {result['pred_boxes'].max():.1f}]")

            # Label distribution analysis helps verify that the model is
            # detecting the expected classes and that class  predictions
            # are not skewed.
            unique_labels, counts = np.unique(result['pred_labels'].astype(int), return_counts=True)
            logger.debug(f"Label distribution: {dict(zip(unique_labels, counts))}")

            # Box size analysis is critical for  medical  imaging  where
            # small lesions are common. This helps verify that small box
            # removal thresholds are working correctly and not filtering
            # out valid detections.
            if result['pred_boxes'].shape[1] == 6:  # 3D boxes (z1,y1,x1,z2,y2,x2)
                box_volumes = ((result['pred_boxes'][:, 3] - result['pred_boxes'][:, 0]) *
                              (result['pred_boxes'][:, 4] - result['pred_boxes'][:, 1]) *
                              (result['pred_boxes'][:, 5] - result['pred_boxes'][:, 2]))
                logger.debug(f"Box volume range: [{box_volumes.min():.1f}, {box_volumes.max():.1f}]")
            else:  # 2D boxes (y1,x1,y2,x2)
                box_areas = ((result['pred_boxes'][:, 2] - result['pred_boxes'][:, 0]) *
                            (result['pred_boxes'][:, 3] - result['pred_boxes'][:, 1]))
                logger.debug(f"Box area range: [{box_areas.min():.1f}, {box_areas.max():.1f}]")
        else:
            logger.debug("No detections found after post-processing")

        return result
    except Exception as e:
        logger.warning(f"Post-processing failed: {e}. Returning raw outputs.")
        return {
            "raw_outputs": onnx_outputs,
            "pred_boxes": onnx_outputs[0] if len(onnx_outputs) > 0 else np.array([]),
            "pred_scores": onnx_outputs[1] if len(onnx_outputs) > 1 else np.array([]),
            "pred_labels": np.array([])
        }

def _onnx_predict_common(source_dir, target_dir, onnx_model_paths, plan, case_ids=None,
                         restore=True, ensemble=False, **kwargs):
    """
    Core  ONNX  prediction  logic  handling  both  single  and  ensemble
    inference modes.
    """
    source_dir, target_dir = map(Path, (source_dir, target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)

    # The post-processor must be created before any inference to  ensure
    # consistent behavior across all models in an ensemble
    postprocessor = _create_postprocessor_from_plan(plan)

    # ONNX session initialization strategy differs based  on  prediction
    # mode. For ensembles, we maintain  separate  sessions  to  leverage
    # potential hardware optimizations per  model,  while  single  model
    # inference (batch case  as  well)  uses  a  simpler  single-session
    # approach for memory efficiency.
    logger.info(f"Initializing ONNX session(s) for {len(onnx_model_paths)} model(s)...")
    if ensemble:
        # each model gets its own session
        sessions = [ort.InferenceSession(str(p)) for p in onnx_model_paths]
        input_names = [s.get_inputs()[0].name for s in sessions]  # Handle different input names
    else:
        # Single session for individual model inference (batch case as well)
        session = ort.InferenceSession(str(onnx_model_paths[0]))
        input_name = session.get_inputs()[0].name

    # When case_ids is specified, we process only those specific  cases;
    # otherwise we process  all  available  cases.  Ground  truth  files
    # (_gt.npz) are excluded as they  contain  annotations  rather  than
    # input data for inference.
    case_paths = (
        [source_dir / f"{cid}.npz" for cid in case_ids]
        if case_ids is not None
        else [p for p in source_dir.glob('*.npz') if not p.name.endswith('_gt.npz')]
    )

    logger.info(f"Found {len(case_paths)} cases to process")
    if not case_paths:
        logger.warning(f"No cases found in {source_dir}")
        return

    # Process each case individually (per-case debugging enabled)
    for idx, path in enumerate(case_paths, 1):
        case_id = get_case_id_from_path(str(path), remove_modality=False)
        logger.info(f"Processing case {idx}/{len(case_paths)}: {case_id}")

        # Load case data using the same logic as PyTorch  pipeline.  The
        # 'data' key is preferred when  available  (structured  format),
        # but we fall back to  the  raw  array  for  compatibility  with
        # different preprocessing outputs.
        arr = np.load(str(path), allow_pickle=True)
        case = arr['data'] if 'data' in arr else arr

        # Properties file contains metadata for spatial restoration  and
        # output formatting. The transpose_backward flag is  added  from
        # the plan.
        properties = load_pickle(path.parent / f"{case_id}.pkl")
        properties["transpose_backward"] = plan["transpose_backward"]

        # Debug input data characteristics
        logger.debug(f"Input data shape: {case.shape}, dtype: {case.dtype}")
        logger.debug(f"Input data range: [{case.min():.4f}, {case.max():.4f}]")
        logger.debug(f"Input data mean: {case.mean():.4f}, std: {case.std():.4f}")

        # Ensure proper tensor dimensions
        case = case[None] if case.ndim == 4 else case  # Add batch dim if missing
        if case.ndim != 5:
            raise ValueError(f"Unexpected input shape: {case.shape}. "
                           "Expected 4D (C,D,H,W) or 5D (B,C,D,H,W) tensor.")

        image_shape = case.shape[2:]  # (D, H, W) or (H, W)

        # These values are critical for  ensuring  that  detections  are
        # correctly mapped back to original image space.
        logger.debug(f"Original size: {properties.get('original_size_of_raw_data', 'unknown')}")
        logger.debug(f"ITK spacing: {properties.get('itk_spacing', 'unknown')}")
        logger.debug(f"Transpose backward: {properties.get('transpose_backward', 'unknown')}")
        logger.info(f"Input shape: {case.shape}, Image shape: {image_shape}")

        if ensemble:
            # Ensemble inference strategy: run each model independently,
            # then combine results through concatenation and final  NMS.

            # Note: doing it in parallel would be  more  efficient,  but
            # requires  careful  handling   of   session   outputs   and
            # post-processing (will be implemented later)
            logger.debug(f"Running ensemble inference with {len(sessions)} models...")
            model_results = []

            for i, (s, inp) in enumerate(zip(sessions, input_names)):
                logger.debug(f"Processing with model {i+1}/{len(sessions)}...")

                # Ensures  poor  predictions  from   one   model   don't
                # contaminate the ensemble before individual  model  NMS
                # has  filtered   out   low-quality   detections   (runs
                # independently with its own post-processing)
                outputs = s.run(None, {inp: case.astype(np.float32)})

                # Batch dimension  removal  is  necessary  because  ONNX
                # models  output   with   batch   dimension   even   for
                # single-sample inference.
                outputs = [out[0] if out.shape[0] == 1 else out for out in outputs]
                model_result = _apply_onnx_postprocessing(outputs, image_shape, postprocessor)
                model_results.append(model_result)

            logger.debug(f"Combining results from {len(model_results)} models...")

            # Ensemble   combination   strategy:   collect   all   valid
            # detections from individual models and apply  a  final  NMS
            # pass.  This  allows  the  ensemble  to  benefit  from  the
            # strengths  of  each   model   while   removing   redundant
            # detections across the ensemble.
            all_boxes, all_scores, all_labels = [], [], []
            for model_result in model_results:
                if len(model_result["pred_boxes"]) > 0:
                    all_boxes.append(model_result["pred_boxes"])
                    all_scores.append(model_result["pred_scores"])
                    all_labels.append(model_result["pred_labels"])

            if all_boxes:
                # Concatenate all model outputs into unified arrays  for
                # ensemble NMS
                ensemble_boxes = np.concatenate(all_boxes, axis=0)
                ensemble_scores = np.concatenate(all_scores, axis=0)
                ensemble_labels = np.concatenate(all_labels, axis=0)

                # remove  duplicates  while   preserving   complementary
                # detections from different models.  This  threshold  is
                # higher than individual model NMS to allow  for  slight
                # spatial variations between models.
                from nndet.inference.onnx_postprocess import batched_nms
                keep_indices = batched_nms(ensemble_boxes, ensemble_scores, ensemble_labels, 0.5)

                # Limit the number of detections per image
                if len(keep_indices) > postprocessor.detections_per_img:
                    logger.warning(f"Limiting detections to {postprocessor.detections_per_img} per image")
                    sorted_indices = np.argsort(ensemble_scores[keep_indices])[::-1]
                    keep_indices = keep_indices[sorted_indices[:postprocessor.detections_per_img]]

                result = {
                    "pred_boxes": ensemble_boxes[keep_indices].astype(np.float32),
                    "pred_scores": ensemble_scores[keep_indices].astype(np.float32),
                    "pred_labels": ensemble_labels[keep_indices].astype(np.float32)
                }
            else:
                # handle ensemble case where  no  model  produced  valid
                # detections
                result = {
                    "pred_boxes": np.zeros((0, 6), dtype=np.float32),
                    "pred_scores": np.zeros(0, dtype=np.float32),
                    "pred_labels": np.zeros(0, dtype=np.float32)
                }
        else:
            # Single model prediction
            logger.debug(f"Running ONNX inference...")
            onnx_outputs = session.run(None, {input_name: case.astype(np.float32)})

            # Output shape analysis before  and  after  batch  dimension
            # removal
            logger.debug(f"Raw ONNX output shapes: {[out.shape for out in onnx_outputs]}")
            # Remove   batch   dimension   to   match   post-processing
            # expectations
            onnx_outputs = [out[0] if out.shape[0] == 1 else out for out in onnx_outputs]
            logger.debug(f"Post-batch-removal shapes: {[out.shape for out in onnx_outputs]}")

            result = _apply_onnx_postprocessing(onnx_outputs, image_shape, postprocessor)

        # Result structure must exactly match PyTorch pipeline output
        result.update({
            "restore": restore,  # enable/disable spatial restoration
            "original_size_of_raw_data": properties["original_size_of_raw_data"],
            "itk_origin": properties["itk_origin"],
            "itk_spacing": properties["itk_spacing"],
            "itk_direction": properties["itk_direction"],
        })

        # File naming convention matches PyTorch exactly: _boxes.pkl for
        # detections and _properties.pkl for metadata
        save_pickle(result, target_dir / f"{case_id}_boxes.pkl")
        save_pickle(properties, target_dir / f"{case_id}_properties.pkl")

        num_detections = len(result.get('pred_boxes', []))
        # Summary info for torch vs onnx comparison
        if num_detections > 0:
            max_score = result['pred_scores'].max()
            logger.debug(f"Highest confidence detection: {max_score:.4f}")
        logger.info(f"Case {case_id} completed: {num_detections} detections")

    logger.info(f"ONNX prediction completed. Processed {len(case_paths)} cases -> {target_dir}")