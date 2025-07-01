
import numpy as np
import onnxruntime as ort
from pathlib import Path
from loguru import logger
from nndet.io.load import load_pickle, save_pickle
from nndet.io.paths import get_case_id_from_path


def onnx_predict_ensemble_dir(source_dir, target_dir, onnx_model_paths, plan, case_ids=None,
                              restore=True, **kwargs):
    """Entry point for ensemble prediction - delegates to shared implementation"""

    return _onnx_predict_common(source_dir, target_dir, onnx_model_paths, plan, case_ids=case_ids,
                                restore=restore, ensemble=True, **kwargs)

def onnx_predict_dir(source_dir, target_dir, onnx_model_path, plan, case_ids=None,
                     restore=True, **kwargs):
    """Entry point for single model prediction - wraps path in list for shared impl"""

    return _onnx_predict_common(source_dir, target_dir, [onnx_model_path], plan, case_ids=case_ids,
                                restore=restore, ensemble=False, **kwargs)



def _onnx_predict_common(source_dir, target_dir, onnx_model_paths, plan, case_ids=None,
                         restore=True, ensemble=False, **kwargs):
    """
    Shared ONNX prediction logic. If ensemble=True, averages predictions from all models.
    """

    source_dir, target_dir = map(Path, (source_dir, target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)

    if ensemble:
        logger.info(f"Using ONNX ensemble: {onnx_model_paths}")
        sessions = [ort.InferenceSession(str(p)) for p in onnx_model_paths]
        input_names = [s.get_inputs()[0].name for s in sessions]
        output_names = [s.get_outputs()[0].name for s in sessions]
    else:
        logger.info(f"Using ONNX model: {onnx_model_paths[0]}")
        session = ort.InferenceSession(str(onnx_model_paths[0]))
        input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name

    case_paths = (
        [source_dir / f"{cid}.npz" for cid in case_ids]
        if case_ids is not None
        # Excludes ground truth files (_gt.npz) automatically
        else [p for p in source_dir.glob('*.npz') if not p.name.endswith('_gt.npz')]
    )
    logger.info(f"Found {len(case_paths)} files for ONNX{' ensemble' if ensemble else ''} inference.")

    for idx, path in enumerate(case_paths, 1):
        case_id = get_case_id_from_path(str(path), remove_modality=False)
        logger.info(f"ONNX{' Ensemble' if ensemble else ''} Predicting case {idx}/{len(case_paths)}: {case_id}")

        arr = np.load(str(path), allow_pickle=True)
        case = arr['data'] if 'data' in arr else arr

        properties = load_pickle(path.parent / f"{case_id}.pkl")
        properties["transpose_backward"] = plan["transpose_backward"]
        case = case[None] if case.ndim == 4 else case

        if case.ndim != 5:
            raise ValueError(
                f"Unexpected input shape: {case.shape}. "
                "Expected 4D (C,D,H,W) or 5D (B,C,D,H,W) tensor."
            )
        if ensemble:
            # Run each model and stack results before averaging
            preds = [s.run([out], {inp: case.astype(np.float32)})[0]
                     for s, inp, out in zip(sessions, input_names, output_names)]
            result = np.mean(preds, axis=0) # Simple average ensemble
        else:
            result = session.run([output_name], {input_name: case.astype(np.float32)})[0]

         # Remove batch dimension if present before saving
        result = result[0] if result.shape[0] == 1 else result

        save_pickle(result, target_dir / f"{case_id}_pred.pkl")
        save_pickle(properties, target_dir / f"{case_id}_properties.pkl")

    logger.info(f"ONNX{' ensemble' if ensemble else ''} prediction finished.")