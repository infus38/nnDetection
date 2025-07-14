import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class ONNXBoxCoder:
    """
    Box coder for decoding relative offsets to absolute box coordinates.
    """

    def __init__(self, weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 bbox_xform_clip: float = np.log(1000.0 / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode_single(self, rel_codes: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Decode relative codes to absolute box coordinates.
        """

        if boxes.shape[1] == 4:
            return self._decode_2d(rel_codes, boxes)
        elif boxes.shape[1] == 6:
            return self._decode_3d(rel_codes, boxes)
        else:
            raise ValueError(f"Unsupported box dimension: {boxes.shape[1]}")

    def _decode_2d(self, rel_codes: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Decode 2D boxes using the standard R-CNN transformation equations.
        """
        wx, wy, ww, wh = self.weights[:4]

        # Extract anchor properties: center points and dimensions  (same
        # sequence as nndet/core/boxes/coder.py:decode_single())
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        # Apply weight normalization to deltas
        dx, dy = rel_codes[:, 0] / wx, rel_codes[:, 1] / wy

        # Clip size deltas to  prevent  exponential  explosion  (matches
        # PyTorch torch.clamp in nndet/core/boxes/coder.py)
        dw = np.clip(rel_codes[:, 2] / ww, a_max=self.bbox_xform_clip, a_min=None)
        dh = np.clip(rel_codes[:, 3] / wh, a_max=self.bbox_xform_clip, a_min=None)

        # Transform from center+size back to corner coordinates
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = np.exp(dw) * widths  # Exponential for positive sizes
        pred_h = np.exp(dh) * heights

        return np.stack([
            pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h
        ], axis=1)

    def _decode_3d(self, rel_codes: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Decode 3D boxes extending 2D principles  to  volumetric  medical
        images.
        """
        wx, wy, ww, wh, wz, wd = self.weights[:6]

        # Extract 3D anchor properties:  center  points  and  dimensions
        # (same   pattern   as   2D   but   with   depth    -    matches
        # nndet/core/boxes/coder.py)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        depths = boxes[:, 5] - boxes[:, 4]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        ctr_z = boxes[:, 4] + 0.5 * depths

        # Apply weight normalization to all 6 deltas (x,y,z positions  +
        # w,h,d sizes)
        dx, dy, dz = rel_codes[:, 0] / wx, rel_codes[:, 1] / wy, rel_codes[:, 4] / wz
        # Clip all size deltas to prevent exponential explosion  in  any
        # dimension     (replicates     PyTorch     torch.clamp     from
        # nndet/core/boxes/coder.py)
        dw = np.clip(rel_codes[:, 2] / ww, a_max=self.bbox_xform_clip, a_min=None)
        dh = np.clip(rel_codes[:, 3] / wh, a_max=self.bbox_xform_clip, a_min=None)
        dd = np.clip(rel_codes[:, 5] / wd, a_max=self.bbox_xform_clip, a_min=None)

        # Transform 3D coordinates from center+size back to corner format
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_ctr_z = dz * depths + ctr_z
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights
        pred_d = np.exp(dd) * depths

        return np.stack([
            pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h,
            pred_ctr_z - 0.5 * pred_d, pred_ctr_z + 0.5 * pred_d
        ], axis=1)


class ONNXClassifier:
    """
    Classifier for converting classification logits to probabilities.
    """

    def __init__(self, classifier_type: str = "sigmoid"):
        """
        initialize classifier with specified activation function.
        """

        if classifier_type not in ["sigmoid", "softmax"]:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        self.classifier_type = classifier_type

    def box_logits_to_probs(self, box_logits: np.ndarray) -> np.ndarray:
        """
        Convert classification logits to calibrated probability scores.
        """

        if self.classifier_type == "sigmoid":
            # Standard sigmoid activation - matches PyTorch training behavior
            probs = 1.0 / (1.0 + np.exp(-box_logits))
            return probs[:, 1:] if probs.shape[1] > 1 else probs

        else:  # softmax

            # Subtract max for numerical stability -  prevents  overflow
            # in exp() (same as PyTorch F.softmax implementation)
            shifted_logits = box_logits - np.max(box_logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Skip background class (index 0) - only  return  foreground
            # probabilities
            return probs[:, 1:]


def clip_boxes_to_image(boxes: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Clip bounding boxes to image  boundaries  to  prevent  out-of-bounds
    coordinates.
    """

    if len(image_shape) == 2:  # 2D image: (height, width)
        h, w = image_shape

        # Clip x coordinates (width dimension)  to  [0,  width-1]  (same
        # boundary clipping as
        # nndet/core/boxes/clip.py:clip_boxes_to_image_2d_())
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)  # x1 (left)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)  # x2 (right)

        # Clip y coordinates (height dimension) to [0, height-1]
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)  # y1 (top)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)  # y2 (bottom)

    elif len(image_shape) == 3:  # 3D image: (depth, height, width)
        d, h, w = image_shape

        # Clip x coordinates (width dimension) to [0,  width-1]  extends
        # 2D clipping to 3D
        # (same as nndet/core/boxes/clip.py:clip_boxes_to_image_3d_())
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)  # x1 (left)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)  # x2 (right)

        # Clip y coordinates (height dimension) to [0, height-1]
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)  # y1 (top)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)  # y2 (bottom)

        # Clip z coordinates (depth dimension) to [0, depth-1]
        boxes[:, 4] = np.clip(boxes[:, 4], 0, d - 1)  # z1 (front/superior)
        boxes[:, 5] = np.clip(boxes[:, 5], 0, d - 1)  # z2 (back/inferior)

    else:
        raise ValueError(f"Unsupported image shape: {image_shape}")

    # Return modified boxes array with coordinates safely bounded
    return boxes


def remove_small_boxes(boxes: np.ndarray, min_size: float) -> np.ndarray:
    """
    Remove boxes smaller than  minimum  size  threshold  to  filter  out
    low-quality detections.
    """

    if boxes.shape[1] == 4:  # 2D boxes: [x1, y1, x2, y2]
        widths = boxes[:, 2] - boxes[:, 0]   # x2 - x1
        heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1

        # Use minimum dimension rather than area to catch thin artifacts
        # (same approach as PyTorch training - filters based on smallest
        # dimension)
        min_sizes = np.minimum(widths, heights)

    elif boxes.shape[1] == 6:  # 3D boxes: [x1, y1, x2, y2, z1, z2]
        widths = boxes[:, 2] - boxes[:, 0]   # x2 - x1
        heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1
        depths = boxes[:, 5] - boxes[:, 4]   # z2 - z1
        # Minimum across all three dimensions
        min_sizes = np.minimum(np.minimum(widths, heights), depths)

    else:
        raise ValueError(f"Unsupported box shape: {boxes.shape[1]}")

    # Return boolean mask for boxes meeting size requirement
    return min_sizes >= min_size


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) between two sets of boxes.
    """

    # Dispatch to 2D or 3D  implementation  based  on  coordinate  count
    # (same dispatching logic as nndet/core/boxes/ops.py:box_iou())
    return box_iou_2d(boxes1, boxes2) if boxes1.shape[1] == 4 else box_iou_3d(boxes1, boxes2)


def box_iou_2d(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute 2D Intersection over Union between two sets of boxes.
    """

    # Compute areas once, reuse for all pairwise calculations
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Find intersection coordinates using broadcasting ([:, 0:1] creates
    # [N,1] shape, boxes2[:, 0] is  [M]  shape  ->  [N,M]  result)  Same
    # broadcasting approach as nndet/core/boxes/ops_np.py:box_iou_2d()
    inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])  # Shape: [N, M]
    inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])  # Shape: [N, M]
    inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])  # Shape: [N, M]
    inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])  # Shape: [N, M]

    # Zero area if no overlap in either dimension
    inter_area = (np.maximum(0, inter_x2 - inter_x1) *
                  np.maximum(0, inter_y2 - inter_y1))

    # IoU = intersection / union, with epsilon to avoid division by zero
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


def box_iou_3d(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute 3D Intersection over Union between two  sets  of  volumetric
    boxes.
    """

    # Compute 3D volumes (width × height × depth)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) *
             (boxes1[:, 3] - boxes1[:, 1]) *
             (boxes1[:, 5] - boxes1[:, 4]))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) *
             (boxes2[:, 3] - boxes2[:, 1]) *
             (boxes2[:, 5] - boxes2[:, 4]))

    # Find 3D intersection coordinates using  broadcasting  (extends  2D
    # intersection  logic  to  include   depth   dimension   -   matches
    # nndet/core/boxes/ops_np.py:box_iou_3d())
    inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    inter_z1 = np.maximum(boxes1[:, 4:5], boxes2[:, 4])  # Depth minimum
    inter_z2 = np.minimum(boxes1[:, 5:6], boxes2[:, 5])  # Depth maximum

    # Zero volume if no overlap in any dimension
    inter_area = (np.maximum(0, inter_x2 - inter_x1) *
                  np.maximum(0, inter_y2 - inter_y1) *
                  np.maximum(0, inter_z2 - inter_z1))

    # IoU = intersection / union
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to eliminate duplicate detections.
    """

    if len(scores) == 0:
        # Early exit for empty input - return empty array  with  correct
        # dtype
        return np.array([], dtype=np.int64)

    # Process  detections  in  descending  score  order   (same   greedy
    # algorithm as nndet/core/boxes/nms.py:nms_cpu())
    sorted_indices = np.argsort(scores)[::-1]
    keep = []

    while len(sorted_indices) > 0:
        # Always keep the highest scoring remaining detection
        current = sorted_indices[0]
        keep.append(current)

        if len(sorted_indices) == 1:
            # Only one detection left - no more comparisons needed
            break

        # Check IoU between current box and all remaining boxes
        current_box = boxes[current:current+1]       # Shape: [1, 4/6]
        remaining_boxes = boxes[sorted_indices[1:]]  # Shape: [M-1, 4/6]
        ious = box_iou(current_box, remaining_boxes).flatten()

        # Keep only boxes with low overlap (below threshold)
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    # Convert  list  to  numpy  array  with  int64  dtype  for  indexing
    # compatibility
    return np.array(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                iou_threshold: float) -> np.ndarray:
    """
    Apply class-aware Non-Maximum Suppression (batched NMS).
    """

    if len(boxes) == 0:
        # Early exit for empty input to maintain consistent return type
        return np.array([], dtype=np.int64)

    # Spatial offset trick: shift  each  class  to  separate  coordinate
    # space (this makes cross-class IoU =  0,  so  NMS  only  suppresses
    # within classes) Same  trick  used  in  torchvision.ops.batched_nms
    # implementation
    max_coordinate = np.max(boxes)
    offsets = labels * (max_coordinate + 1)
    offset_boxes = boxes + offsets[:, None]

    # Run  standard  NMS  on  offset  boxes  -   automatically   becomes
    # class-aware (leverages existing NMS  implementation  with  spatial
    # separation trick)
    return nms(offset_boxes, scores, iou_threshold)


class ONNXPostProcessor:
    """
    ONNX post-processing pipeline that  replicates  nndet  torch  models
    behavior.
    """

    def __init__(self,
                 num_foreground_classes: int = 1,
                 classifier_type: str = "sigmoid",
                 coder_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 topk_candidates: Optional[int] = 1000,
                 score_thresh: Optional[float] = 0.05,
                 remove_small_boxes_thresh: Optional[float] = 1e-2,
                 nms_thresh: float = 0.5,
                 detections_per_img: Optional[int] = 100):

        self.num_foreground_classes = num_foreground_classes
        self.topk_candidates = topk_candidates
        self.score_thresh = score_thresh
        self.remove_small_boxes_thresh = remove_small_boxes_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.coder = ONNXBoxCoder(weights=coder_weights)
        self.classifier = ONNXClassifier(classifier_type=classifier_type)

    def postprocess_detections_single_image(self,  boxes: np.ndarray, probs: np.ndarray, image_shape: Tuple[int, ...]
                                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process detections for a  single  image
        """

        if len(boxes) == 0:
            # Return  properly  shaped  empty  arrays  to  maintain  API
            # consistency
            empty_shape = (0, boxes.shape[1]) if len(boxes.shape) > 1 else (0, 4)
            return (np.zeros(empty_shape, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int64))

        # Step 1: Ensure all boxes are within image boundaries
        boxes = clip_boxes_to_image(boxes, image_shape)

        # Step  2:   Select   top   candidates   by   flattening   class
        # probabilities
        probs_flat = probs.flatten()
        if self.topk_candidates is not None:
            num_topk = min(self.topk_candidates, len(probs_flat))

            # Use partial sort to find top-k, then sort those for proper
            # ordering
            top_indices = np.argpartition(probs_flat, -num_topk)[-num_topk:]
            sorted_indices = top_indices[np.argsort(probs_flat[top_indices])[::-1]]
        else:
            sorted_indices = np.argsort(probs_flat)[::-1]

        probs_flat = probs_flat[sorted_indices]
        idx = sorted_indices

        # Step 3: Apply confidence threshold
        if self.score_thresh is not None:
            keep_idxs = probs_flat > self.score_thresh
            probs_flat = probs_flat[keep_idxs]
            idx = idx[keep_idxs]

        # Early exit if no detections pass confidence threshold
        if len(idx) == 0:
            empty_shape = (0, boxes.shape[1])
            return (np.zeros(empty_shape, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int64))

        # Step 4: Convert flat indices back to  (anchor_idx,  class_idx)
        # pairs  (reverses  the  flattening  done   during   probability
        # computation - same as PyTorch training)
        anchor_idxs = idx // self.num_foreground_classes
        labels = idx % self.num_foreground_classes
        boxes = boxes[anchor_idxs]

        # Step 5: Remove geometrically invalid small boxes
        if self.remove_small_boxes_thresh is not None:
            keep = remove_small_boxes(boxes, self.remove_small_boxes_thresh)
            boxes = boxes[keep]
            probs_flat = probs_flat[keep]
            labels = labels[keep]

        # Early exit if no boxes survive small box removal
        if len(boxes) == 0:
            empty_shape = (0, boxes.shape[1])
            return (np.zeros(empty_shape, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int64))

        # Step 6: Apply class-aware NMS to remove duplicate detections
        keep = batched_nms(boxes, probs_flat, labels, self.nms_thresh)

        # Step 7: Limit final detection count if specified
        if self.detections_per_img is not None:
            keep = keep[:self.detections_per_img]

        return boxes[keep], probs_flat[keep], labels[keep]

    def postprocess_onnx_output(self, onnx_outputs: List[np.ndarray],
                                image_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:

        # [CRITICAL] Validate input format to  prevent  silent  failures
        # downstream
        if len(onnx_outputs) < 3:
            raise ValueError(f"Expected at least 3 ONNX outputs, got {len(onnx_outputs)}")

        box_deltas, box_logits, anchors = onnx_outputs[:3]

        if box_deltas.shape[0] != anchors.shape[0]:
            # Handle shape mismatches that can occur during ONNX export
            if box_deltas.shape[0] > anchors.shape[0]:
                # Too many  predictions  -  select  top  predictions  by
                # confidence   This   preserves   the    highest-quality
                # detections when ONNX  export  changes  the  prediction
                # tensor size but keeps anchor count fixed

                all_probs = self.classifier.box_logits_to_probs(box_logits)
                # Handle both multi-class and single-class scenarios
                all_probs_max = (np.max(all_probs, axis=1) if all_probs.shape[1] > 1
                               else all_probs.flatten())

                num_anchors = anchors.shape[0]
                # Use partial sorting for efficiency - only need  top  N
                # indices
                top_indices = np.argpartition(all_probs_max, -num_anchors)[-num_anchors:]
                # Sort the selected indices  by  confidence  for  proper
                # ordering
                top_indices = top_indices[np.argsort(all_probs_max[top_indices])[::-1]]

                box_deltas = box_deltas[top_indices]
                box_logits = box_logits[top_indices]
            else:
                # Too many  anchors  -  truncate  anchor  set  to  match
                # predictions  This  handles  cases  where  ONNX  export
                # reduces prediction tensor but keeps all anchor layers
                anchors = anchors[:box_deltas.shape[0]]

        # Apply box decoding to  convert  relative  deltas  to  absolute
        # coordinates
        pred_boxes = self.coder.decode_single(box_deltas, anchors)

        # Convert classification logits to calibrated probability scores
        pred_probs = self.classifier.box_logits_to_probs(box_logits)

        # Execute complete post-processing pipeline
        final_boxes, final_scores, final_labels = self.postprocess_detections_single_image(
            pred_boxes, pred_probs, image_shape
        )

        return {
            "pred_boxes": final_boxes.astype(np.float32),
            "pred_scores": final_scores.astype(np.float32),
            "pred_labels": final_labels.astype(np.float32)
        }


def create_postprocessor_from_config(config: Dict[str, Any]) -> ONNXPostProcessor:
    """
    Create ONNX post-processor from model configuration dictionary.
    """

    return ONNXPostProcessor(
        num_foreground_classes=config.get("num_classes", 1),
        classifier_type=config.get("classifier_type", "sigmoid"),
        coder_weights=config.get("coder_weights", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        topk_candidates=config.get("topk_candidates"),  # None by default
        score_thresh=config.get("score_thresh", 0.05),
        remove_small_boxes_thresh=config.get("remove_small_boxes", 1e-2),
        nms_thresh=config.get("nms_thresh", 0.5),
        detections_per_img=config.get("detections_per_img", 100)
    )