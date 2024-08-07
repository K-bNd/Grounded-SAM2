from segment_anything_2.sam2 import (
    build_sam,
    sam2_image_predictor,
    sam2_video_predictor,
)
from grounding_dino import predict
from PIL.Image import Image
from supervision import Detections


def predict_image(
    image: Image,
    sam2_checkpoint: str,
    model_cfg: str,
    detections: list[Detections],
):
    sam2_model = build_sam.build_sam2(model_cfg, sam2_checkpoint)
    predictor = sam2_image_predictor.SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    output_masks = []
    output_scores = []
    output_logits = []

    for detection in detections:
        masks, scores, logits = predictor.predict(box=detection.xyxy)
        output_masks.append(masks)
        output_scores.append(scores)
        output_logits.append(logits)
    return output_masks, output_scores, output_logits
