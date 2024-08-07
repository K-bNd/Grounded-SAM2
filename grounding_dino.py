import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from detections import from_transformers, annotate_transformers


def predict(image, text: str, model_id: str = "IDEA-Research/grounding-dino-tiny"):
    results = get_detections(image, text, model_id=model_id)
    detections_results = from_transformers(results)
    return detections_results


def get_detections(
    image, text: str, model_id: str = "IDEA-Research/grounding-dino-tiny"
):

    # VERY important: text queries need to be lowercased + end with a dot
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    return results


if __name__ == "__main__":
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    detections_results = predict(image, "a cat. a remote control.")
    annotated_frame = annotate_transformers(image, detections_results[0])
    annotated_frame.show()
