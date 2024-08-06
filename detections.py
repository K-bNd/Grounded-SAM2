from supervision import Detections, BoxAnnotator, LabelAnnotator
import numpy as np


def from_transformers(transformers_results: list[dict]):
    def create_detection(transformers_result):
        data = {}
        data["class_name"] = np.array(transformers_result["labels"])
        return Detections(
            xyxy=transformers_result["boxes"].cpu().numpy(),
            confidence=transformers_result["scores"].cpu().numpy(),
            data=data,
            class_id=create_class_id_lookup(data["class_name"]),
        )

    return [create_detection(result) for result in transformers_results]


def create_class_id_lookup(labels):
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    lookup = np.array([label_to_id[label] for label in labels], dtype=np.uint8)
    return lookup


def annotate_transformers(image, detections_result: Detections):
    box_annotator = BoxAnnotator()
    label_annotator = LabelAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections_result,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections_result,
        labels=detections_result.data["class_name"],
    )
    return annotated_frame
