"""
Mangrove detection and survival-status classification, returning structured results.

Replaces the annotate-then-upload flow in inference_light.py. The caller passes an
image already in memory; this module performs NO disk I/O and renders no annotated
image. It returns bounding boxes and per-detection probabilities so the client can
draw its own visualisation, which means the photo is never stored a second time.

Efficiency notes, relative to inference_light.run_survivability_inference():
  * both models are loaded ONCE per process, not once per request -- the previous
    flow re-read the YOLO weights and the Keras SavedModel on every call
  * no image is written to or read from disk at any point
  * detections are assembled as plain lists rather than by concatenating a pandas
    DataFrame per detection inside the loop
  * the annotated-image render (utils.plot_boxes) is gone

The detection and cropping maths is unchanged from inference_light, so results are
identical to the previous endpoint; only the transport and the plumbing differ.
"""

import time
import uuid

import numpy as np

from PIL import ImageOps
from tensorflow import keras
from ultralytics import YOLO

from src.libs import utils
from src.utils import context


config = context.config
logger = context.logger


# Ordered to match the classifier's output columns.
CLASS_LABELS = ('alive', 'dead', 'unclear')


_detection_model = None
_classification_model = None


def load_models():
    """
    Load the YOLO detector and Keras classifier once per process.

    Idempotent: later calls return the already-loaded models. Call at application
    startup to move the cost off the first request.
    """

    global _detection_model, _classification_model

    if _detection_model is not None and _classification_model is not None:
        return _detection_model, _classification_model

    detection_path = context.root_dir + config.pipeline.object_model_path
    logger.info(f"survivability_detection loading detector: path={detection_path}")
    _detection_model = YOLO(detection_path)

    classification_path = context.root_dir + config.pipeline.survival_model_path
    logger.info(f"survivability_detection loading classifier: path={classification_path}")
    _classification_model = keras.models.load_model(classification_path)

    logger.info("survivability_detection models loaded")

    return _detection_model, _classification_model


def _classify_crops(image, boxes_xywhn):
    """
    Crop each detection out of the image and classify the batch in one pass.

    Cropping uses the same expand/pad/resize preprocessing the classifier was
    trained with, so predictions match the previous implementation.

    :param image: PIL image, already EXIF-oriented
    :param boxes_xywhn: list of normalised [x_center, y_center, w, h] boxes
    :return: ndarray of shape (n_detections, 3) -- P(alive), P(dead), P(unclear)
    """

    expanded = ImageOps.expand(image, border=config.pipeline_meta.border)

    crops = []
    for bbox in boxes_xywhn:
        crop = utils.crop_image(expanded, bbox,
                               config.pipeline_meta.border,
                               config.pipeline_meta.pad_fraction,
                               config.pipeline_meta.min_size)

        crop = utils.transform_img_dimensions(crop,
                                              config.pipeline_meta.input_width,
                                              config.pipeline_meta.input_height)

        crops.append(keras.utils.img_to_array(crop))

    _, classification_model = load_models()

    return classification_model(np.array(crops)).numpy()


def detect_survivability(image, session_id=None):
    """
    Detect mangroves in an image and classify each as alive, dead or unclear.

    :param image: PIL image (typically opened from S3 bytes via io.BytesIO)
    :param session_id: correlation id for logging; generated when omitted
    :return: dict with image dimensions, per-detection boxes and probabilities,
             and the aggregate counts
    """

    session_id = session_id or str(uuid.uuid4())
    start_time_clock = time.time()

    logger.info(f"survivability_detection started: session_id={session_id}")

    # Honour EXIF orientation in memory. utils.rotate_image cannot be used here: its
    # except branch reads image.filename, which does not exist on an image opened
    # from a BytesIO buffer. exif_transpose also covers all eight orientations
    # rather than the three that helper handles.
    image = ImageOps.exif_transpose(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size

    detection_model, _ = load_models()
    results = detection_model(image, conf=config.pipeline.confidence, verbose=False)[0]

    detections = []
    counts = {label: 0 for label in CLASS_LABELS}

    if len(results) > 0:
        logger.info(f"survivability_detection detected: session_id={session_id} mangroves={len(results)}")

        boxes_xywhn = results.boxes.xywhn.tolist()
        boxes_xyxy = results.boxes.xyxy.tolist()
        confidences = results.boxes.conf.tolist()

        probabilities = _classify_crops(image, boxes_xywhn)

        for index, (xywhn, xyxy, confidence, probs) in enumerate(
                zip(boxes_xywhn, boxes_xyxy, confidences, probabilities)):

            status = CLASS_LABELS[int(np.argmax(probs))]
            counts[status] += 1

            detections.append({
                "detection_id": str(uuid.uuid4()),
                "index": index,
                "status": status,
                "detection_confidence": round(float(confidence), 4),
                # absolute pixels [x1, y1, x2, y2] -- draw directly on the source image
                "bbox_xyxy": [round(float(v), 2) for v in xyxy],
                # normalised [x_center, y_center, w, h] -- resolution independent
                "bbox_xywhn": [round(float(v), 6) for v in xywhn],
                "probabilities": {
                    label: round(float(probs[i]), 4)
                    for i, label in enumerate(CLASS_LABELS)
                },
            })
    else:
        logger.warning(f"survivability_detection found nothing: session_id={session_id} "
                       f"threshold {config.pipeline.confidence}")

    payload = {
        "image": {"width": width, "height": height},
        "counts": {
            "number_mangroves": len(detections),
            "number_alive": counts['alive'],
            "number_dead": counts['dead'],
            "number_unclear": counts['unclear'],
        },
        "detections": detections,
    }

    duration = time.time() - start_time_clock
    counts = payload['counts']
    logger.info(f"survivability_detection finished: session_id={session_id} "
                f"mangroves={counts['number_mangroves']} alive={counts['number_alive']} "
                f"dead={counts['number_dead']} unclear={counts['number_unclear']} "
                f"duration={duration:.2f}s")

    return payload
