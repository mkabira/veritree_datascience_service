"""
Image preprocessing for the survivability classifier.

Crops each YOLO detection out of the source image and reshapes it to the input
geometry the Keras classifier was trained on. Consumed by
src/services/computer_vision/survivability_detection.py.

Trimmed from the veritree-survivability original, which also carried batch-pipeline
helpers (directory walking, result plotting, SNS notifications, one-hot decoding) --
all of which went with the batch tasks and the annotate-and-upload endpoints.
"""

import numpy as np

from PIL import ImageOps

from src.utils import context


config = context.config
logger = context.logger


def get_box_coords(image, label, pad_fraction, border):
    """
    Compute the pixel corners and centre of a normalised bounding box.

    :param image: PIL image the box refers to, already zero-padded by ``border``
    :param label: normalised YOLO box as [x_center, y_center, width, height]
    :param pad_fraction: fraction of the box's own dimensions to expand each side by
    :param border: zero-padding in pixels previously added around the image
    :return: dict of left/right/top/bottom pixel coordinates plus x_center/y_center
    """
    box_coords = {}
    pix_width = image.size[0] - 2 * border
    pix_height = image.size[1] - 2 * border
    # pix_width, pix_height = image.size
    x_center = label[0]
    box_coords['x_center'] = x_center
    y_center = label[1]
    box_coords['y_center'] = x_center
    width = label[2]
    height = label[3]
    # getting box corner coordinates in pixels
    box_coords['left'] = round(pix_width * (x_center - width / 2 - pad_fraction * width) + border)
    box_coords['right'] = round(pix_width * (x_center + width / 2 + pad_fraction * width) + border)
    box_coords['bottom'] = round(pix_height * (y_center - height / 2 - pad_fraction * height) + border)
    box_coords['top'] = round(pix_height * (y_center + height / 2 + pad_fraction * height) + border)

    return box_coords


def crop_image(image, bbox, border, pad_fraction, min_size):
    """
    Crop a detection out of the image, with padding, to the classifier's expectations.

    Boxes near an edge are why the caller pads the image first: the crop window can
    extend past the original bounds. Crops smaller than ``min_size`` on a side are
    widened around their centre so small detections still carry context.

    :param image: PIL image, already zero-padded by ``border`` on every side
    :param bbox: normalised YOLO box as [x_center, y_center, width, height]
    :param border: zero-padding in pixels previously added around the image
    :param pad_fraction: fraction of the box's own dimensions to expand each side by
    :param min_size: minimum side length in pixels before the crop is widened
    :return: the cropped PIL image
    """
    pix_width = image.size[0] - 2 * border
    pix_height = image.size[1] - 2 * border
    # pix_width, pix_height = image.size
    box_coords = get_box_coords(image, bbox, pad_fraction, border)
    # resizing image if width too small
    if abs(box_coords['right'] - box_coords['left']) < min_size:
        box_coords['left'] = round(pix_width * (box_coords['x_center']) - min_size + border)
        box_coords['right'] = round(pix_width * (box_coords['x_center']) + min_size + border)
    # resizing image if height too small
    if abs(box_coords['top'] - box_coords['bottom']) < min_size:
        box_coords['bottom'] = round(pix_height * (box_coords['y_center']) - min_size + border)
        box_coords['top'] = round(pix_height * (box_coords['y_center']) + min_size + border)

    cropped_image = image.crop((box_coords['left'], box_coords['bottom'], box_coords['right'], box_coords['top']))

    return cropped_image


def transform_img_dimensions(cropped_detected_mangrove, trans_width, trans_height):
    """
    Resize a crop to the classifier's input geometry without distorting it.

    Scales the longest side to the target and letterboxes the remainder, so aspect
    ratio is preserved -- stretching a tall mangrove to a square would shift the
    features the classifier keys on.

    :param cropped_detected_mangrove: PIL image of a single detection
    :param trans_width: target width in pixels
    :param trans_height: target height in pixels
    :return: the resized and padded PIL image
    """

    # TODO: these should use the trans_width/trans_height arguments rather than
    # reading config.pipeline_meta directly; the arguments are currently ignored.
    # check if width/height equal
    if cropped_detected_mangrove.size[0] == cropped_detected_mangrove.size[1]:
        cropped_detected_mangrove = cropped_detected_mangrove.resize(
            (config.pipeline_meta.input_width, config.pipeline_meta.input_height))
    else:
        small_idx = np.argmin(cropped_detected_mangrove.size)
        large_idx = np.argmax(cropped_detected_mangrove.size)

        # getting resizing dimensions
        new_small_length = int(config.pipeline_meta.input_width * (
                    cropped_detected_mangrove.size[small_idx] / cropped_detected_mangrove.size[large_idx]))
        new_size = [0, 0]
        new_size[small_idx] = new_small_length
        new_size[large_idx] = config.pipeline_meta.input_width
        cropped_detected_mangrove = cropped_detected_mangrove.resize(tuple(new_size))

    # padding rest of image with zeros to be equal to input_height
    cropped_detected_mangrove = ImageOps.pad(cropped_detected_mangrove,
                                             (config.pipeline_meta.input_width, config.pipeline_meta.input_height))

    return cropped_detected_mangrove
