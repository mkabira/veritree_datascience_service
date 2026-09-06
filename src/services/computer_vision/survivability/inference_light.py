import datetime
import os
import time
import uuid
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from tensorflow import keras
from ultralytics import YOLO

from src.libs import utils
from src.utils import context


config = context.config
logger = context.logger
session_id = str(uuid.uuid4())


def run_survivability_inference(config, path_input_img='', session_id=None):

    logger.info(f"\nSTARTING SURVIVABILITY INFERENCE TASK: SessionID:{session_id}")

    start_time_clock = time.time()

    str_to_onehot, onehot_to_str = utils.cast_classes(dict(config.classes))

    # load object detection model
    logger.info(f"Loading mangrove detection model using YOLO model at: {config.pipeline.object_model_path}")
    try:
        mangrove_detection_model = YOLO(context.root_dir + config.pipeline.object_model_path)
    except Exception as e:
        logger.error(f"Unable to load mangrove detection model: {e}")
    
    # load classifier model
    logger.info(f"Loading mangrove classification model using KERAS model at: {config.pipeline.survival_model_path}")
    try:
        mangrove_classification_model = keras.models.load_model(context.root_dir + config.pipeline.survival_model_path)
    except Exception as e:
        logger.error(f"Unable to load mangrove classification model: {e}")

    _image, _surv_results, image = None, None, None
    assert os.path.exists(path_input_img), logger.warn("No file found to process")


    df_survivability_results = pd.DataFrame()

    logger.info(f"[Survivability model]: Running pipeline on image: {path_input_img.split('/')[-1]}")

    try:
        image = Image.open(path_input_img)
    except Exception as e:
        logger.error(f"Unable to read input image: {e}")

    image = utils.rotate_image(image, path_input_img)

    # running object prediction on image
    logger.info('** Model ** Running mangrove object detection model on images')
    results_mangrove_detections = mangrove_detection_model(image, conf=config.pipeline.confidence, verbose=False)[0]

    results_detections = {}
    dataset_metadata = pd.DataFrame()

    if len(results_mangrove_detections) > 0:

        logger.info(f"Mangrove detection model found {len(results_mangrove_detections)} mangroves in image {path_input_img}")

        for index, result in enumerate(results_mangrove_detections):
            xywhn = result.boxes.xywhn.flatten().tolist()  # normalized
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box

            results_detections['image_filename'] = path_input_img
            results_detections['detection_index'] = index
            results_detections['detection_id'] = str(uuid.uuid4())
            results_detections['detection_bbox_xywhn'] = xywhn
            results_detections['detection_name'] = names[0]
            results_detections['detection_score'] = float(confs)

            df_temp = pd.DataFrame.from_dict(results_detections.values(), orient='columns').reset_index(drop=True).T
            df_temp.columns = list(results_detections.keys())
            dataset_metadata = pd.concat([dataset_metadata, df_temp])

        dataset_metadata = dataset_metadata.reset_index(drop=True).reset_index()

        # cropping detected boxes
        list_detected_mangroves = []
        exp_image = ImageOps.expand(image, border=config.pipeline_meta.border)

        logger.info('Cropping data and running classification models')

        for idx, bbox in zip(dataset_metadata['detection_index'], dataset_metadata['detection_bbox_xywhn']):
            cropped_detected_mangrove = utils.crop_image(exp_image, bbox,
                                                         config.pipeline_meta.border,
                                                         config.pipeline_meta.pad_fraction,
                                                         config.pipeline_meta.min_size)

            cropped_detected_mangrove = utils.transform_img_dimensions(cropped_detected_mangrove,
                                                                       config.pipeline_meta.input_width,
                                                                       config.pipeline_meta.input_height)

            cropped_detected_mangrove = keras.utils.img_to_array(cropped_detected_mangrove)
            list_detected_mangroves.append(cropped_detected_mangrove)

        logger.info('** Model ** Running mangrove object classification model on images')
        results_mangrove_classifications = mangrove_classification_model(np.array(list_detected_mangroves)).numpy()
        results_classifications = pd.DataFrame(results_mangrove_classifications, columns=['prob_alive', 'prob_dead', 'prob_unclear']).reset_index()

        results_classes = results_classifications[['prob_alive', 'prob_dead', 'prob_unclear']]
        results_classes = (results_classes.T == results_classes.T.max()).T.astype(int)
        results_classes = results_classes.rename(columns={'prob_alive': 'class_alive',
                                                          'prob_dead': 'class_dead',
                                                          'prob_unclear': 'class_unclear'}).reset_index()

        boxes = results_mangrove_detections.boxes.xywhn.tolist()
        result = [[str(x) for x in y] for y in results_mangrove_classifications]

        _image = utils.plot_boxes(image, boxes, result, onehot_to_str, path_save=False, legend=False)

        temp_survivability_results = pd.concat([dataset_metadata, results_classifications, results_classes], axis=1)
        df_survivability_results = pd.concat([df_survivability_results, temp_survivability_results])
        df_survivability_results['timestamp_now'] = str(datetime.datetime.today())
        df_survivability_results = df_survivability_results.drop(columns=['index'])

        n_mangroves = len(df_survivability_results)
        n_alive = df_survivability_results['class_alive'].sum()
        n_dead = df_survivability_results['class_dead'].sum()
        n_unclear = df_survivability_results['class_unclear'].sum()

    else:
        logger.warning(f"No mangroves found in image {path_input_img} using confidence threshold {config.pipeline.confidence}")
        _image = image
        n_mangroves, n_alive, n_dead, n_unclear = 0, 0, 0, 0

    _surv_results = dict({
        "n_mangroves": f"{n_mangroves}",
        "n_alive": f"{n_alive}",
        "n_dead": f"{n_dead}",
        "n_unclear": f"{n_unclear}"
    })

    logger.info(f"Inference completed on image: {path_input_img} with "
                f"{n_mangroves} mangroves, "
                f"{n_alive} alive, "
                f"{n_dead} dead, "
                f"{n_unclear} unclear")


    end_time_clock = time.time()
    duration = end_time_clock - start_time_clock
    logger.info(f"Task Completed: {duration//60}mins or : {np.round(duration,2)}secs")

    logger.info('TASK COMPLETED!')

    return _image, _surv_results
