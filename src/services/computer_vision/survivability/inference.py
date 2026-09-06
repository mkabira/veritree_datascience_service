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

from src.awskit.datahandlers import S3StorageHandler
from src.awskit.datahandlers import RDSPostgresHandler

config = context.config
logger = context.logger
session_id = str(uuid.uuid4())

# THIS IS A REDACTED MODULE, FOR TESTING, FILL OUT PARAMETERS
s3_handler = None
postgres_handler = None

def run_survivability_inference(config):

    logger.info(f"\nSTARTING SURVIVABILITY INFERENCE TASK: SessionID:{session_id}")

    datenow = datetime.datetime.today().strftime('%Y-%m-%d').replace('-', '')
    start_time_clock = time.time()

    logger.info(f"OUTPUT = {config}: {datenow}: {start_time_clock}")

    logger.info(f"Setting up project paths and initializing directories, and folders")
    image_paths, results_dir = utils.setup_paths(context.root_dir + config.pipeline.image_path,
                                                 context.root_dir + config.pipeline.save_dir)

    str_to_onehot, onehot_to_str = utils.cast_classes(dict(config.classes))

    # load object detection model
    logger.info(f"Loading mangrove detection model using YOLO model at: {config.pipeline.object_model_path}")
    mangrove_detection_model = YOLO(context.root_dir + config.pipeline.object_model_path)

    # load classifier model
    logger.info(f"Loading mangrove classification model using KERAS model at: {config.pipeline.survival_model_path}")
    mangrove_classification_model = keras.models.load_model(context.root_dir + config.pipeline.survival_model_path)

    # image_paths = [image_paths[0]]
    _image, _surv_results = None, None
    assert len(image_paths) > 0, logger.warn("No files found to process")

    for image_path in image_paths:

        df_survivability_results = pd.DataFrame()

        logger.info(f"[Survivability model]: Running pipeline on image: {image_path.split('/')[-1]}")

        image = Image.open(image_path)
        image = utils.rotate_image(image, image_path)

        # running object prediction on image
        logger.info('** Model ** Running mangrove object detection model on images')
        results_mangrove_detections = mangrove_detection_model(image, conf=config.pipeline.confidence, verbose=False)[0]

        results_detections = {}
        dataset_metadata = pd.DataFrame()

        if len(results_mangrove_detections) > 0:

            logger.info(f"Mangrove detection model found {len(results_mangrove_detections)} mangroves in image {image_path}")

            for index, result in enumerate(results_mangrove_detections):
                xywhn = result.boxes.xywhn.flatten().tolist()  # normalized
                names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
                confs = result.boxes.conf  # confidence score of each box

                results_detections['image_filename'] = image_path
                results_detections['detection_index'] = index
                results_detections['detection_id'] = str(uuid.uuid4())
                results_detections['detection_bbox_xywhn'] = xywhn
                results_detections['detection_name'] = names[0]
                results_detections['detection_score'] = float(confs)

                df_temp = pd.DataFrame.from_dict(results_detections.values(), orient='columns').reset_index(drop=True).T
                df_temp.columns = list(results_detections.keys())
                dataset_metadata = pd.concat([dataset_metadata, df_temp])
        else:
            logger.warning(f"No mangroves found in image {image_path} using confidence threshold {config.pipeline.confidence}")

            results_detections['image_filename'] = image_path
            results_detections['detection_index'] = ''
            results_detections['detection_id'] = ''
            results_detections['detection_bbox_xywhn'] = ''
            results_detections['detection_name'] = ''
            results_detections['detection_score'] = ''

            df_temp = pd.DataFrame.from_dict(results_detections.values(), orient='columns').reset_index(drop=True).T
            df_temp.columns = list(results_detections.keys())
            dataset_metadata = pd.concat([dataset_metadata, df_temp])

            continue

        dataset_metadata = dataset_metadata.reset_index(drop=True).reset_index()

        # cropping detected boxes
        list_detected_mangroves = []
        exp_image = ImageOps.expand(image, border=config.pipeline_meta.border)

        logger.info('Cropping data and running class models')

        for idx, bbox in zip(dataset_metadata['detection_index'], dataset_metadata['detection_bbox_xywhn']):
            cropped_detected_mangrove = utils.crop_image(exp_image, bbox,
                                                         config.pipeline_meta.border,
                                                         config.pipeline_meta.pad_fraction,
                                                         config.pipeline_meta.min_size)
            # saving cropped images
            image_id, _ = os.path.splitext(image_path.split('/')[-1])
            image_resultpath = os.path.join(results_dir, image_id)
            save_filename = f'{image_id}_mangrove_{str(idx).zfill(3)}.jpg'
            save_path = os.path.join(image_resultpath, save_filename)

            if config.pipeline.save_cropped_mangroves:
                cropped_detected_mangrove.save(save_path)

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
        results_classes = results_classes.rename(columns={'prob_alive': 'class_alive', 'prob_dead': 'class_dead', 'prob_unclear': 'class_unclear'}).reset_index()

        boxes = results_mangrove_detections.boxes.xywhn.tolist()
        result = [[str(x) for x in y] for y in results_mangrove_classifications]
        filename_annot = f'{image_id}_mangroves_annotated.jpg'
        path_save = os.path.join(image_resultpath, filename_annot)
        _image = utils.plot_boxes(image, boxes, result, onehot_to_str, path_save)

        location_source = path_save
        location_destination = f"outputs/{filename_annot}.png"
        bool_source_exists = utils.check_file_exists(location_source, path_type='local')
        if bool_source_exists and config.prod_mode:
            logger.info(f"Moving image with annotated detections to AWS S3")
            s3_handler.upload(location_source, location_destination)

        temp_survivability_results = pd.concat([dataset_metadata, results_classifications, results_classes], axis=1)
        df_survivability_results = pd.concat([df_survivability_results, temp_survivability_results])
        df_survivability_results['timestamp_now'] = str(datetime.datetime.today())
        df_survivability_results = df_survivability_results.drop(columns=['index'])

        n_mangroves = len(df_survivability_results)
        n_alive = df_survivability_results['class_alive'].sum()
        n_dead = df_survivability_results['class_dead'].sum()
        n_unclear = df_survivability_results['class_unclear'].sum()

        _surv_results = dict({
            "n_mangroves": f"{n_mangroves}",
            "n_alive": f"{n_alive}",
            "n_dead": f"{n_dead}",
            "n_unclear": f"{n_unclear}"
        })

        logger.info(f"Inference completed on image: {image_path} with "
                    f"{n_mangroves} mangroves, "
                    f"{n_alive} alive, "
                    f"{n_dead} dead, "
                    f"{n_unclear} unclear")

        if config.prod_mode:
            logger.info(f"Writing image detection and classification results to database [tbl_survivability_inference]")
            df_survivability_results.to_sql(
                'tbl_survivability_inference',
                postgres_handler.engine,
                schema='survivability',
                if_exists='append'
            )

        path_save = os.path.join(results_dir, 'df_survivability_results.csv')
        df_survivability_results.to_csv(path_save, index=False)
        logger.info(f"Written image detection and classification results to file [df_survivability_results.csv]")

    end_time_clock = time.time()
    duration = end_time_clock - start_time_clock
    logger.info(f"Task Completed: {duration//60}mins or : {np.round(duration,2)}secs")

    logger.info('TASK COMPLETED!')

    return _image, _surv_results

if __name__ == '__main__':
    # _, _ = run_survivability_inference(config=config)
    pass