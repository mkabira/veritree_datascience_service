"""
Computer vision routes.

  /computer_vision/survivability_classifier   mangrove detection + alive/dead classification
  /computer_vision/content_tagging            field photo evidence tagging (people, meterstick, ...)
  /computer_vision/content_moderation         AI content moderation over field photos

Ported from veritree-survivability :: api/surv_endpoint.py. The provider-specific
veritag_{cvmodel,gemini,openai,anthropic} routes of that service are collapsed here into a
single content_tagging route with a `provider` field; anthropic remains the production default.
"""

import io
import os
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from PIL import Image
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from api.dependencies import api_authentication

from src.awskit.datahandlers import S3StorageHandler

from src.libs import veritag
from src.libs.utils import copy_exif, delete_file, empty_folder

from src.services.computer_vision.survivability import inference_light
from src.services.computer_vision.survivability.inference import run_survivability_inference
from src.services.computer_vision.photo_tagging.verification_photo_tagging import (
    run_veritag_anthropic,
    run_veritag_cvmodel,
    run_veritag_gemini,
    run_veritag_openai,
)

from src.utils import context


config = context.config
logger = context.logger


router = APIRouter(prefix="/computer_vision", tags=["computer_vision"])


# ---------------------------------------------------------------------------
# S3 handlers
#
# Two distinct buckets: images arrive in a source bucket and annotated outputs are
# written to a veritree-owned destination bucket, so these are separate from the
# single shared s3_handler in src.awskit.datahandlers.
# ---------------------------------------------------------------------------

s3_configuration_default = {
    "source_bucket": {
        "aws_accesskey": os.getenv("AWS_SOURCE_ACCESS_KEY_ID"),
        "aws_secretkey": os.getenv("AWS_SOURCE_SECRET_ACCESS_KEY"),
        "aws_regionname": os.getenv("AWS_SOURCE_REGION_NAME"),
        "aws_bucketname": os.getenv("AWS_SOURCE_S3_BUCKET_NAME"),
        "prod_mode": config.prod_mode
    },
    "destination_bucket": {
        "aws_accesskey": os.getenv("AWS_DESTINATION_ACCESS_KEY_ID"),
        "aws_secretkey": os.getenv("AWS_DESTINATION_SECRET_ACCESS_KEY"),
        "aws_regionname": os.getenv("AWS_DESTINATION_REGION_NAME"),
        "aws_bucketname": os.getenv("AWS_DESTINATION_S3_BUCKET_NAME"),
        "prod_mode": config.prod_mode
    }
}

s3_source_handler = S3StorageHandler(**s3_configuration_default['source_bucket'])
s3_destination_handler = S3StorageHandler(**s3_configuration_default['destination_bucket'])


# ---------------------------------------------------------------------------
# OpenCLIP model, loaded once at import so the weights are not re-read per request
# ---------------------------------------------------------------------------

model, preprocess, tokenizer = veritag.load_openclip(
    model_name=config.l3_verification_cvmodel.model_name,
    pretrained=config.l3_verification_cvmodel.pretrained)

label_keys, text_features = veritag.build_text_features(
    labels_dict=config.l3_verification_cvmodel.verification_tags,
    model=model,
    tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SurvS3Source(BaseModel):
    org_id:     str
    image_url:  str


class SurvS3Response(BaseModel):
    session_id:        str
    s3cv_url:          str
    number_mangroves:  int
    number_alive:      int
    number_dead:       int
    number_unclear:    int


class TagInput(BaseModel):
    image_url: str
    provider: Literal['anthropic', 'gemini', 'openai', 'cvmodel'] = 'anthropic'


class TagOutput(BaseModel):
    session_id: str
    tags:       list
    scores:     Optional[dict] = None


class ModerationInput(BaseModel):
    image_url: str


class ModerationOutput(BaseModel):
    session_id: str
    flagged:    bool
    tags:       list
    scores:     Optional[dict] = None


# ---------------------------------------------------------------------------
# survivability_classifier
# ---------------------------------------------------------------------------

@router.post("/survivability_classifier/", response_model=SurvS3Response,
             dependencies=[Depends(api_authentication)])
async def survivability_classifier(params: SurvS3Source):
    """
    Return survivability metrics for an image held in S3, and write the annotated
    image back to the destination bucket.

    :param params: json payload of the form {'org_id': '123', 'image_url': 's3://xyz'}
    :return: processed image s3 path and computer vision metrics
    """

    logger.info("computer_vision survivability_classifier: ENTRY")

    session_id = str(uuid.uuid4())
    datenow = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    s3_source_img_url = params.image_url
    s3_filename = s3_source_img_url.split('/')[-1]
    fileprefix = s3_filename.split('.')[0]
    filesuffix = s3_filename.split('.')[1]
    local_filename = f"{fileprefix}_cv_{session_id}.{filesuffix}"
    local_filepath = context.root_dir + config.api_config.local_data_input_dir + local_filename

    s3_source_handler.download(s3_source_img_url, local_filepath)

    logger.info(f'client reached: /computer_vision/survivability_classifier: session_id={session_id}')

    try:
        _image, _surv_results = inference_light.run_survivability_inference(
            config=config,
            path_input_img=local_filepath,
            session_id=session_id)

        local_savepath = context.root_dir + config.api_config.local_data_output_dir + local_filename
        _image.save(local_savepath)

        copy_exif(source_path=local_filepath, target_path=local_savepath, output_path=local_savepath)

        location_destination = f"{config.api_config.s3_destination_basedir}/{params.org_id}/{datenow}/{local_filename}"
        s3_destination_handler.upload(local_savepath, location_destination)
        logger.info(f'Survivability computer vision output uploaded to {location_destination}')

        logger.info('Cleaning up generated local files')
        delete_file(local_filepath)
        delete_file(local_savepath)

        logger.info("computer_vision survivability_classifier: EXIT")

        return SurvS3Response(
            session_id=session_id,
            s3cv_url=location_destination,
            number_mangroves=_surv_results["n_mangroves"],
            number_alive=_surv_results["n_alive"],
            number_dead=_surv_results["n_dead"],
            number_unclear=_surv_results["n_unclear"],
        )

    except Exception as e:
        err_string = f"Exception while processing survivability_classifier: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)


@router.post("/survivability_classifier/image/", dependencies=[Depends(api_authentication)])
async def survivability_classifier_image(file: UploadFile = File(...)):
    """
    Process a directly uploaded image and return it annotated with mangrove
    detections and classifications.

    :param file: image file
    :return: image bytes
    """

    logger.info('client reached: /computer_vision/survivability_classifier/image')

    local_data_input_dir = config.api_config.local_data_input_dir
    local_data_output_dir = config.api_config.local_data_output_dir
    empty_folder(context.root_dir + local_data_input_dir)
    empty_folder(context.root_dir + local_data_output_dir)

    try:
        image_bytes = await file.read()
        image_in = Image.open(io.BytesIO(image_bytes))
        image_in.save(f"{context.root_dir + local_data_input_dir}/surv_image.png")

        config.pipeline.prod_mode = False
        config.pipeline.save_cropped_mangroves = False
        config.pipeline.image_path = local_data_input_dir
        config.pipeline.save_dir = local_data_output_dir

        _, _ = run_survivability_inference(config=config)

        image_out = Image.open(
            f"{context.root_dir + local_data_output_dir}/surv_image/surv_image_mangroves_annotated.jpg")
        image_out_bytes = io.BytesIO()
        image_out.save(image_out_bytes, format=image_out.format or "png")

        return Response(
            content=image_out_bytes.getvalue(),
            media_type=f"image/{image_out.format.lower() if image_out.format else 'png'}")

    except Exception as e:
        logger.error(f"Exception while processing survivability_classifier/image: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/survivability_classifier/results/", dependencies=[Depends(api_authentication)])
async def survivability_classifier_results(file: UploadFile = File(...)):
    """
    Return survivability metrics for a directly uploaded image: number of mangroves,
    dead, alive, and unclear.

    :param file: image file
    :return: json/dictionary of metrics
    """

    logger.info('client reached: /computer_vision/survivability_classifier/results')

    local_data_input_dir = config.api_config.local_data_input_dir
    local_data_output_dir = config.api_config.local_data_output_dir
    empty_folder(context.root_dir + local_data_input_dir)
    empty_folder(context.root_dir + local_data_output_dir)

    try:
        image_bytes = await file.read()
        image_in = Image.open(io.BytesIO(image_bytes))
        image_in.save(f"{context.root_dir + local_data_input_dir}/surv_image.png")

        config.pipeline.prod_mode = False
        config.pipeline.save_cropped_mangroves = False
        config.pipeline.image_path = local_data_input_dir
        config.pipeline.save_dir = local_data_output_dir

        _, _surv_results = run_survivability_inference(config=config)

        return JSONResponse(content=_surv_results)

    except Exception as e:
        logger.error(f"Exception while processing survivability_classifier/results: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ---------------------------------------------------------------------------
# content_tagging
# ---------------------------------------------------------------------------

def _run_content_tagging(provider: str, image_url: str, session_id: str):
    """
    Dispatch a tagging request to the configured provider.

    :return: (matches, dict_scores)
    """

    if provider == 'cvmodel':
        cfg = config.l3_verification_cvmodel
        return run_veritag_cvmodel(
            model=model,
            preprocess=preprocess,
            label_keys=label_keys,
            text_features=text_features,
            path_input_img=image_url,
            thresholds=cfg.verification_thresholds,
            session_id=session_id)

    runners = {
        'anthropic': (run_veritag_anthropic, config.l3_verification_anthropic),
        'gemini':    (run_veritag_gemini,    config.l3_verification_gemini),
        'openai':    (run_veritag_openai,    config.l3_verification_openai),
    }

    runner, cfg = runners[provider]

    return runner(
        model_name=cfg.model_name,
        path_input_img=image_url,
        tag_names=cfg.verification_tags,
        system_prompt=cfg.system_prompt,
        threshold=cfg.verification_threshold,
        session_id=session_id)


@router.post("/content_tagging/", response_model=TagOutput,
             dependencies=[Depends(api_authentication)])
async def content_tagging(params: TagInput):
    """
    Tag field photo evidence for L3 verification (people, meterstick, ...).

    When `people` is detected the image is additionally routed through content
    moderation, and any moderation tags are merged into the response -- this
    preserves the AICM chaining behaviour of the deployed veritag_anthropic route.

    :param params: {'image_url': 'https://veritreephotos..', 'provider': 'anthropic'}
    :return: session_id and list of tags for the photo evidence
    """

    logger.info(f"computer_vision content_tagging (provider={params.provider}): ENTRY")

    session_id = str(uuid.uuid4())

    try:
        matches, dict_matches = _run_content_tagging(params.provider, params.image_url, session_id)

        if params.provider == 'cvmodel':
            matches = list(matches.keys())

        if 'people' in matches:
            logger.info('person detected in the image, routing to AI Content Moderation node')

            matches_aicm, dict_score_aicm = _run_content_moderation(params.image_url, session_id)

            matches = list(matches) + list(matches_aicm)
            dict_matches = dict(dict_matches) | dict(dict_score_aicm)

            logger.warning(f"AICM matches found: {matches_aicm} with scores: {dict_score_aicm}")
        else:
            logger.info('no people detected, skipping AI Content Moderation node')

        logger.info(f"computer_vision content_tagging: EXIT session_id={session_id}")

        return TagOutput(session_id=session_id, tags=list(matches), scores=dict(dict_matches))

    except Exception as e:
        err_string = f"Exception while processing content_tagging: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)


# ---------------------------------------------------------------------------
# content_moderation
# ---------------------------------------------------------------------------

def _run_content_moderation(image_url: str, session_id: str):
    """
    Run AI content moderation (AICM) over a field photo.

    :return: (matches, dict_scores)
    """

    cfg = config.aicm_anthropic

    return run_veritag_anthropic(
        model_name=cfg.model_name,
        path_input_img=image_url,
        tag_names=cfg.verification_tags,
        system_prompt=cfg.system_prompt,
        threshold=cfg.verification_threshold,
        session_id=session_id)


@router.post("/content_moderation/", response_model=ModerationOutput,
             dependencies=[Depends(api_authentication)])
async def content_moderation(params: ModerationInput):
    """
    Run AI Content Moderation over a field photo, flagging images that require
    review before publication (e.g. images containing minors).

    Exposed as a standalone route here; it is also chained automatically from
    /computer_vision/content_tagging whenever people are detected.

    :param params: {'image_url': 'https://veritreephotos..'}
    :return: session_id, flagged boolean and moderation tags
    """

    logger.info("computer_vision content_moderation: ENTRY")

    session_id = str(uuid.uuid4())

    try:
        matches, dict_matches = _run_content_moderation(params.image_url, session_id)

        flagged = bool(matches)

        if flagged:
            logger.warning(f"AICM flagged image with tags {matches} and scores {dict_matches}")

        logger.info(f"computer_vision content_moderation: EXIT session_id={session_id}")

        return ModerationOutput(
            session_id=session_id,
            flagged=flagged,
            tags=list(matches),
            scores=dict(dict_matches))

    except Exception as e:
        err_string = f"Exception while processing content_moderation: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)
