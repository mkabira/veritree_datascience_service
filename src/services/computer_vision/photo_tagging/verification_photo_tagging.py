import time
import uuid
import numpy as np

from src.libs import veritag
from src.utils import context

config = context.config
logger = context.logger
session_id = str(uuid.uuid4())


model, preprocess, tokenizer = veritag.load_openclip(
    model_name=config.l3_verification_cvmodel.model_name,
    pretrained=config.l3_verification_cvmodel.pretrained)

label_keys, text_features = veritag.build_text_features(
    labels_dict=config.l3_verification_cvmodel.verification_tags,
    model=model,
    tokenizer=tokenizer)

def run_veritag_cvmodel(model,
                       preprocess,
                       label_keys,
                       text_features,
                       path_input_img,
                       thresholds,
                       session_id):

    logger.info(f"STARTING VERIFICATION PHOTO TAGGING W CVMODEL: SessionID:{session_id}")

    start_time_clock = time.time()

    logger.info(f"Performing CV object detection on input image: {path_input_img}")
    matches, dict_score = veritag.classify_image_multi(
        image_path=path_input_img,
        preprocess=preprocess,
        model=model,
        label_keys=label_keys,
        text_features=text_features,
        thresholds=thresholds)

    end_time_clock = time.time()
    duration = end_time_clock - start_time_clock
    logger.info(f"TASK COMPLETED!: {duration//60}mins or : {np.round(duration,2)}secs")

    return matches, dict_score


client_gemini = veritag.load_gemini_client()
def run_veritag_gemini(model_name,
                       path_input_img,
                       tag_names,
                       system_prompt,
                       threshold,
                       session_id):

    logger.info(f"STARTING VERIFICATION PHOTO TAGGING W GEMINI: SessionID:{session_id}")

    start_time_clock = time.time()

    logger.info(f"Performing CV object detection on input image: {path_input_img}")
    matches = veritag.classify_image_gemini(client=client_gemini,
                                            model_name=model_name,
                                            image_path=path_input_img,
                                            tag_names=tag_names,
                                            system_prompt=system_prompt,
                                            threshold=threshold)

    end_time_clock = time.time()
    duration = end_time_clock - start_time_clock
    logger.info(f"TASK COMPLETED!: {duration//60}mins or : {np.round(duration,2)}secs")

    return list(matches.keys()), matches


client_openai = veritag.load_openai_client()
def run_veritag_openai(model_name,
                       path_input_img,
                       tag_names,
                       system_prompt,
                       threshold,
                       session_id):

    logger.info(f"STARTING VERIFICATION PHOTO TAGGING W OPENAI: SessionID:{session_id}")

    start_time_clock = time.time()

    logger.info(f"Performing CV object detection on input image: {path_input_img}")
    matches = veritag.classify_image_openai(client=client_openai,
                                            model_name=model_name,
                                            image_path=path_input_img,
                                            tag_names=tag_names,
                                            system_prompt=system_prompt,
                                            threshold=threshold)

    end_time_clock = time.time()
    duration = end_time_clock - start_time_clock
    logger.info(f"TASK COMPLETED!: {duration//60}mins or : {np.round(duration,2)}secs")

    return list(matches.keys()), matches


client_anthropic = veritag.load_anthropic_client()
def run_veritag_anthropic(model_name,
                          path_input_img,
                          tag_names,
                          system_prompt,
                          threshold,
                          session_id):

    logger.info(f"STARTING VERIFICATION PHOTO TAGGING W ANTHROPIC: SessionID:{session_id}")

    start_time_clock = time.time()

    logger.info(f"Performing CV object detection on input image: {path_input_img}")
    matches = veritag.classify_image_anthropic(client=client_anthropic,
                                               model_name=model_name,
                                               image_path=path_input_img,
                                               tag_names=tag_names,
                                               system_prompt=system_prompt,
                                               threshold=threshold)

    end_time_clock = time.time()
    duration = end_time_clock - start_time_clock
    logger.info(f"TASK COMPLETED!: {duration//60}mins or : {np.round(duration,2)}secs")

    return list(matches.keys()), matches

if __name__ == '__main__':

    path_image = 'https://veritreephotos.s3.us-east-2.amazonaws.com/bulk_uploads/33/user_1328_utc_2026-02-04-05-00-00/X-I9/zgQ6Z6vDeErXFqhJxeu9ZPI1LyipsN4R-zgQ6Z6vDeErXFqhJxeu9ZPI1LyipsN4R.jpeg'
    # path_image = 'https://veritreephotos.s3.us-east-2.amazonaws.com/bulk_uploads/44/user_1328_utc_2025-05-17-13-00-00/zQAS/0IzqWCN8CixIPV5NaObdRE2E8xeqU9Bk.jpeg'
    # path_image = 'https://veritreephotos.s3.us-east-2.amazonaws.com/bulk_uploads/44/user_1328_utc_2025-05-19-11-00-00/lKnL/fGpa4vW1o81iEB3sBSOrz3PW1rAv5i0d.jpeg'
    # path_image = 'https://veritreephotos.s3.us-east-2.amazonaws.com/bulk_uploads/33/user_1328_utc_2026-02-17-14-00-00/HznQ/zsWEzz8gP2D2cLRTv2KCiMnnp8ODiPuK-zsWEzz8gP2D2cLRTv2KCiMnnp8ODiPuK.jpeg'

    ## OPENCLIP PIPELINE
    # matches, dict_score = run_veritag_cvmodel(
    #     model=model,
    #     preprocess=preprocess,
    #     label_keys=label_keys,
    #     text_features=text_features,
    #     path_input_img=path_image,
    #     thresholds=config.l3_verification_cvmodel.verification_thresholds,
    #     session_id=None
    # )
    # print(f"Matches found: {matches} with overall scores: {dict_score}")

    ## GEMINI PIPELINE
    # matches, dict_score = run_veritag_gemini(
    #     model_name=config.l3_verification_gemini.model_name,
    #     path_input_img=path_image,
    #     tag_names=config.l3_verification_gemini.verification_tags,
    #     system_prompt=config.l3_verification_gemini.system_prompt,
    #     threshold=config.l3_verification_gemini.verification_threshold,
    #     session_id=None
    # )
    # print(f"Matches found: {matches} with overall scores: {dict_score}")

    ## OPENAI PIPELINE
    # matches, dict_score = run_veritag_openai(
    #     model_name=config.l3_verification_openai.model_name,
    #     path_input_img=path_image,
    #     tag_names=config.l3_verification_openai.verification_tags,
    #     system_prompt=config.l3_verification_openai.system_prompt,
    #     threshold=config.l3_verification_openai.verification_threshold,
    #     session_id=None
    # )
    # print(f"Matches found: {matches} with overall scores: {dict_score}")

    ## ANTHROPIC PIPELINE
    matches, dict_score = run_veritag_anthropic(
        model_name=config.l3_verification_anthropic.model_name,
        path_input_img=path_image,
        tag_names=config.l3_verification_anthropic.verification_tags,
        system_prompt=config.l3_verification_anthropic.system_prompt,
        threshold=config.l3_verification_anthropic.verification_threshold,
        session_id=None
    )
    print(f"Matches found: {matches} with overall scores: {dict_score}")