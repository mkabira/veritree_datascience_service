"""
Content tagging of field photo evidence for L3 verification.

Tags images with the candidate labels defined in
configs/config_cv_content_tagging.yaml (``people``, ``meterstick``, ...), returning
the tags that cleared their threshold alongside every tag's score.

Anthropic is the only backend. The OpenCLIP, Gemini and OpenAI runners this module
once offered were retired once Anthropic was measured as the best performer on
veritree field photos; the router no longer takes a ``provider`` field.

The same runner backs content moderation, called with the AICM model, prompt, tags
and threshold from configs/config_cv_content_moderation.yaml.
"""

import time


from src.libs import veritag
from src.utils import context


config = context.config
logger = context.logger


client_anthropic = veritag.load_anthropic_client()


def run_veritag_anthropic(model_name, image, tag_names,
                          system_prompt, threshold, session_id):
    """
    Tag an image with an Anthropic vision model.

    Backs both content tagging and content moderation: the moderation route calls
    this with the AICM model, prompt, tags and threshold.

    :param model_name: Anthropic model identifier
    :param image: PIL image to tag, already loaded by the caller
    :param tag_names: mapping of tag name to the question describing it
    :param system_prompt: system message setting the classifier's role
    :param threshold: minimum probability for a tag to be reported
    :param session_id: correlation id for logging
    :return: (matches, scores) -- matches is the list of tags that cleared the
        threshold; scores maps every tag to its probability
    """

    logger.info(f"content_tagging started: session_id={session_id} model={model_name}")

    start_time_clock = time.time()

    logger.info(f"content_tagging classifying: session_id={session_id} image={image.width}x{image.height} tags={len(tag_names)}")
    matches = veritag.classify_image_anthropic(client=client_anthropic,
                                               model_name=model_name,
                                               image=image,
                                               tag_names=tag_names,
                                               system_prompt=system_prompt,
                                               threshold=threshold)

    duration = time.time() - start_time_clock
    logger.info(f"content_tagging finished: session_id={session_id} matches={len(matches)} duration={duration:.2f}s")

    return list(matches.keys()), matches
