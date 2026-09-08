"""
Anthropic vision classification for field photo evidence.

Loads an image, encodes it for the Messages API, and asks the model for a calibrated
probability per candidate tag via a forced ``record_detection`` tool call. Only the
detection flag and its probability cross the boundary, so the caller compares against
a single threshold.

Anthropic is the only backend: the OpenCLIP, Gemini and OpenAI taggers this module
once carried were retired once Anthropic was measured as the best performer on
veritree field photos.

Consumed by src/services/computer_vision/content_tagging.py.
"""

import base64
import io
import os

import requests

from anthropic import Anthropic
from PIL import Image

from src.utils import context


config = context.config
logger = context.logger


def load_photo(url, show=False):
    """
    Fetch an image over HTTP and return it as RGB.

    :param url: URL of the image, e.g. the veritreephotos public endpoint
    :param show: open the image in the default viewer, for interactive use
    :return: the loaded PIL image
    :raises requests.HTTPError: if the URL returns a non-2xx status
    """

    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    if show:
        img.show()

    return img


def load_anthropic_client(api_key=os.getenv("ANTHROPIC_API_KEY")):
    """
    Construct the Anthropic API client.

    :param api_key: Anthropic API key; defaults to $ANTHROPIC_API_KEY read at import
    :return: a configured Anthropic client
    """
    client = Anthropic(api_key=api_key)
    return client


def classify_image_anthropic(client, model_name, image, tag_names, system_prompt, threshold):
    """
    Classify an already-loaded image against several tag names.

    Takes a PIL image rather than a path so the caller owns image loading: the
    computer_vision route reads the source once and hands the same decoded image to
    whichever service was requested, instead of each classifier re-fetching it.

    :param client: anthropic.Anthropic() client instance
    :param model_name: Anthropic model identifier
    :param image: PIL image to classify
    :param tag_names: mapping of tag name to the question describing it
    :param system_prompt: system instruction text
    :param threshold: minimum probability for a tag to be reported
    :return: mapping of each tag above the threshold to its probability
    """

    try:
        # Claude takes the image inline as base64 with an explicit media type.
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        media_type = "image/jpeg"

    except Exception as e:
        logger.error(f"veritag encode failed: error={e}")
        return {}

    matches = {}

    # Equivalent of the Pydantic Detection model as a Claude tool
    tools = [
        {
            "name": "record_detection",
            "description": "Record the detection result and its probability.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detection": {
                        "type": "boolean",
                        "description": "True if the condition is met, False otherwise."
                    },
                    "probability": {
                        "type": "number",
                        "description": "The probability of the detection, between 0.0 and 1.0."
                    }
                },
                "required": ["detection", "probability"]
            }
        }
    ]

    for i_tag_name, query_prompt in tag_names.items():

        try:
            response = client.messages.create(
                model=model_name,
                system=system_prompt,
                max_tokens=1024,
                tools=tools,
                tool_choice={"type": "tool", "name": "record_detection"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": query_prompt
                            }
                        ]
                    }
                ]
            )

            # Find the tool use block in Claude's response
            for block in response.content:
                if block.type == "tool_use" and block.name == "record_detection":
                    # block.input contains the parsed JSON dictionary
                    probability = block.input.get("probability", 0.0)

                    if probability > threshold:
                        matches[i_tag_name] = probability

        except Exception as e:
            logger.error(f"veritag tag failed: tag={i_tag_name} error={e}")

    return matches
