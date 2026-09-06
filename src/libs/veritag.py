import base64
import io
import os
import json
import uuid

from urllib.parse import urlparse

import requests
import torch

import open_clip
import numpy as np

from anthropic import Anthropic
from google import genai
from openai import OpenAI

from google.genai import types
from pydantic import BaseModel

from io import BytesIO
from PIL import Image

from src.utils import context

config = context.config
logger = context.logger


def load_openclip(model_name, pretrained):
    """
    Setup OpenClip classification model
    :param model_name:
    :param pretrained:
    :return:
    """

    model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def load_photo(url, show=False):
    """
    Load image from url (veritreephotos public endpoint)
    :param url:
    :return:
    """

    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    if show:
        img.show()

    return img


def load_photo_b64(image_path, bool_encode=True):
    """
    Load image from url with base64 encoding support
    :param image_path:
    :return:
    """

    if image_path.startswith("http"):
        response = requests.get(image_path)
        output = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        output = Image.open(image_path).convert("RGB")

    if bool_encode:
        # Convert PIL image to base64
        buffered = io.BytesIO()
        output.save(buffered, format="JPEG")
        output = base64.b64encode(buffered.getvalue()).decode()

    return output


def save_photos(url, save_dir, autoname=False):

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # Fallback filename if URL doesn't contain one
    if not filename:
        rand_id = uuid.uuid4()
        filename = f"downloaded_image_{rand_id}.jpg"

    if autoname:
        rand_id = uuid.uuid4()
        filename = f"downloaded_image_{rand_id}.jpg"

    file_path = os.path.join(save_dir, filename)

    try:
        # Stream download to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Image saved to: {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None


def build_text_features(labels_dict, model, tokenizer):
    """
    Build text features based on labels dictionary.
    :param labels_dict:
    :param model:
    :param tokenizer:
    :return:
    """

    label_keys = list(labels_dict.keys())
    all_features = []

    for key in label_keys:
        prompts = labels_dict[key]
        text_tokens = tokenizer(prompts)
        with torch.no_grad():
            features = model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=0, keepdim=True)
            all_features.append(features)

    text_features = torch.cat(all_features, dim=0)
    return label_keys, text_features


def classify_image_multi(image_path, preprocess, model, label_keys, text_features, thresholds=None):
    """
    Classify an image against multiple labels with per-label thresholds.

    Args:
        image_path (str): Path to the image file (of the form: 'https://veritreephotos.s3...' or local path)
        preprocess (obj): Preprocessing project
        model (obj): Model object for CLIP classification
        label_keys (list): List of label keys
        text_features (torch.Tensor): Precomputed text embeddings for labels
        thresholds (dict, optional): Per-label threshold, e.g., {"seedling":0.25,...}
                                     If None, default threshold 0.255 is used for all.

    Returns:
        dict: {label: similarity_score} for all labels above threshold,
              or {"": max_score} if there are no matches
    """

    if thresholds is None:
        thresholds = {k: 0.255 for k in label_keys}

    try:
        if 'http' in image_path:
            img_obj = load_photo(image_path)
            image = preprocess(img_obj).unsqueeze(0)
        else:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error: unable to load verification photo: {e}")

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)

    similarity = similarity.cpu().numpy()
    similarity = np.round(similarity, 5)

    dict_scores = {}
    matches = {}
    for i, label in enumerate(label_keys):
        if similarity[i] >= thresholds.get(label, 0.25):
            matches[label] = similarity[i]
        dict_scores[label] = similarity[i]

    if not matches:
        max_idx = similarity.argmax()
        matches = {"": similarity[max_idx]}

    return matches, dict_scores


def load_gemini_client(api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")):
    """
    Startup Google-Gemini API client
    :return:
    """
    client = genai.Client(api_key=api_key)
    return client


def load_openai_client(api_key=os.getenv("OPENAI_API_KEY")):
    """
    Startup OpenAI-GPT API client
    :return:
    """
    client = OpenAI(api_key=api_key)
    return client


def load_anthropic_client(api_key=os.getenv("ANTHROPIC_API_KEY")):
    """
    Startup Anthropic-Claude API client
    :return:
    """
    client = Anthropic(api_key=api_key)
    return client


def classify_image_gemini(client, model_name, image_path, tag_names, system_prompt, threshold):
    """
    Classify an image against several tag_names using gemini AI model.
    :param client:
    :param model_name:
    :param image_path:
    :param tag_names:
    :param system_prompt:
    :param threshold:
    :return:
    """

    # client = genai.Client(api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"))

    class Detection(BaseModel):
        detection: bool
        probability: float

    try:
        if 'http' in image_path:
            image = load_photo(image_path)
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error: unable to load verification photo: {e}")

    matches = {}

    for i_tag_name in tag_names.keys():

        query_prompt = tag_names[i_tag_name]

        response = client.models.generate_content(
            model=model_name,
            contents=[query_prompt, image],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Detection
            )
        )

        result = response.parsed

        if result.probability > threshold:
            matches[i_tag_name] = result.probability

    return matches


def classify_image_gemini(client, model_name, image_path, tag_names, system_prompt, threshold):
    """
    Classify an image against several tag_names using gemini AI model.
    :param client:
    :param model_name:
    :param image_path:
    :param tag_names:
    :param system_prompt:
    :param threshold:
    :return:
    """

    # client = genai.Client(api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"))

    class Detection(BaseModel):
        detection: bool
        probability: float

    try:
        if 'http' in image_path:
            image = load_photo(image_path)
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error: unable to load verification photo: {e}")

    matches = {}

    for i_tag_name in tag_names.keys():

        query_prompt = tag_names[i_tag_name]

        response = client.models.generate_content(
            model=model_name,
            contents=[query_prompt, image],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Detection
            )
        )

        result = response.parsed

        if result.probability > threshold:
            matches[i_tag_name] = result.probability

    return matches


def classify_image_openai(client, model_name, image_path, tag_names, system_prompt, threshold):
    """
    Classify an image against several tag_names using OpenAI model.
    :param client:
    :param model_name:
    :param image_path:
    :param tag_names:
    :param system_prompt:
    :param threshold:
    :return:
    """

    class Detection(BaseModel):
        detection: bool
        probability: float

    try:
        image_b64 = load_photo_b64(image_path)
    except Exception as e:
        logger.error(f"Error: unable to load verification photo: {e}")
        return {}

    matches = {}

    for tag_key, query_prompt in tag_names.items():

        response = client.responses.parse(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": query_prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    ]
                }
            ],
            text_format=Detection
        )

        result = response.output_parsed

        if result and result.probability > threshold:
            matches[tag_key] = result.probability

    return matches


def classify_image_anthropic(client, model_name, image_path, tag_names, system_prompt, threshold):
    """
    Classify an image against several tag_names using the Anthropic Claude API.

    :param client: anthropic.Anthropic() client instance
    :param model_name: e.g. 'claude-3-5-sonnet-20241022'
    :param image_path: URL or local file path
    :param tag_names: dict of tag names and their corresponding prompts
    :param system_prompt: System instruction text
    :param threshold: float probability threshold
    :return: dict of matches and their probabilities
    """

    try:
        if 'http' in image_path:
            image = load_photo(image_path)
        else:
            image = Image.open(image_path).convert("RGB")

        # Claude requires a base64 encoded string and a specific media type.
        # Also ensure it's RGB and save it to an in-memory buffer as a JPEG.
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        media_type = "image/jpeg"

    except Exception as e:
        logger.error(f"Error: unable to load verification photo: {e}")
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
            logger.error(f"Error processing tag '{i_tag_name}': {e}")

    return matches