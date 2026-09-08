"""
Provider-agnostic LLM chat routing.

Extracted from src/libs/utils.py so that callers needing only text-in/text-out LLM
access -- the /analyses routes -- do not transitively import the computer-vision
stack (PIL, piexif, numpy, boto3, wikipedia) that the rest of utils.py requires.
utils.py re-exports both names, so existing `from src.libs.utils import llm_chat`
imports still resolve.
"""

import os

import litellm

from src.utils import context


config = context.config
logger = context.logger


PROVIDER_ALIASES = {
    'gpt': 'gpt-4o',
    'claude': 'anthropic/claude-sonnet-4-5',
    'gemini': 'gemini/gemini-2.5-flash',
}


def llm_chat(system_prompt: str, user_prompt: str, model: str = 'gpt-4o', temperature: float = 0.2) -> str:
    """
    Send a single system+user exchange to the configured provider.

    :param system_prompt: system message setting the model's role
    :param user_prompt: the user message to complete
    :param model: a short alias ('gpt', 'claude', 'gemini') or a full model name
        carrying a recognised prefix ('gemini-', 'claude-', 'gpt-', 'o1-', 'o3-')
    :param temperature: sampling temperature passed through to the provider
    :return: the model's reply text
    :raises ValueError: if the model name is unrecognised, or the provider's API key
        is absent from the environment
    """

    if model in PROVIDER_ALIASES:
        litellm_model = PROVIDER_ALIASES[model]
    elif model.startswith('gemini-'):
        litellm_model = f"gemini/{model}"
    elif model.startswith('claude-'):
        litellm_model = f"anthropic/{model}"
    elif model.startswith('gpt-') or model.startswith('o1-') or model.startswith('o3-'):
        litellm_model = model
    else:
        raise ValueError(f"Unknown model '{model}'. Use an alias ('gpt', 'claude', 'gemini') or a full model name with a recognised prefix.")

    if 'anthropic' in litellm_model:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        key_name = "ANTHROPIC_API_KEY"
    elif 'gemini' in litellm_model:
        api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        key_name = "GOOGLE_AI_STUDIO_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        key_name = "OPENAI_API_KEY"

    if not api_key:
        raise ValueError(f"API key not set. Please set {key_name} in your environment.")

    logger.info(f"llm_chat routing: alias={model} model={litellm_model}")

    response = litellm.completion(
        model=litellm_model,
        api_key=api_key,
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ],
    )
    return response.choices[0].message.content
