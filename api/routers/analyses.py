"""
Analyses routes.

  /analyses/verification_summarization/   plain-language flag messages from rule failures

Analyses are LLM- or model-backed computations performed per request over a payload the
caller supplies. They differ from /computer_vision (which requires an image) and from
/datascience_results (which serves rows a pipeline already published).

Ported from veritree-survivability :: api/surv_endpoint.py, where this route sits
alongside the CV routes despite never touching an image.
"""

import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.dependencies import api_authentication

from src.libs.llm import llm_chat

from src.utils import context


config = context.config
logger = context.logger


router = APIRouter(prefix="/analyses", tags=["analyses"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RuleComment(BaseModel):
    """A single comment left on a verification rule."""

    source: str
    actor_id: str
    message: str
    created_at: str


class VerificationRule(BaseModel):
    """One verification rule and its outcome for a planting session."""

    rule_public_id: str
    name: str
    status: str
    failure_reason: Optional[dict] = None
    flagged: bool
    comments: List[RuleComment] = []
    category: Optional[str] = None
    group: Optional[str] = None


class FlagSuggestionInput(BaseModel):
    """Request body: the verification rules to summarise."""

    rules: List[VerificationRule]


class FlagSuggestionResult(BaseModel):
    """A plain-language flag message for one rule."""

    rule_public_id: str
    message: str


class FlagSuggestionOutput(BaseModel):
    """Response body: one suggested flag per rule needing field-team action."""

    session_id: str
    suggested_flags: List[FlagSuggestionResult]


# ---------------------------------------------------------------------------
# verification_summarization
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """
    Remove the markdown code fence some models wrap around JSON output.

    Handles both ``` and ```json openers; returns the text unchanged when unfenced.

    :param text: raw model output
    :return: the text with any surrounding fence removed
    """

    text = text.strip()

    if not text.startswith("```"):
        return text

    return text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()


@router.post(
    "/verification_summarization/",
    response_model=FlagSuggestionOutput,
    summary="Turn verification rule failures into field-team flag messages",
    response_description="One plain-language flag per rule needing action",
    dependencies=[Depends(api_authentication)],
    responses={
        401: {"description": "Missing or invalid `Token` header"},
        422: {"description": "Malformed rule payload"},
        500: {"description": "The model returned something other than a JSON array"},
    },
)
async def verification_summarization(params: FlagSuggestionInput):
    """
    Convert raw verification rule failures into user-friendly flag messages for the
    field team.

    :param params: json payload containing a list of verification rules
    :return: session_id and plain-language flag summary
    """

    logger.info("analyses verification_summarization: ENTRY")

    rule_ids = [rule.rule_public_id for rule in params.rules]
    logger.info(f"verification_summarization received {len(params.rules)} rules: {rule_ids}")

    session_id = str(uuid.uuid4())

    try:
        rules_payload = [
            {
                "rule_public_id": rule.rule_public_id,
                "name": rule.name,
                "status": rule.status,
                "failure_reason": rule.failure_reason,
                "comments": [comment.message for comment in rule.comments],
            }
            for rule in params.rules
        ]

        flag_config = config.verification_summarization
        user_prompt = flag_config.user_prompt_template.format(
            failed_rules=json.dumps(rules_payload, indent=2)
        )

        logger.info(f"verification_summarization: using model='{flag_config.model}'")
        llm_response = llm_chat(
            system_prompt=flag_config.system_prompt,
            user_prompt=user_prompt,
            model=flag_config.model,
            temperature=float(flag_config.temperature),
        )

        llm_results = json.loads(_strip_code_fences(llm_response))

        if not isinstance(llm_results, list):
            raise ValueError(f"LLM returned unexpected response type: {type(llm_results).__name__}")

        suggested_flags = [
            FlagSuggestionResult(rule_public_id=item["rule_public_id"], message=item["message"])
            for item in llm_results
        ]

        logger.info(f"analyses verification_summarization - session_id={session_id}: EXIT")

        return FlagSuggestionOutput(session_id=session_id, suggested_flags=suggested_flags)

    except Exception as e:
        err_string = f"Exception while processing verification_summarization: {str(e)}"
        logger.error(err_string)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_string)
