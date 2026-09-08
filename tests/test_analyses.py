"""Route-level tests for /analyses."""

import json

import pytest


ROUTE = "/analyses/verification_summarization/"


# A MISSING header is rejected by FastAPI's own APIKeyHeader, whose status code is a
# framework detail: 403 up to fastapi 0.115.x, 401 from ~0.14x. Asserting either exact
# code ties the suite to a pinned version, so these check the property that matters --
# the request is refused. A WRONG token is our own code path and is pinned to 401.
UNAUTHENTICATED = {401, 403}


class TestAuthentication:

    def test_missing_token_is_rejected(self, analyses_client, rule_payload):
        client, _ = analyses_client
        assert client.post(ROUTE, json=rule_payload).status_code in UNAUTHENTICATED

    def test_wrong_token_is_rejected(self, analyses_client, rule_payload):
        client, _ = analyses_client
        response = client.post(ROUTE, json=rule_payload, headers={"Token": "nope"})
        assert response.status_code == 401


class TestVerificationSummarization:

    def test_returns_a_flag_per_rule(self, analyses_client, auth, rule_payload):
        client, stub = analyses_client
        stub.response = json.dumps([
            {"rule_public_id": "rule_abc123", "message": "Please retake the photo with the meterstick visible."}
        ])

        body = client.post(ROUTE, json=rule_payload, headers=auth).json()

        assert body["suggested_flags"][0]["rule_public_id"] == "rule_abc123"
        assert "meterstick" in body["suggested_flags"][0]["message"]
        assert body["session_id"]

    def test_uses_the_configured_model_and_temperature(self, analyses_client, auth, rule_payload):
        from src.utils import context
        client, stub = analyses_client
        stub.response = "[]"

        client.post(ROUTE, json=rule_payload, headers=auth)

        assert stub.calls[0]["model"] == context.config.verification_summarization.model
        assert stub.calls[0]["temperature"] == float(context.config.verification_summarization.temperature)

    def test_rule_data_reaches_the_prompt(self, analyses_client, auth, rule_payload):
        client, stub = analyses_client
        stub.response = "[]"

        client.post(ROUTE, json=rule_payload, headers=auth)

        prompt = stub.calls[0]["user_prompt"]
        assert "rule_abc123" in prompt
        assert "stick was out of frame" in prompt, "rule comments must reach the model"

    @pytest.mark.parametrize("fenced", [
        '```json\n[{"rule_public_id": "r1", "message": "m"}]\n```',
        '```\n[{"rule_public_id": "r1", "message": "m"}]\n```',
        '[{"rule_public_id": "r1", "message": "m"}]',
    ])
    def test_markdown_code_fences_are_stripped(self, analyses_client, auth, rule_payload, fenced):
        """Models wrap JSON in fences inconsistently; all three forms must parse."""
        client, stub = analyses_client
        stub.response = fenced

        body = client.post(ROUTE, json=rule_payload, headers=auth).json()

        assert body["suggested_flags"] == [{"rule_public_id": "r1", "message": "m"}]

    def test_empty_result_is_valid(self, analyses_client, auth, rule_payload):
        client, stub = analyses_client
        stub.response = "[]"

        body = client.post(ROUTE, json=rule_payload, headers=auth).json()

        assert body["suggested_flags"] == []


class TestFailureModes:

    def test_non_json_model_output_is_a_500_not_a_crash(self, analyses_client, auth, rule_payload):
        client, stub = analyses_client
        stub.response = "I could not process that request."

        assert client.post(ROUTE, json=rule_payload, headers=auth).status_code == 500

    def test_json_object_instead_of_list_is_rejected(self, analyses_client, auth, rule_payload):
        client, stub = analyses_client
        stub.response = '{"rule_public_id": "r1", "message": "m"}'

        response = client.post(ROUTE, json=rule_payload, headers=auth)
        assert response.status_code == 500
        assert "unexpected response type" in response.json()["detail"]

    def test_provider_error_surfaces_as_500(self, analyses_client, auth, rule_payload):
        client, stub = analyses_client
        stub.response = RuntimeError("provider timeout")

        assert client.post(ROUTE, json=rule_payload, headers=auth).status_code == 500

    def test_malformed_request_body_is_422(self, analyses_client, auth):
        client, _ = analyses_client
        response = client.post(ROUTE, json={"rules": [{"name": "missing required fields"}]},
                               headers=auth)
        assert response.status_code == 422
