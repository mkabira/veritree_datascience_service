"""
Route-level tests for the merged /computer_vision/ endpoint.

The models are stubbed via conftest: loading YOLO, Keras and the Anthropic client
costs seconds and needs credentials. What is under test is the route's contract --
service dispatch, the shared image load, the response envelope, and error mapping.
"""

import pytest


ROUTE = "/computer_vision/"

SERVICES = ["survivability_detection", "content_tagging", "content_moderation"]


class TestAuthentication:

    def test_missing_token_is_rejected(self, cv_client):
        client, _ = cv_client
        response = client.post(ROUTE, json={"service": "content_tagging",
                                            "image_url": "a.jpg"})
        assert response.status_code == 401


class TestServiceDispatch:
    """`service` must select the pipeline, and be echoed so clients can branch on it."""

    @pytest.mark.parametrize("service", SERVICES)
    def test_every_service_is_reachable(self, cv_client, auth, service):
        client, _ = cv_client
        response = client.post(ROUTE, json={"service": service, "image_url": "s/a.jpg"},
                               headers=auth)
        assert response.status_code == 200, response.text
        assert response.json()["service"] == service

    def test_an_unknown_service_is_rejected_before_any_work(self, cv_client, auth):
        """A typo must 422 on the schema, not 500 from a KeyError mid-request."""
        client, fake = cv_client
        response = client.post(ROUTE, json={"service": "survivability",
                                            "image_url": "s/a.jpg"}, headers=auth)
        assert response.status_code == 422
        assert fake.requested == [], "no image should be fetched for an invalid service"

    def test_service_is_required(self, cv_client, auth):
        client, _ = cv_client
        assert client.post(ROUTE, json={"image_url": "s/a.jpg"},
                           headers=auth).status_code == 422

    def test_image_url_is_required(self, cv_client, auth):
        client, _ = cv_client
        assert client.post(ROUTE, json={"service": "content_tagging"},
                           headers=auth).status_code == 422


class TestOrgIdRemoved:
    """
    org_id was required only to build the S3 destination path. The service no longer
    writes back, and the org already appears in the source key, so the field is gone.
    """

    def test_a_caller_still_sending_org_id_is_not_broken(self, cv_client, auth):
        """pydantic drops unknown fields, so old callers need no coordinated deploy."""
        client, _ = cv_client
        response = client.post(ROUTE, json={"service": "content_tagging",
                                            "image_url": "s/a.jpg",
                                            "org_id": "42"}, headers=auth)
        assert response.status_code == 200

    def test_org_id_is_absent_from_the_request_schema(self, cv_client, auth):
        from api.routers.computer_vision import ComputerVisionInput

        assert "org_id" not in ComputerVisionInput.model_fields

    def test_org_id_is_absent_from_the_response(self, cv_client, auth):
        client, _ = cv_client
        body = client.post(ROUTE, json={"service": "content_tagging",
                                        "image_url": "s/a.jpg"}, headers=auth).json()
        assert "org_id" not in body


class TestSharedEnvelope:

    @pytest.mark.parametrize("service", SERVICES)
    def test_envelope_fields_are_the_same_for_every_service(self, cv_client, auth, service):
        client, _ = cv_client
        body = client.post(ROUTE, json={"service": service, "image_url": "s/a.jpg"},
                           headers=auth).json()

        assert set(body) == {"service", "session_id", "image_url", "image", "result"}
        assert body["image_url"] == "s/a.jpg"
        assert body["session_id"]

    @pytest.mark.parametrize("service", SERVICES)
    def test_image_dimensions_are_always_reported(self, cv_client, auth, service):
        """Boxes are in pixels, so clients need the frame they were measured in."""
        client, _ = cv_client
        body = client.post(ROUTE, json={"service": service, "image_url": "s/a.jpg"},
                           headers=auth).json()

        assert body["image"] == {"width": 1920, "height": 1080}

    @pytest.mark.parametrize("service", SERVICES)
    def test_the_source_is_read_exactly_once(self, cv_client, auth, service):
        """One read per request whichever service runs; nothing is written back."""
        client, fake = cv_client
        client.post(ROUTE, json={"service": service, "image_url": "s/a.jpg"}, headers=auth)
        assert fake.requested == ["s/a.jpg"]


class TestSurvivabilityDetection:

    def test_returns_counts_and_boxes(self, cv_client, auth):
        client, _ = cv_client
        result = client.post(ROUTE, json={"service": "survivability_detection",
                                          "image_url": "s/a.jpg"}, headers=auth).json()["result"]

        assert result["counts"]["number_mangroves"] == 2
        assert result["counts"]["number_alive"] == 1
        assert result["counts"]["number_dead"] == 1
        assert len(result["detections"]) == 2

    def test_boxes_come_in_both_coordinate_systems(self, cv_client, auth):
        client, _ = cv_client
        detection = client.post(ROUTE, json={"service": "survivability_detection",
                                             "image_url": "s/a.jpg"},
                                headers=auth).json()["result"]["detections"][0]

        assert len(detection["bbox_xyxy"]) == 4
        assert len(detection["bbox_xywhn"]) == 4
        assert all(0.0 <= v <= 1.0 for v in detection["bbox_xywhn"])


class TestContentTagging:

    def test_returns_tags_and_scores(self, cv_client, auth):
        client, _ = cv_client
        result = client.post(ROUTE, json={"service": "content_tagging",
                                          "image_url": "s/a.jpg"}, headers=auth).json()["result"]

        assert "tags" in result and "scores" in result

    def test_detecting_people_chains_into_moderation(self, cv_client, auth, monkeypatch):
        """The AICM chain from the deployed service must survive the merge."""
        from api.routers import computer_vision

        calls = []

        def fake_runner(**kwargs):
            calls.append(kwargs["model_name"])
            if len(calls) == 1:
                return ["people"], {"people": 0.97}
            return ["minor_flagged"], {"minor_flagged": 0.81}

        monkeypatch.setattr(computer_vision, "run_veritag_anthropic", fake_runner)

        client, _ = cv_client
        body = client.post(ROUTE, json={"service": "content_tagging",
                                        "image_url": "s/a.jpg"},
                           headers=auth).json()["result"]

        assert len(calls) == 2, "moderation should have been invoked as a second pass"
        assert "people" in body["tags"]
        assert "minor_flagged" in body["tags"], "moderation tags must be merged in"

    def test_no_people_means_no_moderation_pass(self, cv_client, auth, monkeypatch):
        from api.routers import computer_vision

        calls = []

        def fake_runner(**kwargs):
            calls.append(kwargs["model_name"])
            return ["meterstick"], {"meterstick": 0.9}

        monkeypatch.setattr(computer_vision, "run_veritag_anthropic", fake_runner)

        client, _ = cv_client
        client.post(ROUTE, json={"service": "content_tagging",
                                 "image_url": "s/a.jpg"}, headers=auth)

        assert len(calls) == 1, "moderation should be skipped when no people are present"


class TestContentModeration:

    def test_reports_whether_the_image_is_flagged(self, cv_client, auth):
        client, _ = cv_client
        result = client.post(ROUTE, json={"service": "content_moderation",
                                          "image_url": "s/a.jpg"}, headers=auth).json()["result"]

        assert "flagged" in result
        assert isinstance(result["flagged"], bool)


class TestImageLoading:
    """One loading path serves both the S3 keys and https URLs in use today."""

    def test_an_s3_key_is_read_through_the_bucket_handler(self, cv_client, auth):
        client, fake = cv_client
        client.post(ROUTE, json={"service": "content_tagging",
                                 "image_url": "survivability/org/img.jpg"}, headers=auth)
        assert fake.requested == ["survivability/org/img.jpg"]

    def test_an_https_url_bypasses_the_bucket_handler(self, cv_client, auth, monkeypatch):
        from api.routers import computer_vision
        from PIL import Image

        fetched = []

        def fake_load_photo(url, show=False):
            fetched.append(url)
            return Image.new("RGB", (1920, 1080))

        monkeypatch.setattr(computer_vision, "load_photo", fake_load_photo)

        client, fake = cv_client
        url = "https://veritreephotos.s3.us-east-2.amazonaws.com/bulk/a.jpeg"
        response = client.post(ROUTE, json={"service": "content_tagging",
                                            "image_url": url}, headers=auth)

        assert response.status_code == 200
        assert fetched == [url]
        assert fake.requested == [], "an https URL must not go through the S3 handler"


class TestFailureModes:

    def test_unreadable_source_is_502(self, cv_client, auth):
        """Upstream storage failure is distinguishable from a bug in this service."""
        client, fake = cv_client
        fake.payload = RuntimeError("NoSuchKey")

        response = client.post(ROUTE, json={"service": "content_tagging",
                                            "image_url": "s/missing.jpg"}, headers=auth)
        assert response.status_code == 502
        assert "Unable to read image" in response.json()["detail"]

    def test_a_source_that_is_not_an_image_is_400(self, cv_client, auth):
        """The object was fetched fine; the caller pointed at the wrong thing."""
        client, fake = cv_client
        fake.payload = b"this is not an image"

        response = client.post(ROUTE, json={"service": "content_tagging",
                                            "image_url": "s/notes.txt"}, headers=auth)
        assert response.status_code == 400
        assert "not a readable image" in response.json()["detail"]


class TestSupersededRoutes:

    @pytest.mark.parametrize("path", [
        "/computer_vision/survivability_classifier/",
        "/computer_vision/content_tagging/",
        "/computer_vision/content_moderation/",
    ])
    def test_the_per_service_routes_are_gone(self, cv_client, auth, path):
        client, _ = cv_client
        assert client.post(path, json={"image_url": "a.jpg"},
                           headers=auth).status_code == 404


class TestOpenApiDocumentation:
    """
    The /docs page is the contract other teams read. These pin the parts that would
    otherwise rot silently: a new service added to SERVICES but not to the examples,
    or an error code the route raises but never documents.
    """

    def _schema(self, cv_client):
        client, _ = cv_client
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routers import computer_vision

        app = FastAPI()
        app.include_router(computer_vision.router)
        return TestClient(app).get("/openapi.json").json()

    def test_every_service_has_a_worked_example(self, cv_client):
        """Each service needs a runnable example in Swagger's Try it out dropdown."""
        schema = self._schema(cv_client)
        examples = schema["paths"]["/computer_vision/"]["post"]["requestBody"][
            "content"]["application/json"]["examples"]

        documented = {e["value"]["service"] for e in examples.values()}
        assert set(SERVICES) <= documented, f"undocumented services: {set(SERVICES) - documented}"

    def test_the_service_enum_is_published(self, cv_client):
        """Clients should discover the valid values from the schema, not the source."""
        schema = self._schema(cv_client)
        field = schema["components"]["schemas"]["ComputerVisionInput"]["properties"]["service"]
        assert set(field["enum"]) == set(SERVICES)

    @pytest.mark.parametrize("code", ["400", "401", "422", "502"])
    def test_error_responses_are_documented(self, cv_client, code):
        schema = self._schema(cv_client)
        responses = schema["paths"]["/computer_vision/"]["post"]["responses"]
        assert code in responses
        assert responses[code]["description"]

    def test_the_endpoint_has_a_summary(self, cv_client):
        schema = self._schema(cv_client)
        assert schema["paths"]["/computer_vision/"]["post"]["summary"]

    @pytest.mark.parametrize("model", ["SurvivabilityResult", "TaggingResult", "ModerationResult"])
    def test_each_result_shape_carries_an_example(self, cv_client, model):
        """The result is a union, so a worked example per shape is what makes it legible."""
        schema = self._schema(cv_client)
        assert "example" in schema["components"]["schemas"][model]
