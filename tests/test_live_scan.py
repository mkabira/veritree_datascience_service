"""
Tests for the live S3 scan, focused on its time budget.

A scan lists an entire prefix, so its cost grows with the bucket rather than with the
result. Unbounded, one request could hold a threadpool slot until the client gave up.
"""

import time

import pytest


class TestScanDeadline:

    def test_a_fresh_deadline_has_not_expired(self):
        from src.services.live_scan import _Deadline

        _Deadline(300).check("listing")   # must not raise

    def test_an_expired_deadline_raises(self):
        from src.services.live_scan import _Deadline, LiveScanTimeout

        deadline = _Deadline(0.01)
        time.sleep(0.02)

        with pytest.raises(LiveScanTimeout, match="exceeded"):
            deadline.check("listing")

    def test_the_message_says_what_was_in_progress(self):
        from src.services.live_scan import _Deadline, LiveScanTimeout

        deadline = _Deadline(0)

        with pytest.raises(LiveScanTimeout, match="reading headers"):
            deadline.check("reading headers")

    def test_the_message_points_at_the_setting_to_raise(self):
        from src.services.live_scan import _Deadline, LiveScanTimeout

        with pytest.raises(LiveScanTimeout, match="timeout_seconds"):
            _Deadline(0).check("listing")

    def test_the_default_budget_is_five_minutes(self):
        from src.services.live_scan import SCAN_TIMEOUT_SECONDS

        assert SCAN_TIMEOUT_SECONDS == 300

    def test_a_domain_can_override_the_budget(self):
        from src.services.live_scan import _deadline_for

        assert _deadline_for({"timeout_seconds": 42}).seconds == 42

    def test_an_absent_override_falls_back_to_the_default(self):
        from src.services.live_scan import SCAN_TIMEOUT_SECONDS, _deadline_for

        assert _deadline_for({}).seconds == SCAN_TIMEOUT_SECONDS


class TestListingIsBounded:

    def test_the_deadline_is_checked_between_pages(self):
        """
        boto3 calls are not interruptible, so the budget is enforced cooperatively at
        page boundaries -- overshoot is one S3 request, not unbounded.
        """
        from src.services.live_scan import _Deadline, LiveScanTimeout, _list_keys

        class FakePaginator:
            def paginate(self, **kwargs):
                for _ in range(1000):
                    yield {"Contents": [{"Key": "a/b.tif", "Size": 1}]}

        class FakeClient:
            def get_paginator(self, name):
                return FakePaginator()

        deadline = _Deadline(0.05)
        with pytest.raises(LiveScanTimeout):
            for _ in _list_keys(FakeClient(), "bkt", "p/", (".tif",), deadline):
                time.sleep(0.001)

    def test_a_scan_inside_budget_completes(self):
        from src.services.live_scan import _Deadline, _list_keys

        class FakeClient:
            def get_paginator(self, name):
                class P:
                    def paginate(self, **kwargs):
                        yield {"Contents": [{"Key": "a/b.tif", "Size": 1},
                                            {"Key": "a/c.txt", "Size": 1}]}
                return P()

        keys = list(_list_keys(FakeClient(), "bkt", "p/", (".tif",), _Deadline(300)))

        assert len(keys) == 1, "only matching extensions are yielded"


class TestTimeoutSurfacesAsGatewayTimeout:

    def test_the_route_maps_a_scan_timeout_to_504(self, client, auth, monkeypatch):
        """504, not 502: the upstream did not fail, we stopped waiting for it."""
        from api.routers import datascience_results as module
        from src.services.live_scan import LiveScanTimeout

        def slow(**kwargs):
            raise LiveScanTimeout("Live scan exceeded 300s while listing")

        monkeypatch.setattr(module.live_scan_module, "scan_multispectral_rasters", slow)

        response = client.get(
            "/datascience_results/multispectral_results/?live_scan=true", headers=auth)

        assert response.status_code == 504
        assert "exceeded 300s" in response.json()["detail"]
