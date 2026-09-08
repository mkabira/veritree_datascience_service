"""
Tests for the properties that only matter under concurrent load.

Handlers that perform blocking I/O must be sync so FastAPI runs them in a threadpool;
declared `async def`, their blocking body would stall the event loop and every other
request with it -- including the /health probe the ECS target group polls.
"""

import inspect
import threading

import pytest


class TestBlockingHandlersAreSync:
    """
    The regression this guards is silent: adding `async` back still works in
    development and only shows up as health-check flapping under load.
    """

    @pytest.mark.parametrize("module_path,handler", [
        ("api.routers.datascience_results", "bioacoustics_results"),
        ("api.routers.datascience_results", "multispectral_results"),
        ("api.routers.datascience_results", "treetracker_results"),
        ("api.routers.computer_vision", "computer_vision"),
        ("api.routers.analyses", "verification_summarization"),
    ])
    def test_handlers_doing_blocking_io_are_not_coroutines(self, module_path, handler):
        import importlib

        function = getattr(importlib.import_module(module_path), handler)

        assert not inspect.iscoroutinefunction(function), (
            f"{handler} performs blocking I/O; as `async def` it would run on the event "
            f"loop and stall every concurrent request, /health included")

    @pytest.mark.parametrize("handler", ["landing", "health", "favicon"])
    def test_trivial_handlers_stay_async(self, handler):
        """No blocking work, so async avoids a needless threadpool hop."""
        import api.main

        assert inspect.iscoroutinefunction(getattr(api.main, handler))


class TestTunnelRestartIsSerialised:
    """
    Now that handlers run in a threadpool, several requests can find the tunnel dead
    at once. Restarting concurrently would strand the connections another thread just
    established.
    """

    def _handler(self):
        from src.awskit.datahandlers import RDSPostgresHandler

        return RDSPostgresHandler(host="db.example.com", port="5432",
                                  database="analytics", user="u", password="p")

    def test_concurrent_restarts_happen_once(self, monkeypatch):
        handler = self._handler()

        class SlowTunnel:
            def __init__(self):
                self.is_active = False
                self.local_bind_port = 40000
                self.restarts = 0

            def restart(self):
                self.restarts += 1
                # widen the window a real restart would occupy
                threading.Event().wait(0.05)
                self.is_active = True
                self.local_bind_port += 1

        handler.tunnel = SlowTunnel()
        monkeypatch.setattr(handler.engine, "dispose", lambda: None)

        threads = [threading.Thread(target=handler.ensure_ready) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert handler.tunnel.restarts == 1, (
            f"restarted {handler.tunnel.restarts} times; the lock must serialise it "
            f"and the re-check must stop the losers restarting again")

    def test_a_healthy_tunnel_is_never_restarted(self):
        handler = self._handler()

        class Healthy:
            is_active = True
            local_bind_port = 40000
            restarts = 0

            def restart(self):
                Healthy.restarts += 1

        handler.tunnel = Healthy()
        for _ in range(5):
            handler.ensure_ready()

        assert Healthy.restarts == 0

    def test_the_lock_exists_on_the_handler(self):
        handler = self._handler()

        assert isinstance(handler._tunnel_lock, type(threading.Lock()))
