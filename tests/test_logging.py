"""
Tests for the logging convention documented in src/utils/context.py.

Logs are the only view into a running task. These guard the properties that make them
usable: a consistent shape, a correlation id on request-scoped lines, bounded growth,
and no credentials.
"""

import ast
import os
import re

import pytest

from src.utils import context


SKIP_DIRS = {'.git', '__pycache__', '.idea', 'models', 'notebooks', 'tests', 'venv'}


def _logger_calls():
    """
    Yield (relative path, line number, level, message text) for every logger call.

    :return: iterator over the service's logging statements
    """
    for root, dirs, files in os.walk(context.root_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if not name.endswith('.py'):
                continue
            path = os.path.join(root, name)
            source = open(path).read()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            lines = source.splitlines()
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'logger'
                        and node.func.attr in {'debug', 'info', 'warning', 'error'}):
                    text = " ".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
                    yield (os.path.relpath(path, context.root_dir),
                           node.lineno, node.func.attr, text)


class TestConvention:

    def test_there_are_log_statements_to_check(self):
        assert len(list(_logger_calls())) > 40, "the walker found suspiciously few calls"

    def test_no_shouting(self):
        """Upper case is reserved for acronyms; SHOUTED lines came from the source repos."""
        offenders = [f"{p}:{n} {t.strip()[:70]}"
                     for p, n, _, t in _logger_calls()
                     if re.search(r'\b[A-Z]{4,}\b', t.split('(', 1)[-1])
                     and not re.search(r'AICM|AWS_|JSON|HTTP|YOLO|SSH|URI|GPS', t)]
        assert not offenders, offenders

    def test_no_exclamation_marks(self):
        offenders = [f"{p}:{n}" for p, n, _, t in _logger_calls()
                     if '!' in t.split('(', 1)[-1]]
        assert not offenders, offenders

    def test_durations_are_formatted_consistently(self):
        """`duration=1.60s`, never `0.0mins or : 1.6secs`."""
        offenders = [f"{p}:{n} {t.strip()[:70]}" for p, n, _, t in _logger_calls()
                     if 'duration' in t and 'duration={' not in t]
        assert not offenders, offenders

    def test_no_credential_is_logged(self):
        """A password reaches the log only through SQLAlchemy's masked URL rendering."""
        offenders = []
        for path, line, _, text in _logger_calls():
            message = text.split('(', 1)[-1]
            if re.search(r'\bpassword\b|\bsecret\b|API_KEY|access_key', message, re.I):
                if 'render_as_string' not in message:
                    offenders.append(f"{path}:{line} {message.strip()[:70]}")
        assert not offenders, offenders


class TestCorrelation:
    """
    Handlers run concurrently in a threadpool, so without an id per line two
    interleaved requests cannot be told apart in the log.
    """

    ROUTERS = ['api/routers/computer_vision.py', 'api/routers/analyses.py',
               'api/routers/datascience_results.py']

    @pytest.mark.parametrize("router", ROUTERS)
    def test_every_line_that_can_correlate_does(self, router):
        """
        The rule is precise rather than a list of exceptions: a log statement inside a
        function that has a correlation id in scope must include it. Module-level and
        infrastructure lines have no request to correlate with and are exempt by
        construction, not by name.
        """
        source = open(os.path.join(context.root_dir, router)).read()
        tree = ast.parse(source)
        lines = source.splitlines()
        missing = []

        for function in [n for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            body = "\n".join(lines[function.lineno - 1:function.end_lineno])
            if 'session_id' not in body and 'request_id' not in body:
                continue   # nothing to correlate with in this scope

            for node in ast.walk(function):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'logger'
                        and node.func.attr in {'debug', 'info', 'warning', 'error'}):
                    text = " ".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
                    if 'session_id=' not in text and 'request_id=' not in text:
                        missing.append(f"{router}:{node.lineno} in {function.name}(): "
                                       f"{text.strip()[:60]}")

        assert not missing, missing

    def test_every_route_generates_an_id(self):
        """A correlation id is useless if a route never mints one."""
        for router in self.ROUTERS:
            source = open(os.path.join(context.root_dir, router)).read()
            assert 'uuid.uuid4()' in source, f"{router} mints no correlation id"


class TestRotation:
    """Without rotation the log grows until it fills the task's ephemeral storage."""

    def test_the_handler_rotates(self):

        handlers = [type(h).__name__ for h in context.logger.handlers]
        assert 'RotatingFileHandler' in handlers, handlers

    def test_stdout_is_also_attached(self):
        """ECS ships stdout to CloudWatch; the file is for exec-ing into a task."""
        import logging

        assert any(isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler)
                   for h in context.logger.handlers)

    def test_disk_usage_is_bounded(self):
        settings = context._logging_settings()
        ceiling = settings['max_bytes'] * (settings['backup_count'] + 1)

        assert 0 < ceiling <= 500 * 1024 * 1024, f"{ceiling} bytes is not a sane ceiling"

    def test_handlers_are_not_stacked_on_repeated_import(self):
        before = len(context.logger.handlers)
        context.load_logging()
        assert len(context.logger.handlers) == before
