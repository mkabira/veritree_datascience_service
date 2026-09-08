"""
Tests for the task/pipeline CLI in src/main.py.

The CLI is the container's entry point, so its contract is the deploy contract: the
ECS task definitions invoke `python3 src/main.py pipeline pipeline_api`. These cover
the dispatch surface and the failure modes; the process-replacement behaviour is
verified by running it, since it cannot be observed in-process.
"""

import importlib
import sys

import pytest
from click.testing import CliRunner


@pytest.fixture(scope="module")
def cli_module():
    """
    Import src/main.py the way the script runs it, with src/ on the path.

    :return: the imported module
    """
    src_dir = str(importlib.import_module('src').__path__[0])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    return importlib.import_module('main')


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def rebind_choices(cli_module):
    """
    Temporarily point the CLI's argument choices at a test registry.

    click binds ``click.Choice`` at decoration time, so a test that swaps TASKS or
    PIPELINES must also swap the choices. Assigning to the param object directly is
    not something monkeypatch can undo, so this restores it explicitly -- without
    that, a mutated Choice leaks into every later test in the module.
    """
    import click

    original = {
        'task': cli_module.run_task.params[0].type,
        'pipeline': cli_module.run_pipeline.params[0].type,
    }

    def rebind(tasks=None, pipelines=None):
        if tasks is not None:
            cli_module.run_task.params[0].type = click.Choice(sorted(tasks))
        if pipelines is not None:
            cli_module.run_pipeline.params[0].type = click.Choice(sorted(pipelines))

    yield rebind

    cli_module.run_task.params[0].type = original['task']
    cli_module.run_pipeline.params[0].type = original['pipeline']


class TestRegistry:

    def test_the_deploy_contract_is_registered(self, cli_module):
        """The Dockerfile CMD and the ECS task definitions invoke this by name."""
        assert 'pipeline_api' in cli_module.PIPELINES
        assert 'task_launch_api' in cli_module.TASKS

    def test_every_pipeline_names_only_registered_tasks(self, cli_module):
        for pipeline, tasks in cli_module.PIPELINES.items():
            unknown = [t for t in tasks if t not in cli_module.TASKS]
            assert not unknown, f"{pipeline} references unregistered {unknown}"

    def test_every_task_is_callable(self, cli_module):
        for name, task in cli_module.TASKS.items():
            assert callable(task), name

    def test_terminal_tasks_are_registered_tasks(self, cli_module):
        assert cli_module.TERMINAL_TASKS <= set(cli_module.TASKS)


class TestDispatch:

    def test_list_shows_tasks_and_pipelines(self, cli_module, runner):
        result = runner.invoke(cli_module.cli, ['list'])

        assert result.exit_code == 0
        assert 'task_launch_api' in result.output
        assert 'pipeline_api' in result.output

    def test_a_pipeline_runs_its_tasks_in_order(self, cli_module, runner, monkeypatch,
                                                rebind_choices):
        ran = []
        monkeypatch.setattr(cli_module, 'TASKS',
                            {'a': lambda: ran.append('a'), 'b': lambda: ran.append('b')})
        monkeypatch.setattr(cli_module, 'PIPELINES', {'p': ['a', 'b']})
        monkeypatch.setattr(cli_module, 'TERMINAL_TASKS', set())
        rebind_choices(pipelines={'p'})

        assert runner.invoke(cli_module.cli, ['pipeline', 'p']).exit_code == 0
        assert ran == ['a', 'b']


class TestFailureModes:
    """
    The previous implementation exited 0 for every one of these, having silently done
    nothing — the worst outcome for a container entry point.
    """

    def test_an_unknown_subcommand_exits_non_zero(self, cli_module, runner):
        result = runner.invoke(cli_module.cli, ['tsak', 'pipeline_api'])

        assert result.exit_code != 0
        assert 'No such command' in result.output

    def test_an_unknown_pipeline_exits_non_zero_and_lists_the_options(
            self, cli_module, runner):
        result = runner.invoke(cli_module.cli, ['pipeline', 'nope'])

        assert result.exit_code != 0
        assert 'pipeline_api' in result.output

    def test_an_unknown_task_exits_non_zero(self, cli_module, runner):
        assert runner.invoke(cli_module.cli, ['task', 'nope']).exit_code != 0

    def test_a_missing_argument_exits_non_zero(self, cli_module, runner):
        assert runner.invoke(cli_module.cli, ['pipeline']).exit_code != 0

    def test_a_failing_task_exits_non_zero_without_a_traceback(
            self, cli_module, runner, monkeypatch):
        def boom():
            raise RuntimeError("upstream exploded")

        monkeypatch.setattr(cli_module, 'TASKS', {'task_launch_api': boom})

        result = runner.invoke(cli_module.cli, ['task', 'task_launch_api'])

        assert result.exit_code != 0
        assert 'upstream exploded' in result.output
        assert 'Traceback' not in result.output

    def test_a_terminal_task_may_not_precede_others(self, cli_module, runner,
                                                    monkeypatch, rebind_choices):
        """Anything after a process-replacing task would silently never run."""
        monkeypatch.setattr(cli_module, 'TASKS',
                            {'task_launch_api': lambda: None, 'after': lambda: None})
        monkeypatch.setattr(cli_module, 'PIPELINES',
                            {'bad': ['task_launch_api', 'after']})
        rebind_choices(pipelines={'bad'})

        result = runner.invoke(cli_module.cli, ['pipeline', 'bad'])

        assert result.exit_code != 0
        assert 'must be the final task' in result.output


class TestLauncher:

    def test_the_api_launcher_uses_the_running_interpreter(self, cli_module):
        """
        Not the `uvicorn` console script: that is only on PATH when the environment is
        activated, so a bare execvp fails with ENOENT under cron or a container whose
        PATH differs.
        """
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(cli_module.task_launch_api))
        function = ast.parse(source).body[0]

        # drop the docstring: it names subprocess.run to explain why it is not used
        body = function.body[1:] if ast.get_docstring(function) else function.body
        code = "\n".join(ast.unparse(node) for node in body)

        assert 'sys.executable' in code
        assert 'os.execv(' in code
        assert 'subprocess' not in code, (
            "subprocess.run leaves uvicorn as a child, which never receives the SIGTERM "
            "ECS sends to PID 1")
