"""
Repository hygiene checks.

The trailing-newline check exists because it has already caused two real defects
while porting from veritree-survivability: a file whose last line lacks a newline
is one line longer than `wc -l` reports, so any line-range extraction (sed -n
'a,bp', slicing splitlines()) silently drops that final line. It cost us
`aicm_anthropic.verification_threshold` in the config and the `return` statement
in llm_chat. Both would have failed only at request time.
"""

import os

import pytest

from src.utils import context


def _repo_files(*extensions):
    skip_dirs = {'.git', '__pycache__', 'venv', '.venv', 'node_modules',
                 '.pytest_cache', 'models', 'logs', 'notebooks'}
    for root, dirs, files in os.walk(context.root_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            if name.endswith(extensions):
                yield os.path.join(root, name)


class TestTrailingNewlines:

    def test_every_source_file_ends_with_a_newline(self):
        """A missing final newline makes line-range edits drop the last line."""
        offenders = []
        for path in _repo_files('.py', '.yaml', '.yml', '.txt', '.md'):
            with open(path, 'rb') as f:
                content = f.read()
            if content and not content.endswith(b'\n'):
                offenders.append(os.path.relpath(path, context.root_dir))

        assert not offenders, (
            "files missing a trailing newline (line-range edits will silently drop "
            f"their last line): {sorted(offenders)}")


class TestSecrets:

    @pytest.mark.parametrize("name", ["credentials.json", ".env", "service_account.json"])
    def test_credential_files_are_gitignored(self, name):
        """veritree_multispectral_foresthealth carries an untracked credentials.json
        that its own .gitignore does not cover; ours must."""
        import subprocess
        result = subprocess.run(
            ["git", "check-ignore", "-q", name],
            cwd=context.root_dir, capture_output=True)
        assert result.returncode == 0, f"{name} is not gitignored"

    def test_no_secrets_are_committed(self):
        for path in _repo_files('.json', '.env'):
            rel = os.path.relpath(path, context.root_dir)
            assert 'credentials' not in rel.lower(), f"credential file present: {rel}"


class TestDependencyCompleteness:
    """
    Every third-party module imported at module level must be declared in
    requirements.txt.

    This exists because `wikipedia` was once dropped while merging the three source
    requirements files, while src/libs/utils.py still imported it at module level --
    which broke startup of the entire computer_vision router, at run time, in a fresh
    environment, long after the edit that caused it. (Both the import and the package
    have since been removed as dead code; the guard remains.)
    """

    # distribution name on PyPI -> module name you import, where they differ
    ALIASES = {
        'opencv-python': 'cv2', 'pillow': 'PIL', 'python-box': 'box',
        'python-dotenv': 'dotenv', 'pyyaml': 'yaml', 'psycopg2-binary': 'psycopg2',
        'open_clip_torch': 'open_clip', 'open-clip-torch': 'open_clip',
        'google-genai': 'google', 'python-multipart': 'multipart', 'gitpython': 'git',
        'requests-oauthlib': 'requests_oauthlib', 'scikit-learn': 'sklearn',
    }

    # resolved via sys.path at run time rather than installed: `src/main.py` runs with
    # src/ as sys.path[0], so `from utils import context` means src/utils
    LOCAL_MODULES = {'src', 'api', 'tests', 'configs', 'utils', 'libs', 'awskit', 'services'}

    def _declared(self):
        import re
        declared = set()
        with open(os.path.join(context.root_dir, 'requirements.txt')) as f:
            for line in f:
                line = line.split('#')[0].strip()
                if not line or line.startswith('-'):
                    continue
                dist = re.split(r'[=<>!~\[]', line)[0].strip().lower()
                declared.add(self.ALIASES.get(dist, dist.replace('-', '_')))
        return declared

    def _imported(self):
        import ast
        import sys
        stdlib = set(sys.stdlib_module_names)
        found = {}
        for path in _repo_files('.py'):
            try:
                tree = ast.parse(open(path).read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name.split('.')[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module.split('.')[0]]
                else:
                    continue
                for module in modules:
                    if module in stdlib or module in self.LOCAL_MODULES:
                        continue
                    found.setdefault(module, set()).add(
                        os.path.relpath(path, context.root_dir))
        return found

    def test_every_imported_package_is_declared(self):
        declared = self._declared()
        undeclared = {m: sorted(f) for m, f in self._imported().items() if m not in declared}

        assert not undeclared, (
            "imported but missing from requirements.txt (will fail at startup in a "
            f"clean environment): {undeclared}")
