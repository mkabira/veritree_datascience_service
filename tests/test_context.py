"""Tests for configuration loading and merging."""

import os

import pytest
import yaml

from src.utils import context


class TestConfigMerge:

    def test_service_identity_is_loaded(self):
        assert context.config.repo.name == "veritree_datascience_service"

    def test_keys_from_every_config_file_share_one_namespace(self):
        """config.<key> must work regardless of which configs/*.yaml defines it."""
        assert context.config.repo is not None                    # config.yaml
        assert context.config.aicm_anthropic.model_name           # config_computer_vision.yaml
        assert context.config.datascience_results.bioacoustics    # config_datascience_results.yaml

    def test_every_config_file_parses(self):
        config_dir = os.path.join(context.root_dir, "configs")
        for name in os.listdir(config_dir):
            if name.endswith(".yaml"):
                with open(os.path.join(config_dir, name)) as f:
                    assert yaml.safe_load(f) is not None, f"{name} parsed as empty"

    def test_no_key_is_defined_in_two_files(self):
        config_dir = os.path.join(context.root_dir, "configs")
        seen = {}
        for name in sorted(os.listdir(config_dir)):
            if not name.endswith(".yaml"):
                continue
            with open(os.path.join(config_dir, name)) as f:
                for key in (yaml.safe_load(f) or {}):
                    assert key not in seen, f"'{key}' defined in both {seen[key]} and {name}"
                    seen[key] = name

    def test_duplicate_keys_raise(self, tmp_path):
        (tmp_path / "a.yaml").write_text("shared_key: 1\n")
        (tmp_path / "b.yaml").write_text("shared_key: 2\n")
        with pytest.raises(ValueError, match="Duplicate configuration keys"):
            context.load_config(str(tmp_path))


class TestLogging:

    def test_handlers_are_not_stacked_on_repeated_load(self):
        """context is imported under two identities in one run; handlers must attach once."""
        logger = context.load_logging()
        before = len(logger.handlers)
        context.load_logging()
        assert len(logger.handlers) == before
