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
        assert context.config.aicm_anthropic.model_name           # config_cv_content_moderation.yaml
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


class TestComputerVisionConfig:
    """
    Guards the config_cv_*.yaml split.

    These assert that each capability's config carries every field the routers read.
    A truncated or mis-split file (e.g. a source file whose last line lacks a trailing
    newline) drops keys silently at load time and only surfaces as an AttributeError
    on the first request; these tests fail at build time instead.
    """

    # Anthropic is the only content-tagging backend; the others were retired.
    PROVIDERS = ["anthropic"]

    def test_survivability_pipeline_is_complete(self):
        pipeline = context.config.pipeline
        for field in ["object_model_path", "survival_model_path", "confidence"]:
            assert field in pipeline, f"pipeline.{field} missing"

        # The service reads images from S3 into memory; the batch pipeline's local
        # path settings (image_path, save_dir, results_csv_path) were removed with it.
        for field in ["image_path", "save_dir", "results_csv_path", "save_cropped_mangroves"]:
            assert field not in pipeline, f"pipeline.{field} is dead config"

        for field in ["border", "pad_fraction", "min_size", "input_width", "input_height"]:
            assert field in context.config.pipeline_meta, f"pipeline_meta.{field} missing"

        assert set(context.config.classes) == {"Alive", "Dead/dormant", "Unclear"}

    def test_model_weights_referenced_by_config_exist_on_disk(self):
        for field in ["object_model_path", "survival_model_path"]:
            path = context.root_dir + context.config.pipeline[field]
            assert os.path.exists(path), f"pipeline.{field} points at missing {path}"

    def test_content_tagging_is_configured(self):
        cfg = context.config.content_tagging_anthropic
        assert cfg.model_name
        assert cfg.verification_tags
        assert cfg.system_prompt
        assert cfg.verification_threshold is not None

    @pytest.mark.parametrize("retired", ["cvmodel", "gemini", "openai"])
    def test_retired_provider_configs_are_gone(self, retired):
        """Anthropic measured best; leaving dead provider blocks invites drift."""
        assert f"content_tagging_{retired}" not in context.config

    def test_content_moderation_is_fully_configured(self):
        """The threshold is the field the router compares against; it must be present."""
        cfg = context.config.aicm_anthropic
        assert cfg.model_name
        assert cfg.system_prompt
        assert cfg.verification_tags
        assert cfg.verification_threshold is not None
        assert 0.0 < float(cfg.verification_threshold) < 1.0

    def test_verification_summarization_is_configured(self):
        cfg = context.config.verification_summarization
        assert cfg.model
        assert cfg.system_prompt
        assert cfg.user_prompt_template
        assert cfg.temperature is not None

    def test_each_capability_lives_in_its_own_file(self):
        """The four-way split must stay a split -- no file may reclaim another's keys."""
        expected = {
            "config_cv_survivability.yaml":
                {"pipeline", "pipeline_meta", "classes"},
            "config_cv_content_tagging.yaml":
                {f"content_tagging_{p}" for p in self.PROVIDERS},
            "config_cv_content_moderation.yaml":
                {"aicm_anthropic"},
            "config_cv_verification_summarization.yaml":
                {"verification_summarization"},
        }
        config_dir = os.path.join(context.root_dir, "configs")
        for name, keys in expected.items():
            with open(os.path.join(config_dir, name)) as f:
                assert set(yaml.safe_load(f)) == keys, f"{name} does not own exactly {keys}"
