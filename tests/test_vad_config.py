from pathlib import Path

import pytest

import __init_path__  # noqa: F401

from entries.config_schema import TaskConfig
from src.audio import vad_agent


class DummyVAD:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("provider", "model", "expected_cls"),
    [
        ("assemblyai", "universal-3-5-pro", "assemblyai"),
        ("pyannote_api", "precision-2", "api"),
        ("pyannote_local", "pyannote/speaker-diarization-3.1", "local"),
        (None, "API", "api"),
        (None, "pyannote/speaker-diarization-3.1", "local"),
    ],
)
def test_create_vad_dispatch(monkeypatch, provider, model, expected_cls):
    created = {}

    def make_factory(name):
        def factory(**kwargs):
            created["provider"] = name
            return DummyVAD(**kwargs)

        return factory

    monkeypatch.setattr(vad_agent, "AssemblyAIUniversalVAD", make_factory("assemblyai"))
    monkeypatch.setattr(vad_agent, "APIPyannoteVAD", make_factory("api"))
    monkeypatch.setattr(vad_agent, "LocalPyannoteVAD", make_factory("local"))

    vad = vad_agent.create_vad(
        provider=provider,
        model_name_or_path=model,
        src_lang="EN",
        tgt_lang="ZH",
        min_segment_seconds=0.5,
    )

    assert isinstance(vad, DummyVAD)
    assert created["provider"] == expected_cls
    assert vad.kwargs["src_lang"] == "EN"
    assert vad.kwargs["tgt_lang"] == "ZH"
    assert vad.kwargs["min_segment_seconds"] == 0.5


def test_create_vad_rejects_unknown_provider():
    with pytest.raises(ValueError, match="VAD_provider"):
        vad_agent.create_vad(
            provider="unknown",
            model_name_or_path="universal-3-5-pro",
            src_lang="EN",
            tgt_lang="ZH",
        )


def test_bundled_task_config_uses_assemblyai_default():
    config = TaskConfig.from_yaml_file(Path("configs/task_config.yaml"))

    assert config.audio.VAD_provider == "assemblyai"
    assert config.audio.VAD_model == "universal-3-5-pro"
    assert config.audio.VAD_options == {"speaker_labels": True}
    assert config.audio.min_segment_seconds == 0.8


def test_legacy_api_vad_model_config_still_validates():
    config = TaskConfig(
        audio={
            "enable_audio": True,
            "audio_agent": "WhisperAudioAgent",
            "VAD_model": "API",
            "src_lang": "EN",
            "tgt_lang": "ZH",
        }
    )

    assert config.audio.VAD_provider is None
    assert config.audio.VAD_model == "API"
