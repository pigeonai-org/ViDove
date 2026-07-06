import os
from pathlib import Path

import pytest
from pydub import AudioSegment

import __init_path__  # noqa: F401

from src.audio.assemblyai_vad import AssemblyAIUniversalVAD


def make_vad(**kwargs):
    return AssemblyAIUniversalVAD(
        model_name_or_path="universal-3-5-pro",
        src_lang="EN",
        tgt_lang="ZH",
        api_token="test-key",
        **kwargs,
    )


def test_messages_to_srt_uses_speech_started_and_final_turn():
    vad = make_vad(min_segment_seconds=0.0)

    srt = vad.messages_to_srt(
        [
            {"type": "SpeechStarted", "timestamp": 1200, "confidence": 0.9},
            {
                "type": "Turn",
                "turn_order": 0,
                "end_of_turn": True,
                "speaker_label": "A",
                "words": [
                    {"text": "Hello", "start": 1300, "end": 1700, "word_is_final": True},
                    {"text": "world", "start": 1750, "end": 2300, "word_is_final": True},
                ],
            },
        ]
    )

    assert len(srt.segments) == 1
    segment = srt.segments[0]
    assert segment.start_time == pytest.approx(1.2)
    assert segment.end_time == pytest.approx(2.3)
    assert segment.speaker == "A"
    assert segment.src_text == ""


def test_messages_to_srt_falls_back_to_first_word_start():
    vad = make_vad(min_segment_seconds=0.0)

    srt = vad.messages_to_srt(
        [
            {
                "type": "Turn",
                "turn_order": 1,
                "end_of_turn": True,
                "words": [
                    {"text": "No", "start": 500, "end": 700, "word_is_final": True},
                    {"text": "marker", "start": 720, "end": 900, "word_is_final": True},
                ],
            }
        ]
    )

    assert len(srt.segments) == 1
    assert srt.segments[0].start_time == pytest.approx(0.5)
    assert srt.segments[0].end_time == pytest.approx(0.9)


def test_messages_to_srt_skips_empty_words():
    vad = make_vad(min_segment_seconds=0.0)

    srt = vad.messages_to_srt(
        [{"type": "Turn", "turn_order": 0, "end_of_turn": True, "words": []}]
    )

    assert srt.segments == []


def test_messages_to_srt_skips_short_segments():
    vad = make_vad(min_segment_seconds=0.8)

    srt = vad.messages_to_srt(
        [
            {
                "type": "Turn",
                "turn_order": 0,
                "end_of_turn": True,
                "words": [
                    {"text": "Short", "start": 1000, "end": 1400, "word_is_final": True}
                ],
            }
        ]
    )

    assert srt.segments == []


def test_messages_to_srt_applies_speaker_revision():
    vad = make_vad(min_segment_seconds=0.0)

    srt = vad.messages_to_srt(
        [
            {
                "type": "Turn",
                "turn_order": 3,
                "end_of_turn": True,
                "speaker_label": "A",
                "words": [
                    {"text": "Hello", "start": 0, "end": 500, "word_is_final": True}
                ],
            },
            {
                "type": "SpeakerRevision",
                "revisions": [{"turn_order": 3, "speaker_label": "B"}],
            },
        ]
    )

    assert len(srt.segments) == 1
    assert srt.segments[0].speaker == "B"


def test_iter_audio_chunks_normalizes_to_pcm_16k_mono():
    audio = AudioSegment.silent(duration=120, frame_rate=8000).set_channels(2)

    chunks = list(
        AssemblyAIUniversalVAD.iter_audio_chunks(audio, chunk_ms=50, sample_rate=16000)
    )

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [1600, 1600, 640]


@pytest.mark.skipif(
    not os.getenv("ASSEMBLYAI_API_KEY"), reason="ASSEMBLYAI_API_KEY is not set"
)
def test_live_assemblyai_smoke(tmp_path: Path):
    audio_path = tmp_path / "silence.wav"
    AudioSegment.silent(duration=1000, frame_rate=16000).export(audio_path, format="wav")
    vad = AssemblyAIUniversalVAD(
        model_name_or_path="universal-3-5-pro",
        src_lang="EN",
        tgt_lang="ZH",
        min_segment_seconds=0.0,
        receive_timeout=30,
    )

    srt = vad.get_speaker_segments(str(audio_path))

    assert srt.segments == []
