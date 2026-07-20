import __init_path__  # noqa: F401

from src.audio.assemblyai_vad import AssemblyAIUniversalVAD


def make_vad(**kwargs):
    return AssemblyAIUniversalVAD(
        src_lang="EN",
        tgt_lang="ZH",
        api_token="test-key",
        **kwargs,
    )


def _word(start, end, speaker="A"):
    return {"text": "x", "start": start, "end": end, "speaker": speaker}


def test_words_split_on_silence_gap():
    # Two bursts separated by a >max_gap silence become two segments.
    vad = make_vad(min_segment_seconds=0.0, max_gap_seconds=0.6)
    srt = vad._build_srt(
        {
            "words": [
                _word(0, 400),
                _word(450, 900),      # gap 0.05s -> same segment
                _word(2000, 2400),    # gap 1.1s  -> new segment
                _word(2450, 2800),
            ]
        }
    )
    assert len(srt.segments) == 2
    assert srt.segments[0].start_time == 0.0
    assert srt.segments[0].end_time == 0.9
    assert srt.segments[1].start_time == 2.0
    assert srt.segments[1].end_time == 2.8


def test_words_split_on_speaker_change():
    vad = make_vad(min_segment_seconds=0.0, max_gap_seconds=5.0)
    srt = vad._build_srt(
        {
            "words": [
                _word(0, 400, "A"),
                _word(450, 900, "A"),
                _word(950, 1400, "B"),  # speaker change -> new segment
            ]
        }
    )
    assert len(srt.segments) == 2
    assert srt.segments[0].speaker == "A"
    assert srt.segments[1].speaker == "B"


def test_words_split_on_max_segment_length():
    vad = make_vad(min_segment_seconds=0.0, max_gap_seconds=100.0, max_segment_seconds=2.0)
    srt = vad._build_srt(
        {
            "words": [
                _word(0, 500),
                _word(600, 1000),
                _word(1100, 1500),
                _word(1600, 2600),   # would exceed 2.0s cap -> new segment
            ]
        }
    )
    assert len(srt.segments) == 2
    assert (srt.segments[0].end_time - srt.segments[0].start_time) <= 2.0


def test_min_segment_seconds_drops_short_segments():
    vad = make_vad(min_segment_seconds=0.8, max_gap_seconds=0.3)
    srt = vad._build_srt(
        {
            "words": [
                _word(0, 200),        # 0.2s isolated burst -> dropped
                _word(2000, 3000),    # 1.0s burst -> kept
            ]
        }
    )
    assert len(srt.segments) == 1
    assert srt.segments[0].start_time == 2.0


def test_dominant_speaker_wins_within_segment():
    vad = make_vad(min_segment_seconds=0.0, max_gap_seconds=5.0, speaker_labels=False)
    srt = vad._build_srt(
        {
            "words": [
                _word(0, 400, "A"),
                _word(450, 900, "A"),
                _word(950, 1400, "B"),
            ]
        }
    )
    # speaker_labels disabled -> no speaker-change split; majority speaker "A".
    assert len(srt.segments) == 1
    assert srt.segments[0].speaker == "A"


def test_empty_words_yields_no_segments():
    vad = make_vad()
    srt = vad._build_srt({"words": []})
    assert srt.segments == []


def test_language_code_resolution():
    assert make_vad().language_code == "en"
    assert make_vad(language_codes="ZH").language_code == "zh"
    # Unmapped language falls back to automatic detection (None).
    assert make_vad(language_codes="xx").language_code == "xx"


def test_legacy_streaming_kwargs_are_accepted_and_ignored():
    # The web app / old configs may still pass realtime/chunk_ms/etc.
    vad = make_vad(realtime=True, chunk_ms=50, mode="balanced")
    assert vad._legacy_streaming_options["realtime"] is True
