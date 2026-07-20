from __future__ import annotations

import os
import time
import warnings
from collections import Counter
from logging import getLogger
from typing import Any, Iterable

import requests

from src.SRT.srt import SrtScript, SrtSegment
from src.audio.VAD import VAD

logger = getLogger(__name__)


ASSEMBLYAI_LANGUAGE_CODES = {
    "EN": "en",
    "ES": "es",
    "FR": "fr",
    "DE": "de",
    "IT": "it",
    "PT": "pt",
    "JA": "ja",
    "ZH": "zh",
    "AR": "ar",
}


class AssemblyAIUniversalVAD(VAD):
    """VAD provider backed by AssemblyAI's async (pre-recorded) transcription API.

    The engine only needs speech-segment boundaries and speaker labels for
    downstream ASR/clipping, not AssemblyAI's transcript text. The async REST
    API processes whole files far faster than real time (no streaming
    rate-limit dance), returns word-level timestamps + per-word speaker labels,
    and we rebuild fine-grained speech segments from those words by splitting on
    silence gaps, speaker changes, and a max-length cap.
    """

    base_url = "https://api.assemblyai.com/v2"

    def __init__(
        self,
        model_name_or_path: str = "",
        src_lang: str = "EN",
        tgt_lang: str = "ZH",
        min_segment_seconds: float = 0.8,
        *,
        api_token: str | None = None,
        speech_model: str | None = None,
        speaker_labels: bool = True,
        language_codes: str | list[str] | None = None,
        # Segment reconstruction knobs
        max_gap_seconds: float = 0.6,
        max_segment_seconds: float = 30.0,
        # Networking
        upload_timeout: float = 600.0,
        request_timeout: float = 60.0,
        poll_interval: float = 3.0,
        poll_timeout: float | None = None,
        **legacy_streaming_options: Any,
    ) -> None:
        super().__init__(src_lang, tgt_lang, min_segment_seconds)
        # ``model_name_or_path`` used to select a streaming speech_model
        # (e.g. "universal-3-5-pro"); those ids are not valid for the async API,
        # so keep it only as a hint and let ``speech_model`` (a valid async
        # value or None -> AssemblyAI default) win.
        self.model_name_or_path = model_name_or_path or ""
        self.speech_model = speech_model
        self.api_token = api_token or os.getenv("ASSEMBLYAI_API_KEY")
        self.speaker_labels = bool(speaker_labels)
        self.language_code = self._resolve_language_code(language_codes)
        self.max_gap_seconds = float(max_gap_seconds)
        self.max_segment_seconds = float(max_segment_seconds)
        self.upload_timeout = float(upload_timeout)
        self.request_timeout = float(request_timeout)
        self.poll_interval = max(1.0, float(poll_interval))
        self.poll_timeout = poll_timeout
        # Accepted for backwards compatibility with the old streaming provider
        # (realtime, chunk_ms, mode, ...); ignored by the async implementation.
        self._legacy_streaming_options = dict(legacy_streaming_options)

        if not self.api_token:
            raise ValueError(
                "Set ASSEMBLYAI_API_KEY or pass api_token to use AssemblyAI VAD"
            )

    # ------------------------------------------------------------------ config
    def _resolve_language_code(
        self, language_codes: str | list[str] | None
    ) -> str | None:
        if language_codes is None:
            mapped = ASSEMBLYAI_LANGUAGE_CODES.get((self.src_lang or "").upper())
            if not mapped:
                warnings.warn(
                    f"AssemblyAI does not map source language {self.src_lang!r}; "
                    "falling back to automatic language detection",
                    stacklevel=2,
                )
                return None
            return mapped
        if isinstance(language_codes, str):
            candidate = language_codes
        else:
            candidate = next(iter(language_codes), "")
        candidate = str(candidate).strip()
        mapped = ASSEMBLYAI_LANGUAGE_CODES.get(candidate.upper(), candidate.lower())
        return mapped or None

    @property
    def _headers(self) -> dict[str, str]:
        return {"authorization": self.api_token or ""}

    # ---------------------------------------------------------------- API calls
    def _upload(self, audio_path: str) -> str:
        with open(audio_path, "rb") as handle:
            resp = requests.post(
                f"{self.base_url}/upload",
                headers=self._headers,
                data=handle,
                timeout=self.upload_timeout,
            )
        resp.raise_for_status()
        upload_url = resp.json().get("upload_url")
        if not upload_url:
            raise RuntimeError(f"AssemblyAI upload returned no upload_url: {resp.text}")
        return upload_url

    def _create_transcript(self, audio_url: str) -> str:
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "speaker_labels": self.speaker_labels,
            "punctuate": True,
        }
        if self.speech_model:
            payload["speech_model"] = self.speech_model
        if self.language_code:
            payload["language_code"] = self.language_code
        else:
            payload["language_detection"] = True
        resp = requests.post(
            f"{self.base_url}/transcript",
            headers=self._headers,
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        transcript_id = resp.json().get("id")
        if not transcript_id:
            raise RuntimeError(f"AssemblyAI transcript create returned no id: {resp.text}")
        return transcript_id

    def _poll(self, transcript_id: str, audio_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + (
            self.poll_timeout if self.poll_timeout else max(600.0, audio_seconds * 3.0)
        )
        url = f"{self.base_url}/transcript/{transcript_id}"
        while True:
            resp = requests.get(url, headers=self._headers, timeout=self.request_timeout)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status")
            if status == "completed":
                return body
            if status == "error":
                raise RuntimeError(
                    f"AssemblyAI transcription failed: {body.get('error')}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"AssemblyAI transcription timed out (id={transcript_id}, "
                    f"last status={status})"
                )
            time.sleep(self.poll_interval)

    # ----------------------------------------------------------- segmentation
    @staticmethod
    def _ms_to_seconds(value: Any) -> float | None:
        try:
            return float(value) / 1000.0
        except (TypeError, ValueError):
            return None

    def _words_to_segments(self, words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group word timestamps into speech segments.

        Split when: the silence gap before a word exceeds ``max_gap_seconds``,
        the speaker changes (when diarization is on), or the running segment
        would exceed ``max_segment_seconds``.
        """
        segments: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for word in words:
            start = self._ms_to_seconds(word.get("start"))
            end = self._ms_to_seconds(word.get("end"))
            if start is None or end is None or end < start:
                continue
            speaker = str(word.get("speaker")) if word.get("speaker") else ""

            if current is None:
                current = {"start": start, "end": end, "speakers": [speaker]}
                continue

            gap = start - current["end"]
            too_long = (end - current["start"]) > self.max_segment_seconds
            speaker_changed = (
                self.speaker_labels
                and speaker
                and current["speakers"][-1]
                and speaker != current["speakers"][-1]
            )
            if gap > self.max_gap_seconds or too_long or speaker_changed:
                segments.append(current)
                current = {"start": start, "end": end, "speakers": [speaker]}
            else:
                current["end"] = end
                current["speakers"].append(speaker)

        if current is not None:
            segments.append(current)
        return segments

    @staticmethod
    def _dominant_speaker(speakers: list[str]) -> str:
        labels = [s for s in speakers if s]
        if not labels:
            return ""
        return Counter(labels).most_common(1)[0][0]

    def _build_srt(self, body: dict[str, Any]) -> SrtScript:
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        words = body.get("words") or []
        if not isinstance(words, list):
            words = []

        for seg in self._words_to_segments(w for w in words if isinstance(w, dict)):
            start_time = float(seg["start"])
            end_time = float(seg["end"])
            if end_time <= start_time:
                continue
            if (end_time - start_time) < self.min_segment_seconds:
                continue
            srt.segments.append(
                SrtSegment(
                    src_lang=self.src_lang,
                    tgt_lang=self.tgt_lang,
                    src_text="",
                    translation="",
                    speaker=self._dominant_speaker(seg["speakers"]),
                    start_time=start_time,
                    end_time=end_time,
                    idx=len(srt.segments),
                )
            )

        self.srt = srt
        return srt

    # ------------------------------------------------------------------- entry
    def get_speaker_segments(
        self, audio_path: str, webhook_url: str | None = None
    ) -> SrtScript:  # noqa: ARG002
        logger.info("Processing audio file with AssemblyAI async VAD: %s", audio_path)
        try:
            import wave

            with wave.open(audio_path, "rb") as wf:
                audio_seconds = wf.getnframes() / float(wf.getframerate() or 16000)
        except Exception:  # pragma: no cover - best-effort duration for poll budget
            audio_seconds = 0.0

        upload_url = self._upload(audio_path)
        transcript_id = self._create_transcript(upload_url)
        body = self._poll(transcript_id, audio_seconds)
        return self._build_srt(body)


__all__ = ["AssemblyAIUniversalVAD"]
