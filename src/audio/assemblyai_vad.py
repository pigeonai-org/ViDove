from __future__ import annotations

import json
import os
import time
import warnings
from collections import Counter
from logging import getLogger
from typing import Any, Iterable
from urllib.parse import urlencode

from pydub import AudioSegment

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
    """VAD provider backed by AssemblyAI's v3 streaming endpoint."""

    endpoint = "wss://streaming.assemblyai.com/v3/ws"

    def __init__(
        self,
        model_name_or_path: str,
        src_lang: str,
        tgt_lang: str,
        min_segment_seconds: float = 0.8,
        *,
        api_token: str | None = None,
        mode: str = "balanced",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        include_partial_turns: bool = False,
        speaker_labels: bool = True,
        chunk_ms: int = 50,
        max_audio_seconds: float = 3 * 60 * 60,
        connect_timeout: float = 30.0,
        receive_timeout: float = 60.0,
        drain_timeout: float = 0.001,
        drain_every_chunks: int = 20,
        realtime: bool = False,
        language_codes: str | list[str] | None = None,
        **streaming_options: Any,
    ) -> None:
        super().__init__(src_lang, tgt_lang, min_segment_seconds)
        self.model_name_or_path = model_name_or_path or "universal-3-5-pro"
        self.api_token = api_token or os.getenv("ASSEMBLYAI_API_KEY")
        self.mode = mode
        self.encoding = encoding
        self.sample_rate = int(sample_rate)
        self.include_partial_turns = include_partial_turns
        self.speaker_labels = speaker_labels
        self.chunk_ms = int(chunk_ms)
        self.max_audio_seconds = float(max_audio_seconds)
        self.connect_timeout = float(connect_timeout)
        self.receive_timeout = float(receive_timeout)
        self.drain_timeout = float(drain_timeout)
        self.drain_every_chunks = max(1, int(drain_every_chunks))
        self.realtime = bool(realtime)
        self.streaming_options = dict(streaming_options)
        self.language_codes = self._normalize_language_codes(language_codes)

        if not self.api_token:
            raise ValueError(
                "Set ASSEMBLYAI_API_KEY or pass api_token to use AssemblyAI VAD"
            )

    @staticmethod
    def _normalize_audio(audio: AudioSegment, sample_rate: int = 16000) -> AudioSegment:
        return audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)

    @classmethod
    def iter_audio_chunks(
        cls,
        audio: AudioSegment,
        *,
        chunk_ms: int = 50,
        sample_rate: int = 16000,
    ) -> Iterable[bytes]:
        normalized = cls._normalize_audio(audio, sample_rate)
        for start_ms in range(0, len(normalized), chunk_ms):
            chunk = normalized[start_ms : start_ms + chunk_ms]
            if len(chunk) > 0:
                yield bytes(chunk.raw_data)

    def _normalize_language_codes(
        self, language_codes: str | list[str] | None
    ) -> list[str]:
        raw_codes: list[str]
        if language_codes is None:
            mapped = ASSEMBLYAI_LANGUAGE_CODES.get((self.src_lang or "").upper())
            if not mapped:
                warnings.warn(
                    f"AssemblyAI language steering does not support {self.src_lang!r}; omitting language_codes",
                    stacklevel=2,
                )
                return []
            return [mapped]

        if isinstance(language_codes, str):
            raw_codes = [language_codes]
        else:
            raw_codes = list(language_codes)

        supported = set(ASSEMBLYAI_LANGUAGE_CODES.values())
        normalized = []
        for code in raw_codes:
            candidate = str(code).strip().lower()
            mapped = ASSEMBLYAI_LANGUAGE_CODES.get(candidate.upper(), candidate)
            if mapped in supported:
                normalized.append(mapped)
            else:
                warnings.warn(
                    f"AssemblyAI language steering does not support {code!r}; omitting it",
                    stacklevel=2,
                )
        return normalized

    @staticmethod
    def _query_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    def _websocket_url(self) -> str:
        params: dict[str, Any] = {
            "speech_model": self.model_name_or_path,
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "mode": self.mode,
            "include_partial_turns": self.include_partial_turns,
            "speaker_labels": self.speaker_labels,
        }
        if self.language_codes:
            params["language_codes"] = self.language_codes
        params.update(self.streaming_options)

        encoded = urlencode({key: self._query_value(value) for key, value in params.items()})
        separator = "&" if "?" in self.endpoint else "?"
        return f"{self.endpoint}{separator}{encoded}"

    @staticmethod
    def _message_to_json(message: str | bytes) -> dict[str, Any] | None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON AssemblyAI message: %s", message)
            return None
        if isinstance(data, dict):
            return data
        return None

    @staticmethod
    def _is_termination_message(data: dict[str, Any]) -> bool:
        message_type = data.get("type")
        return message_type == "Termination" or (
            "audio_duration_seconds" in data and "session_duration_seconds" in data
        )

    def _receive_available(
        self,
        ws: Any,
        websocket_module: Any,
        messages: list[dict[str, Any]],
        *,
        timeout: float,
    ) -> bool:
        terminated = False
        ws.settimeout(timeout)
        while True:
            try:
                raw_message = ws.recv()
            except websocket_module.WebSocketTimeoutException:
                break
            except websocket_module.WebSocketConnectionClosedException:
                break

            data = self._message_to_json(raw_message)
            if not data:
                continue
            if data.get("type") == "Error" or data.get("error"):
                raise RuntimeError(f"AssemblyAI streaming error: {data}")
            messages.append(data)
            if self._is_termination_message(data):
                terminated = True
                break
        return terminated

    def _stream_messages(self, audio_path: str) -> list[dict[str, Any]]:
        try:
            import websocket  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Install websocket-client to use AssemblyAI VAD"
            ) from exc

        audio = self._normalize_audio(AudioSegment.from_file(audio_path), self.sample_rate)
        duration_seconds = len(audio) / 1000.0
        if duration_seconds > self.max_audio_seconds:
            raise ValueError(
                "AssemblyAI streaming sessions are limited to roughly 3 hours; "
                f"input is {duration_seconds:.1f} seconds"
            )

        messages: list[dict[str, Any]] = []
        ws = websocket.create_connection(
            self._websocket_url(),
            header=[f"Authorization: {self.api_token}"],
            timeout=self.connect_timeout,
        )
        try:
            self._receive_available(
                ws, websocket, messages, timeout=self.drain_timeout
            )

            bytes_per_second = self.sample_rate * 2
            for idx, chunk in enumerate(
                self.iter_audio_chunks(
                    audio, chunk_ms=self.chunk_ms, sample_rate=self.sample_rate
                )
            ):
                ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                if self.realtime:
                    time.sleep(len(chunk) / bytes_per_second)
                if idx % self.drain_every_chunks == 0:
                    self._receive_available(
                        ws, websocket, messages, timeout=self.drain_timeout
                    )

            ws.send(json.dumps({"type": "Terminate"}))

            deadline = time.monotonic() + self.receive_timeout
            while time.monotonic() < deadline:
                if self._receive_available(
                    ws, websocket, messages, timeout=min(1.0, self.receive_timeout)
                ):
                    break
            else:
                raise TimeoutError("Timed out waiting for AssemblyAI termination")
        finally:
            ws.close()

        return messages

    @staticmethod
    def _ms_to_seconds(value: Any) -> float | None:
        try:
            return float(value) / 1000.0
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _speaker_from_words(words: list[dict[str, Any]]) -> str:
        speakers = [str(word.get("speaker")) for word in words if word.get("speaker")]
        if not speakers:
            return ""
        return Counter(speakers).most_common(1)[0][0]

    def _speaker_from_turn(self, turn: dict[str, Any]) -> str:
        speaker_label = turn.get("speaker_label")
        if speaker_label:
            return str(speaker_label)
        words = turn.get("words") or []
        if isinstance(words, list):
            return self._speaker_from_words(
                [word for word in words if isinstance(word, dict)]
            )
        return ""

    def messages_to_srt(self, messages: Iterable[dict[str, Any]]) -> SrtScript:
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        pending_speech_start: float | None = None
        turn_to_segment: dict[int, SrtSegment] = {}
        speaker_revisions: list[dict[str, Any]] = []

        for data in messages:
            message_type = data.get("type")
            if message_type == "SpeechStarted":
                pending_speech_start = self._ms_to_seconds(data.get("timestamp"))
                continue

            if message_type == "SpeakerRevision":
                revisions = data.get("revisions") or []
                if isinstance(revisions, list):
                    speaker_revisions.extend(
                        revision for revision in revisions if isinstance(revision, dict)
                    )
                continue

            if message_type != "Turn" or not data.get("end_of_turn"):
                continue

            words = data.get("words") or []
            if not isinstance(words, list):
                words = []
            word_dicts = [word for word in words if isinstance(word, dict)]
            final_words = [
                word for word in word_dicts if word.get("word_is_final", True) is not False
            ]
            timing_words = final_words or word_dicts
            if not timing_words:
                pending_speech_start = None
                continue

            first_word_start = self._ms_to_seconds(timing_words[0].get("start"))
            last_word_end = self._ms_to_seconds(timing_words[-1].get("end"))
            if last_word_end is None:
                pending_speech_start = None
                continue

            start_time = pending_speech_start
            if start_time is None:
                start_time = first_word_start
            pending_speech_start = None
            if start_time is None or last_word_end <= start_time:
                continue
            if (last_word_end - start_time) < self.min_segment_seconds:
                continue

            try:
                turn_order = int(data.get("turn_order", len(srt.segments)))
            except (TypeError, ValueError):
                turn_order = len(srt.segments)

            segment = SrtSegment(
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                src_text="",
                translation="",
                speaker=self._speaker_from_turn(data),
                start_time=float(start_time),
                end_time=float(last_word_end),
                idx=len(srt.segments),
            )
            srt.segments.append(segment)
            turn_to_segment[turn_order] = segment

        for revision in speaker_revisions:
            try:
                turn_order = int(revision.get("turn_order"))
            except (TypeError, ValueError):
                continue
            segment = turn_to_segment.get(turn_order)
            if not segment:
                continue
            speaker = revision.get("speaker_label")
            if not speaker:
                words = revision.get("words") or []
                if isinstance(words, list):
                    speaker = self._speaker_from_words(
                        [word for word in words if isinstance(word, dict)]
                    )
            if speaker:
                segment.speaker = str(speaker)

        self.srt = srt
        return srt

    def get_speaker_segments(
        self, audio_path: str, webhook_url: str | None = None
    ) -> SrtScript:  # noqa: ARG002
        logger.info("Processing audio file with AssemblyAI Universal VAD: %s", audio_path)
        return self.messages_to_srt(self._stream_messages(audio_path))


__all__ = ["AssemblyAIUniversalVAD"]
