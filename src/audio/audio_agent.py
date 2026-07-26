import json
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import sleep
from uuid import uuid4
from openai import OpenAI

from src.audio.ASR import ASR
from src.audio.audio_prompt import (
    AUDIO_TRANSCRIBE_PROMPT,
    AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES,
)
from src.audio.VAD import VAD
from src.audio.vad_agent import create_vad

logger = logging.getLogger(__name__)


def _audio_duration_secs(audio_path, default: float = 10.0) -> float:
    """Best-effort clip duration in seconds."""
    try:
        from pydub import AudioSegment

        return len(AudioSegment.from_file(audio_path)) / 1000.0
    except Exception:
        return default


class AudioAgent(ABC):
    def __init__(self, model_name, audio_config: dict = None):
        self.model_name = model_name
        self.audio_config = audio_config or {}
        self.device = None
        # usage recording context
        self.task_id = (self.audio_config or {}).get("task_id")
        self.usage_log_path = (self.audio_config or {}).get("usage_log_path")
        # agent conversation logger
        self.agent_history_logger = None
        self.load_model()
        # Initialize VAD only if configured
        self.VAD_model = None
        if all(k in self.audio_config for k in ("VAD_model", "src_lang", "tgt_lang")):
            provider_options = self.audio_config.get("VAD_options") or {}
            if not isinstance(provider_options, dict):
                provider_options = {}
            self.VAD_model = create_vad(
                model_name_or_path=self.audio_config["VAD_model"],
                src_lang=self.audio_config["src_lang"],
                tgt_lang=self.audio_config["tgt_lang"],
                min_segment_seconds=float(
                    self.audio_config.get("min_segment_seconds", 0.8)
                ),
                provider=self.audio_config.get("VAD_provider"),
                **provider_options,
            )

    def set_usage_log_path(self, path: str | None):
        self.usage_log_path = path

    def set_task_id(self, task_id: str | None):
        self.task_id = task_id

    def set_agent_history_logger(self, logger):
        """Set the agent history logger for conversation logging"""
        self.agent_history_logger = logger

    def _record_usage(
        self,
        *,
        provider: str,
        model: str,
        category: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        cached_prompt_tokens: int | None = None,
        phrase_index: int | None = None,
        extra: dict | None = None,
    ) -> None:
        """Best-effort JSONL recorder for per-request usage."""
        if not self.usage_log_path:
            return
        try:
            rec = {
                "request_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "task_id": self.task_id,
                "provider": provider,
                "model": model,
                "category": category,
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "phrase_index": phrase_index,
            }
            if extra:
                rec.update({"extra": extra})
            with open(self.usage_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def segment_audio(self, audio_path, cache_dir):
        if not self.VAD_model:
            raise ValueError(
                "VAD is not initialized for this audio agent. "
                "Configure audio.VAD_model (and src_lang/tgt_lang) in the task config."
            )
        self.segments = self.VAD_model.get_speaker_segments(audio_path)
        VAD.clip_audio_and_save(self.segments, audio_path, cache_dir)
        return self.segments

    def clip_video_and_save(self, video_path, cache_dir):
        # Only attempt clip when VAD exists and produced real segments
        if not self.VAD_model:
            return
        try:
            VAD.clip_video_and_save(self.segments, video_path, cache_dir)
        except Exception:
            # Be resilient in absence of usable segments
            return

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def transcribe(self, audio_path, visual_cues=None):
        pass

    def transcribe_batch(
        self, items: list[dict], max_workers: int = 4
    ) -> dict[int, list[dict]]:
        """
        Concurrent transcription helper.
        items: list of {idx: int, audio_path: str, visual_cues?: str}
        Returns: { idx: [ {start: str, end: str, text: str}, ... ] }
        """
        results: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.transcribe, it["audio_path"], it.get("visual_cues")
                ): it["idx"]
                for it in items
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result() or []
                except Exception as e:
                    logger.error(
                        "Transcription failed for segment %s: %s; leaving it empty.",
                        idx,
                        e,
                    )
                    results[idx] = []
        return results

    @staticmethod
    def _seconds_to_srt_time(secs: float) -> str:
        """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
        if secs is None or secs < 0:
            secs = 0.0
        total_ms = int(round(float(secs) * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class GPT4oAudioAgent(AudioAgent):
    def __init__(
        self, model_name="gpt-4o-mini-transcribe", audio_config: dict | None = None
    ):
        super().__init__(model_name, audio_config)

    def load_model(self):
        # Normalize model name to a valid audio transcription-capable model
        normalized = self.model_name
        # Known audio-capable identifiers as of current SDKs
        if normalized in ("gpt-4o", "gpt-4o-mini"):
            normalized = normalized + "-transcribe"
        self.model_name = normalized

        self.client = OpenAI()

    def transcribe(self, audio_path, visual_cues=None):
        """
        Transcribe a single (VAD-split) audio chunk via OpenAI gpt-4o(-mini)-transcribe.
        API returns plain text. We wrap it in one SRT-timed segment covering the clip.
        """
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "🎧 GPT-4o Audio is processing... hang tight!"}'
            )

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    prompt=AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES.format(
                        visual_cues=visual_cues
                    )
                    if visual_cues
                    else AUDIO_TRANSCRIBE_PROMPT,
                    model=self.model_name,
                    file=audio_file,
                    response_format="json",
                )
            text = None
            if hasattr(response, "text"):
                text = response.text
            elif isinstance(response, dict):
                text = response.get("text")
            else:
                text = str(response)
            duration_secs = _audio_duration_secs(audio_path)

            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "output_tokens", None) if usage else None
            tt = getattr(usage, "total_tokens", None) if usage else None
            self._record_usage(
                provider="openai",
                model=self.model_name,
                category="audio",
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                phrase_index=None,
                extra={"duration_secs": duration_secs, "agent": "audio"},
            )

            return [
                {
                    "start": self._seconds_to_srt_time(0.0),
                    "end": self._seconds_to_srt_time(duration_secs),
                    "text": text or "",
                }
            ]
        except Exception as e:
            logger.error("Error occurred while transcribing %s: %s", audio_path, e)
            return []


class WhisperAudioAgent(AudioAgent):
    def __init__(self, model_name="whisper-api", audio_config: dict | None = None):
        # Enable VAD when audio_config provides it; otherwise operate without VAD.
        super().__init__(model_name, audio_config=audio_config or {})

    def load_model(self):
        self.ASR_model = ASR.create(self.model_name)

    def _parse_srt_to_segments(self, srt_text: str):
        # Convert SRT string to list of dicts with 'start','end','text' (time strings HH:MM:SS,mmm)
        segments = []
        if not srt_text:
            return segments
        lines = [ln.strip("\ufeff").strip() for ln in srt_text.splitlines()]
        i = 0
        while i < len(lines):
            # skip index line if numeric
            if lines[i].isdigit():
                i += 1
            if i >= len(lines):
                break
            # time line
            if "-->" in lines[i]:
                time_line = lines[i]
                i += 1
                text_lines = []
                while i < len(lines) and lines[i] != "":
                    text_lines.append(lines[i])
                    i += 1
                # skip blank
                while i < len(lines) and lines[i] == "":
                    i += 1
                try:
                    start_str, end_str = [t.strip() for t in time_line.split("-->")]
                    text = " ".join(text_lines).strip()
                    if text:
                        segments.append(
                            {"start": start_str, "end": end_str, "text": text}
                        )
                except Exception:
                    # ignore malformed entries
                    pass
            else:
                i += 1
        return segments

    def transcribe(self, audio_path, visual_cues=None):
        # Transcribe the given (per-VAD) audio clip with Whisper ASR.
        # Return SRT-like segments with HH:MM:SS,mmm timestamps relative to this clip.
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "🎤 Whisper is listening... processing audio clip"}'
            )

        src_lang = (self.audio_config or {}).get("src_lang")
        result = self.ASR_model.get_transcript(audio_path, source_lang=src_lang)
        # If this path uses OpenAI Whisper API under the hood, record per-minute usage
        try:
            self._record_usage(
                provider="openai",
                model="whisper-large-v3",
                category="audio",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                extra={
                    "duration_secs": _audio_duration_secs(audio_path, default=0.0),
                    "agent": "audio",
                },
            )
        except Exception:
            pass
        # If Whisper API returns SRT text, keep as SRT time strings
        if isinstance(result, str):
            return self._parse_srt_to_segments(result)
        # If list of dicts with numeric times, convert to SRT strings
        norm = []
        for seg in result or []:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")
            # Some implementations may already return HH:MM:SS,mmm strings
            if isinstance(start, str) and "-->" not in start:
                start_srt = start
            else:
                start_srt = self._seconds_to_srt_time(
                    start if not isinstance(start, str) else 0.0
                )
            if isinstance(end, str) and "-->" not in end:
                end_srt = end
            else:
                end_srt = self._seconds_to_srt_time(
                    end if not isinstance(end, str) else 0.0
                )
            norm.append({"start": start_srt, "end": end_srt, "text": text})
        return norm


class Qwen3ASRAudioAgent(AudioAgent):
    def __init__(
        self, model_name="qwen3-asr-flash", audio_config: dict | None = None
    ):
        super().__init__(model_name, audio_config=audio_config or {})

    def load_model(self):
        cfg = self.audio_config or {}
        asr_options = cfg.get("asr_options") if isinstance(cfg.get("asr_options"), dict) else {}
        api_key = cfg.get("dashscope_api_key") or cfg.get("DASHSCOPE_API_KEY")
        system_prompt = cfg.get("qwen_asr_system_prompt", "")
        self.ASR_model = ASR.create(
            "qwen3-asr-flash",
            model_name=self.model_name,
            api_key=api_key,
            asr_options=asr_options,
            system_prompt=system_prompt,
        )

    def _build_init_prompt(self, visual_cues=None):
        base_prompt = (self.audio_config or {}).get("qwen_asr_system_prompt", "").strip()
        if visual_cues:
            visual_prompt = f"Visual context for disambiguation: {visual_cues}"
            return f"{base_prompt}\n{visual_prompt}".strip() if base_prompt else visual_prompt
        return base_prompt or None

    def _normalize_segments(self, transcript, duration_secs):
        if isinstance(transcript, str):
            return [
                {
                    "start": self._seconds_to_srt_time(0.0),
                    "end": self._seconds_to_srt_time(duration_secs),
                    "text": transcript.strip(),
                }
            ]

        normalized = []
        for seg in transcript or []:
            start = seg.get("start", 0.0)
            end = seg.get("end", duration_secs)
            text = (seg.get("text", "") or "").strip()
            if isinstance(start, str):
                start_srt = start
            else:
                start_srt = self._seconds_to_srt_time(start)
            if isinstance(end, str):
                end_srt = end
            else:
                end_srt = self._seconds_to_srt_time(end)
            if text:
                normalized.append({"start": start_srt, "end": end_srt, "text": text})

        return normalized

    def transcribe(self, audio_path, visual_cues=None):
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "🎙️ Qwen3 ASR is transcribing this audio clip..."}'
            )

        src_lang = (self.audio_config or {}).get("src_lang")
        init_prompt = self._build_init_prompt(visual_cues=visual_cues)
        transcript = self.ASR_model.get_transcript(
            audio_path, source_lang=src_lang, init_prompt=init_prompt
        )
        if not transcript:
            return []

        duration_secs = _audio_duration_secs(audio_path)

        usage = getattr(self.ASR_model, "last_usage", {}) or {}
        self._record_usage(
            provider="dashscope",
            model=self.model_name,
            category="audio",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            extra={"duration_secs": duration_secs, "agent": "audio"},
        )

        return self._normalize_segments(transcript, duration_secs)


class GeminiAudioAgent(AudioAgent):
    def __init__(self, model_name="gemini-3-flash-preview", audio_config: dict = None):
        super().__init__(model_name, audio_config)

    def load_model(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided either directly or via GEMINI_API_KEY environment variable"
            )
        self.model = genai.Client(api_key=api_key)

    @staticmethod
    def parse_response(response):
        """Parse a JSON payload from the model output; returns None on failure."""
        if not response:
            return None
        try:
            if "```json" in response:
                # Extract the actual JSON content from between the code block markers
                json_content = (
                    response.split("```json", 1)[1].split("```", 1)[0].strip()
                )
                return json.loads(json_content)
            return json.loads(response)
        except Exception as e:
            logger.warning("Failed to parse Gemini response as JSON: %s", e)
            return None

    def _normalize_timestamp(self, time_str, audio_duration_secs):
        if not time_str:
            return "00:00:00,000"

        time_str = str(time_str).strip()

        # Case 1: Plain decimal seconds
        try:
            if ":" not in time_str and "->" not in time_str:
                secs = float(time_str)
                secs = max(0.0, min(secs, audio_duration_secs))
                return self._seconds_to_srt_time(secs)
        except ValueError:
            pass

        # Standardize separators to ':' before splitting
        normalized = time_str.replace(",", ":").replace(".", ":")

        # Filter out empty strings and non-digits
        parts = []
        for p in normalized.split(":"):
            if p.isdigit():
                parts.append(int(p))

        if not parts:
            return "00:00:00,000"

        total_secs = 0.0

        if len(parts) == 4:
            # HH:MM:SS:mmm
            hh, mm, ss, ms = parts
            total_secs = hh * 3600 + mm * 60 + ss + ms / 1000.0

        elif len(parts) == 3:
            # Ambiguous: HH:MM:SS vs MM:SS:mmm.
            # Prefer MM:SS:mmm unless HH:MM:SS fits the clip duration better.
            val_ms_based = parts[0] * 60 + parts[1] + parts[2] / 1000.0
            val_hms_based = parts[0] * 3600 + parts[1] * 60 + parts[2]

            if val_hms_based > audio_duration_secs * 1.5:
                total_secs = val_ms_based
            elif val_ms_based <= audio_duration_secs:
                total_secs = val_ms_based
            else:
                total_secs = val_hms_based

        elif len(parts) == 2:
            # MM:SS
            mm, ss = parts
            total_secs = mm * 60 + ss

        elif len(parts) == 1:
            # Just seconds?
            total_secs = float(parts[0])

        # Clamp
        total_secs = max(0.0, min(total_secs, audio_duration_secs))
        return self._seconds_to_srt_time(total_secs)

    def transcribe(self, audio_path, visual_cues=None, max_attempts=5):
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "Audio loaded, let me give it a good ol\' chomp... chomp chomp! 🎧"}'
            )

        with open(audio_path, "rb") as audio:
            audio_data = audio.read()

        # Get audio duration for timestamp validation
        audio_duration_secs = _audio_duration_secs(audio_path)

        from google.genai import types

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES.format(
                            visual_cues=visual_cues
                        )
                        if visual_cues
                        else AUDIO_TRANSCRIBE_PROMPT
                    ),
                    types.Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                ],
            )
        ]

        for attempt in range(max_attempts):
            try:
                response = self.model.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
            except Exception as e:
                logger.warning(
                    "Gemini transcription request failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    e,
                )
                sleep(min(10.0, 1.0 + attempt * 2))
                continue

            # Best-effort usage logging
            try:
                meta = getattr(response, "usage_metadata", None)
                pt = getattr(meta, "prompt_token_count", None) if meta else None
                ct = getattr(meta, "candidates_token_count", None) if meta else None
                tt = getattr(meta, "total_token_count", None) if meta else None
                self._record_usage(
                    provider="gemini",
                    model=self.model_name,
                    category="audio",
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    extra={"duration_secs": audio_duration_secs, "agent": "audio"},
                )
            except Exception:
                pass

            raw_resp = self.parse_response(getattr(response, "text", None))
            if isinstance(raw_resp, dict):
                # Some responses wrap the array, e.g. {"segments": [...]}
                raw_resp = raw_resp.get("segments")
            if isinstance(raw_resp, list):
                for seg in raw_resp:
                    if isinstance(seg, dict):
                        if "start" in seg:
                            seg["start"] = self._normalize_timestamp(
                                seg["start"], audio_duration_secs
                            )
                        if "end" in seg:
                            seg["end"] = self._normalize_timestamp(
                                seg["end"], audio_duration_secs
                            )
                if self.agent_history_logger:
                    self.agent_history_logger.info(
                        '{"role": "audio_agent", "message": "Transcription done! If I missed a beat, blame the waveform, not me 😜"}'
                    )
                return raw_resp

            logger.warning(
                "Gemini transcription returned unparseable output (attempt %d/%d); retrying.",
                attempt + 1,
                max_attempts,
            )

        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "Oops, no luck after retries. Even the best of us have off days!"}'
            )
        logger.error(
            "[GeminiAudioAgent] Failed to transcribe %s after %d attempts.",
            audio_path,
            max_attempts,
        )
        return []
