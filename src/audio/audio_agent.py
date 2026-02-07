import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from uuid import uuid4
from openai import OpenAI

from src.audio.ASR import ASR
from src.audio.audio_prompt import (
    AUDIO_ANALYZE_PROMPT,
    AUDIO_TRANSCRIBE_PROMPT,
    AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES,
)
from src.audio.VAD import VAD
from src.audio.vad_agent import create_vad


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
                    self.audio_config.get("min_segment_seconds", 1.0)
                ),
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
            raise ValueError("VAD is not initialized for this audio agent")
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

    @abstractmethod
    def analyze_audio(self, audio_path):
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
                except Exception:
                    results[idx] = []
        return results


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

    def segment_audio(self, audio_path, cache_dir):
        # Prefer VAD segmentation when configured; otherwise create a single placeholder segment
        if self.VAD_model:
            return super().segment_audio(audio_path, cache_dir)

    def analyze_audio(self, audio_path):
        # Not implemented for GPT4o transcription agent
        return None

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
            # duration for end timestamp
            from pydub import AudioSegment

            seg_audio = AudioSegment.from_file(audio_path)
            duration_secs = len(seg_audio) / 1000.0

            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "output_tokens", None) if usage else None
            tt = getattr(usage, "total_tokens", None) if usage else None
            if hasattr(self, "_record_usage"):
                self._record_usage(
                    provider="openai",
                    model=self.model_name,
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
            print("Error occurred while transcribing:", e)
            return []

    # Removed legacy chunking helpers; VAD already splits inputs for the API.

    def _seconds_to_srt_time(self, secs: float) -> str:
        if secs is None:
            secs = 0.0
        if secs < 0:
            secs = 0.0
        total_ms = int(round(float(secs) * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class WhisperAudioAgent(AudioAgent):
    def __init__(self, model_name="whisper-api", audio_config: dict | None = None):
        # Enable VAD when audio_config provides it; otherwise operate without VAD.
        super().__init__(model_name, audio_config=audio_config or {})

    def load_model(self):
        self.ASR_model = ASR.create(self.model_name)

    def segment_audio(self, audio_path, cache_dir):
        """Create speaker segments and clip per-segment audio.

        - If VAD is configured, use it to segment and clip the original audio into
            cache_dir (each segment will have segment.audio_path set).
        """
        return super().segment_audio(audio_path, cache_dir)

    def analyze_audio(self, audio_path):
        pass  # No specific analysis for classic agent, just return empty result

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

    def _srt_time_to_seconds(self, s: str) -> float:
        # HH:MM:SS,mmm
        try:
            # Handle dot or comma for milliseconds
            s = s.replace(".", ",")
            hh, mm, rest = s.split(":")
            ss, ms = rest.split(",")
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
        except Exception:
            return 0.0

    def _seconds_to_srt_time(self, secs: float) -> str:
        if secs is None:
            secs = 0.0
        if secs < 0:
            secs = 0.0
        total_ms = int(round(float(secs) * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

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
            from pydub import AudioSegment

            seg_audio = AudioSegment.from_file(audio_path)
            duration_secs = len(seg_audio) / 1000.0
            self._record_usage(
                provider="openai",
                model="whisper-1",
                category="audio",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                extra={"duration_secs": duration_secs, "agent": "audio"},
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


class QwenAudioAgent(AudioAgent):
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct"):
        super().__init__(model_name)

    def load_model(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, device_map="auto"
        )

    def transcribe(self, audio_path, visual_cues=None):
        import librosa
        import torch

        audios = [
            librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[
                0
            ]
        ]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {
                        "type": "text",
                        "text": AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES.format(
                            visual_cues=visual_cues
                        )
                        if visual_cues
                        else AUDIO_TRANSCRIBE_PROMPT,
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        # Move all tensors to CUDA while preserving the BatchEncoding structure
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=5000)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def analyze_audio(self, audio_path):
        import librosa
        import torch

        audios = [
            librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[
                0
            ]
        ]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": AUDIO_ANALYZE_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=5000)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


class GeminiAudioAgent(AudioAgent):
    def __init__(self, model_name="gemini-3-flash-preview", audio_config: dict = None):
        super().__init__(model_name, audio_config)

    def _seconds_to_srt_time(self, secs: float) -> str:
        """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
        if secs is None:
            secs = 0.0
        if secs < 0:
            secs = 0.0
        total_ms = int(round(float(secs) * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def load_model(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided either directly or via GEMINI_API_KEY environment variable"
            )
        self.model = genai.Client(api_key=api_key)

    def parse_response(self, response, parsing_retries=5):
        for retry in range(parsing_retries):
            try:
                # Check if response is wrapped in code block markers
                if "```json" in response:
                    # Extract the actual JSON content from between the code block markers
                    json_content = (
                        response.split("```json", 1)[1].split("```", 1)[0].strip()
                    )
                    gemini_results = json.loads(json_content)
                    return gemini_results
                else:
                    # If no code block markers, try parsing directly
                    gemini_results = json.loads(response)
                    return gemini_results

            except Exception as e:
                if retry < parsing_retries - 1:
                    print(
                        f"Failed to parse response (attempt {retry + 1}/{parsing_retries}): {str(e)}"
                    )
                    print("Retrying with new response...")
                    continue
                else:
                    print(f"Failed to parse response after {parsing_retries} attempts")
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

        # Standardize separators
        # Replace non-digit chars (except : and .) with :
        # But actually simple replacement of separators to : is safer
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
            # Ambiguous: HH:MM:SS vs MM:SS:mmm
            # We try MM:SS:mmm first as it is more precise for ASR segments usually
            # val_ms_patt: MM:SS:mmm
            val_ms_based = parts[0] * 60 + parts[1] + parts[2] / 1000.0

            # val_hms_patt: HH:MM:SS
            val_hms_based = parts[0] * 3600 + parts[1] * 60 + parts[2]

            # Heuristic: Check against duration
            # If val_hms_based is within duration + margin, and val_ms_based is tiny relative to content?
            # Or if val_hms_based > duration, it must be val_ms_based?

            if val_hms_based > audio_duration_secs * 1.5:
                total_secs = val_ms_based
            elif val_ms_based <= audio_duration_secs:
                # If both are valid, MM:SS:mmm is preferred standard for 3-part ASR tokens if decimals implied
                # But if it looks like integers? 01:20:30.
                # Usually milliseconds are 3 digits.
                # If last part is < 1000, likely ms.
                # But seconds are < 60.
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

    def transcribe(self, audio_path, visual_cues=None):
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "Audio loaded, let me give it a good ol\' chomp... chomp chomp! 🎧"}'
            )

        with open(audio_path, "rb") as audio:
            audio_data = audio.read()

        # Get audio duration for timestamp validation
        audio_duration_secs = 0.0
        try:
            from pydub import AudioSegment

            seg_audio = AudioSegment.from_file(audio_path)
            audio_duration_secs = len(seg_audio) / 1000.0
        except Exception:
            audio_duration_secs = 10.0  # fallback duration

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

        resp = None
        for retry in range(5):
            if resp is None:
                response = self.model.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                # Parse content
                raw_resp = self.parse_response(response.text)

                # Normalize timestamps in the response
                if raw_resp and isinstance(raw_resp, list):
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
                resp = raw_resp

                # Best-effort usage logging via base
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
                        extra=(
                            {"duration_secs": audio_duration_secs, "agent": "audio"}
                            if audio_duration_secs is not None
                            else {"agent": "audio"}
                        ),
                    )
                except Exception:
                    pass
            else:
                if self.agent_history_logger:
                    self.agent_history_logger.info(
                        '{"role": "audio_agent", "message": "Transcription done! If I missed a beat, blame the waveform, not me 😜"}'
                    )
                return resp

        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "Oops, 5 tries and still no luck. Even the best of us have off days!"}'
            )
        # Fallback plain log
        try:
            print("[GeminiAudioAgent] Failed to transcribe audio after 5 retries.")
        except Exception:
            pass
        return resp

    def analyze_audio(self, audio_path):
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "Analyzing audio... let me put on my sonic goggles! 🥽"}'
            )

        with open(audio_path, "rb") as audio:
            audio_data = audio.read()

        from google.genai import types

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=AUDIO_ANALYZE_PROMPT),
                    types.Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                ],
            )
        ]

        response = self.model.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        result = self.parse_response(response.text)
        if self.agent_history_logger:
            self.agent_history_logger.info(
                '{"role": "audio_agent", "message": "Audio analysis complete! If I missed a note, it was jazz."}'
            )
        return result


if __name__ == "__main__":
    agent = QwenAudioAgent()
    visual_cues = "a man named lowko is talking about the game of starcraft"
    print(agent.transcribe("C:\\Work\\GitRepos\\task.mp3"))
    print(agent.transcribe("C:\\Work\\GitRepos\\task.mp3", visual_cues))
    print(agent.analyze_audio("C:\\Work\\GitRepos\\task.mp3"))
