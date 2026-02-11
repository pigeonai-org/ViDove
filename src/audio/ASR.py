from abc import ABC, abstractmethod
import math
import os
from pathlib import Path
import re
import tempfile
import torch
import traceback
import librosa
from pydub import AudioSegment
from openai import OpenAI

# Optional imports - loaded lazily when needed
stable_whisper = None
whisper = None
WhisperForConditionalGeneration = None
WhisperProcessor = None
dashscope = None


class ASR(ABC):
    """Abstract base class for all ASR implementations"""

    def __init__(self, device=None, logger=None):
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logger

    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def load_audio(self, audio_path):
        # Load and resample the audio using librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        # Convert to float32 tensor
        audio = torch.from_numpy(audio).float()
        return audio

    @abstractmethod
    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        """
        Get transcript from audio file

        Args:
            audio_path: Path to audio file
            source_lang: Source language code
            init_prompt: Initial prompt for the ASR model

        Returns:
            Transcript in the format specified by the implementation
        """
        pass

    @staticmethod
    def create(method, **kwargs):
        """Factory method to create appropriate ASR instance"""
        raw_method = method or ""
        method_name = (
            raw_method.lower() if isinstance(raw_method, str) else str(raw_method).lower()
        )

        if method_name == "whisper-api":
            return WhisperAPIASR(**kwargs)
        elif method_name in {"qwen-asr-flash", "qwen3-asr-flash", "qwen-asr"}:
            model_name = kwargs.pop("model_name", "qwen3-asr-flash")
            return Qwen3ASRFlashASR(model_name=model_name, **kwargs)
        elif "stable" in method_name:
            whisper_model = raw_method.split("-")[2]
            return StableWhisperASR(whisper_model=whisper_model, **kwargs)
        elif "oai" in method_name:
            model_id = raw_method.split("-")[2] if "-" in raw_method else "large-v3"
            return OAIWhisperASR(model_id=model_id, **kwargs)
        else:
            return HuggingfaceWhisperASR(model_id=raw_method, **kwargs)


class WhisperAPIASR(ASR):
    """Implementation of ASR using OpenAI's Whisper API"""

    def __init__(self, client=None, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        if self.client is None:
            self.client = OpenAI()

    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        """Transcribe audio, splitting into chunks when over the API size limit.

        Returns a single SRT string stitched from chunk results.
        """
        try:
            max_bytes = 24 * 1024 * 1024  # keep a 1MB+ safety margin under 25MB
            file_size = os.path.getsize(audio_path)

            if file_size <= max_bytes:
                return self._transcribe_file(audio_path, source_lang, init_prompt)

            # Oversized: split into chunks and stitch
            self.log(
                f"Audio size {file_size} bytes exceeds limit; splitting into chunks…"
            )
            return self._transcribe_in_chunks(
                audio_path, source_lang, init_prompt, max_bytes=max_bytes
            )

        except Exception as e:
            self.log(f"WhisperAPIASR error: {e}")
            traceback.print_exc()
            return None

    # --- helpers for chunked transcription ---
    def _transcribe_file(self, file_path, source_lang=None, init_prompt=None):
        with open(file_path, "rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt",
                language=source_lang.lower() if source_lang else None,
                prompt=init_prompt or "",
            )
        # The SDK returns a string when response_format="srt"
        return result if isinstance(result, str) else str(result)

    def _transcribe_in_chunks(self, audio_path, source_lang, init_prompt, max_bytes):
        # Decide number of chunks by size, then slice by duration
        total_size = os.path.getsize(audio_path)
        num_chunks = max(2, math.ceil(total_size / max_bytes))

        audio = AudioSegment.from_file(audio_path)
        # Use consistent encoding settings to keep chunk sizes small
        # We'll export chunks as 128kbps mp3 to stay well under limits
        total_ms = len(audio)
        chunk_ms = math.ceil(total_ms / num_chunks)

        all_entries = []
        offset_seconds = 0.0

        with tempfile.TemporaryDirectory(prefix="vidove_asr_") as tmpdir:
            for i, start in enumerate(range(0, total_ms, chunk_ms)):
                end = min(start + chunk_ms, total_ms)
                seg = audio[start:end]
                seg_path = os.path.join(tmpdir, f"chunk_{i:03d}.mp3")

                # Normalize to mono 16kHz for Whisper-friendly input and predictable duration
                seg = seg.set_channels(1).set_frame_rate(16000)
                seg.export(seg_path, format="mp3", bitrate="128k")

                # Transcribe this chunk
                srt_part = self._transcribe_file(seg_path, source_lang, init_prompt)
                if not srt_part:
                    continue
                # Parse and offset timestamps
                entries = self._parse_srt(srt_part)
                for e in entries:
                    e["start"] += offset_seconds
                    e["end"] += offset_seconds
                all_entries.extend(entries)

                # Advance offset by actual chunk duration
                offset_seconds += len(seg) / 1000.0

        # Reformat as one SRT
        return self._format_srt(all_entries)

    # --- SRT utilities ---
    def _srt_time_to_seconds(self, s: str) -> float:
        # HH:MM:SS,mmm
        s = s.replace(".", ",")
        hh, mm, rest = s.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

    def _seconds_to_srt_time(self, secs: float) -> str:
        if secs < 0:
            secs = 0.0
        total_ms = int(round(secs * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _parse_srt(self, srt_str: str):
        entries = []
        if not srt_str:
            return entries
        blocks = re.split(r"\n\s*\n", srt_str.strip(), flags=re.MULTILINE)
        idx = 1
        time_re = re.compile(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
        )
        for blk in blocks:
            lines = [ln for ln in blk.splitlines() if ln.strip()]
            if not lines:
                continue
            # Allow optional numeric index line
            if "-->" in lines[0]:
                time_line = lines[0]
                text_lines = lines[1:]
            elif len(lines) >= 2 and "-->" in lines[1]:
                time_line = lines[1]
                text_lines = lines[2:]
            else:
                continue
            m = time_re.search(time_line)
            if not m:
                continue
            start = self._srt_time_to_seconds(m.group(1))
            end = self._srt_time_to_seconds(m.group(2))
            text = "\n".join(text_lines).strip()
            entries.append({"index": idx, "start": start, "end": end, "text": text})
            idx += 1
        return entries

    def _format_srt(self, entries):
        lines = []
        for i, e in enumerate(entries, start=1):
            start_tc = (
                self._seconds_to_srt_time(e["start"])
                if isinstance(e["start"], (int, float))
                else e["start"]
            )
            end_tc = (
                self._seconds_to_srt_time(e["end"])
                if isinstance(e["end"], (int, float))
                else e["end"]
            )
            lines.append(str(i))
            lines.append(f"{start_tc} --> {end_tc}")
            lines.append(e.get("text", ""))
            lines.append("")
        return "\n".join(lines).strip() + "\n"


class Qwen3ASRFlashASR(ASR):
    """Implementation of ASR using DashScope qwen3-asr-flash."""

    def __init__(
        self,
        model_name="qwen3-asr-flash",
        api_key=None,
        asr_options=None,
        system_prompt="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        global dashscope
        if dashscope is None:
            try:
                import dashscope as _dashscope

                dashscope = _dashscope
            except ImportError:
                raise ImportError("Please install dashscope: pip install dashscope")

        self.model_name = model_name or "qwen3-asr-flash"
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.system_prompt = system_prompt or ""
        self.asr_options = {"enable_lid": True, "enable_itn": False}
        if isinstance(asr_options, dict):
            self.asr_options.update(asr_options)
        self.last_usage = None

    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        try:
            if not self.api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY is required for qwen3-asr-flash transcription."
                )

            messages = [
                {
                    "role": "system",
                    "content": [{"text": (init_prompt or self.system_prompt or "").strip()}],
                },
                {
                    "role": "user",
                    "content": [{"audio": self._to_audio_uri(audio_path)}],
                },
            ]

            opts = dict(self.asr_options)
            if source_lang:
                opts.setdefault("language", str(source_lang).lower())

            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=messages,
                result_format="message",
                asr_options=opts,
            )

            self.last_usage = self._extract_usage(response)
            return self._extract_text(response)
        except Exception as e:
            self.log(f"Qwen3ASRFlashASR error: {e}")
            traceback.print_exc()
            self.last_usage = None
            return None

    def _to_audio_uri(self, audio_path: str) -> str:
        if re.match(r"^https?://", str(audio_path), flags=re.IGNORECASE):
            return str(audio_path)
        local_file = Path(audio_path).expanduser()
        if not local_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return local_file.resolve().as_uri()

    def _extract_text(self, response) -> str | None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) >= 400:
            err = getattr(response, "message", None) or getattr(response, "code", None)
            raise RuntimeError(f"DashScope request failed ({status_code}): {err}")

        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")

        text_items = []

        choices = None
        if output is not None:
            choices = getattr(output, "choices", None)
            if choices is None and isinstance(output, dict):
                choices = output.get("choices")

        for choice in choices or []:
            if isinstance(choice, dict):
                message = choice.get("message")
            else:
                message = getattr(choice, "message", None)
            if not message:
                continue

            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)

            text_items.extend(self._extract_text_items(content))

        if not text_items:
            if output is not None:
                text_items.extend(self._extract_text_items(output))
            text_items.extend(self._extract_text_items(response))

        merged_text = "\n".join(t for t in text_items if t).strip()
        return merged_text or None

    def _extract_text_items(self, payload) -> list[str]:
        if payload is None:
            return []
        if isinstance(payload, str):
            return [payload.strip()] if payload.strip() else []
        if isinstance(payload, list):
            texts = []
            for item in payload:
                texts.extend(self._extract_text_items(item))
            return texts
        if isinstance(payload, dict):
            texts = []
            for key in ("text", "transcript", "asr_text", "value"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
            if not texts:
                for value in payload.values():
                    texts.extend(self._extract_text_items(value))
            return texts
        if hasattr(payload, "text") and isinstance(payload.text, str):
            return [payload.text.strip()] if payload.text.strip() else []
        return []

    def _extract_usage(self, response) -> dict:
        usage = {}

        candidates = [
            getattr(response, "usage", None),
            getattr(getattr(response, "output", None), "usage", None),
        ]
        if isinstance(response, dict):
            candidates.append(response.get("usage"))
            if isinstance(response.get("output"), dict):
                candidates.append(response["output"].get("usage"))

        parsed = None
        for item in candidates:
            if not item:
                continue
            if isinstance(item, dict):
                parsed = item
                break
            parsed = {
                "prompt_tokens": getattr(item, "prompt_tokens", None)
                or getattr(item, "input_tokens", None),
                "completion_tokens": getattr(item, "completion_tokens", None)
                or getattr(item, "output_tokens", None),
                "total_tokens": getattr(item, "total_tokens", None),
            }
            break

        if isinstance(parsed, dict):
            usage["prompt_tokens"] = parsed.get("prompt_tokens") or parsed.get(
                "input_tokens"
            )
            usage["completion_tokens"] = parsed.get("completion_tokens") or parsed.get(
                "output_tokens"
            )
            usage["total_tokens"] = parsed.get("total_tokens")

        return usage


class StableWhisperASR(ASR):
    """Implementation of ASR using Stable Whisper"""

    def __init__(self, whisper_model="large-v2", pre_load_model=None, **kwargs):
        super().__init__(**kwargs)
        self.whisper_model = whisper_model
        self.model = pre_load_model

    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        global stable_whisper
        if stable_whisper is None:
            import stable_whisper as _stable_whisper

            stable_whisper = _stable_whisper
        if self.model is None:
            self.model = stable_whisper.load_model(self.whisper_model, self.device)

        transcript = self.model.transcribe(
            str(audio_path), regroup=False, initial_prompt=init_prompt or ""
        )

        # Process the transcript
        (
            transcript.split_by_punctuation([".", "。", "?"])
            .merge_by_gap(0.15, max_words=3)
            .merge_by_punctuation([" "])
            .split_by_punctuation([".", "。", "?"])
        )

        transcript = transcript.to_dict()
        transcript = transcript["segments"]

        # Release GPU resources
        torch.cuda.empty_cache()

        return transcript


class OAIWhisperASR(ASR):
    """Implementation of ASR using OpenAI's Whisper model"""

    def __init__(self, model_id="large-v3", pre_load_model=None, **kwargs):
        super().__init__(**kwargs)
        global whisper
        if whisper is None:
            try:
                import whisper as _whisper

                whisper = _whisper
            except ImportError:
                raise ImportError("Please install whisper: pip install openai-whisper")
        try:
            self.model = whisper.load_model(
                name=model_id,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.model_name = model_id
            print(f"Loaded local Whisper model: {model_id}")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {e}")

    def get_transcript(self, audio_file, source_lang="en"):
        """
        Get transcript with timestamps from audio file.

        Args:
            audio_file (str): Path to audio file
            source_lang (str): Source language code

        Returns:
            list: List of segment dictionaries with 'start', 'end', 'text' keys
        """
        try:
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_file, language=source_lang, word_timestamps=True
            )

            # Convert to our expected format
            segments = []
            for segment in result.get("segments", []):
                segments.append(
                    {
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "text": segment.get("text", "").strip(),
                    }
                )

            return segments if segments else None

        except Exception as e:
            print(f"Error in HuggingfaceWhisperASR transcription: {e}")
            traceback.print_exc()
            return None


class HuggingfaceWhisperASR(ASR):
    """Implementation of ASR using Whisper models from Huggingface"""

    def __init__(
        self, model_id="openai/whisper-large-v3", pre_load_model=None, **kwargs
    ):
        super().__init__(**kwargs)
        global WhisperForConditionalGeneration, WhisperProcessor
        if WhisperForConditionalGeneration is None:
            from transformers import (
                WhisperForConditionalGeneration as _WhisperForConditionalGeneration,
            )
            from transformers import WhisperProcessor as _WhisperProcessor

            WhisperForConditionalGeneration = _WhisperForConditionalGeneration
            WhisperProcessor = _WhisperProcessor

        self.model_id = model_id
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Initialize model and processor during initialization
        if pre_load_model:
            self.model = pre_load_model
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            self.model.to(self.device)

        self.processor = WhisperProcessor.from_pretrained(
            self.model_id, sample_rate=16000
        )

    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        try:
            # Apply language and prompt if provided
            self.log(f"Transcribing audio file: {audio_path}")
            input_speech = self.load_audio(audio_path)
            input_features = self.processor(
                input_speech, return_tensors="pt", sample_rate=16000
            ).input_features
            input_features = input_features.to(self.device)

            # Set up generation parameters
            generation_kwargs = {
                "return_dict_in_generate": True,
                "return_timestamps": True,
            }

            if init_prompt:
                try:
                    prompt_ids = self.processor.get_prompt_ids(init_prompt)
                    prompt_ids = torch.tensor(prompt_ids).to(self.device)
                    generation_kwargs["prompt_ids"] = prompt_ids
                except Exception as e:
                    self.log(f"Warning: Could not set prompt: {e}")

            # Generate transcription
            outputs = self.model.generate(input_features, **generation_kwargs)

            # Decode the output
            if hasattr(outputs, "sequences"):
                transcript_text = self.processor.decode(
                    outputs.sequences[0], skip_special_tokens=True
                )
            else:
                transcript_text = self.processor.decode(
                    outputs[0], skip_special_tokens=True
                )

            # For now, return a simple segment format (this is a basic implementation)
            # In a full implementation, you would parse timestamps from the model output
            transcript = [
                {
                    "start": 0.0,
                    "end": 30.0,  # Placeholder duration
                    "text": transcript_text,
                }
            ]

            # Release GPU resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return transcript

        except Exception as e:
            self.log(f"Error in HuggingfaceWhisperASR transcription: {e}")
            # Release GPU resources on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None


if __name__ == "__main__":
    asr = HuggingfaceWhisperASR(model_id="openai/whisper-large-v3")

    asr.get_transcript(
        "/home/mlp/eason/ViDove/src/VAD/0a4b82fc-fff5-4254-a7b1-1c6c88ff538a.wav"
    )
