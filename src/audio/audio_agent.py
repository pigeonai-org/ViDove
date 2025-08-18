from transformers import AutoProcessor
import torch
from abc import ABC, abstractmethod
from openai import OpenAI
import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from src.audio.audio_prompt import AUDIO_TRANSCRIBE_PROMPT, AUDIO_ANALYZE_PROMPT, AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES, AUDIO_TRANSCRIBE_GPT_PROMPT
import json
from src.audio.ASR import ASR
from src.SRT.srt import SrtScript, SrtSegment
from src.audio.VAD import VAD
import librosa


class AudioAgent(ABC):
    def __init__(self, model_name, audio_config: dict=None):
        self.model_name = model_name
        self.audio_config = audio_config or {}
        self.device = None
        self.load_model()
        # Initialize VAD only if configured
        self.VAD_model = None
        if all(k in self.audio_config for k in ("VAD_model", "src_lang", "tgt_lang")):
            self.VAD_model = VAD(
                model_name_or_path=self.audio_config["VAD_model"],
                src_lang=self.audio_config["src_lang"],
                tgt_lang=self.audio_config["tgt_lang"],
                min_segment_seconds=float(self.audio_config.get("min_segment_seconds", 1.0)),
            )
    
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

    def transcribe_batch(self, items: list[dict], max_workers: int = 4) -> dict[int, list[dict]]:
        """
        Concurrent transcription helper.
        items: list of {idx: int, audio_path: str, visual_cues?: str}
        Returns: { idx: [ {start: str, end: str, text: str}, ... ] }
        """
        results: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.transcribe, it["audio_path"], it.get("visual_cues")): it["idx"]
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
    def __init__(self, model_name="gpt-4o-mini-transcribe", audio_config: dict | None = None):
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
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    prompt=AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES.format(visual_cues=visual_cues) if visual_cues else AUDIO_TRANSCRIBE_PROMPT,
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
            seg_audio = AudioSegment.from_file(audio_path)
            duration_secs = len(seg_audio) / 1000.0
            return [{
                "start": self._seconds_to_srt_time(0.0),
                "end": self._seconds_to_srt_time(duration_secs),
                "text": text or "",
            }]
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


class ClassicAudioAgent(AudioAgent):
    def __init__(self, model_name="whisper-api", audio_config: dict | None = None):
        # For whisper-api, do not initialize VAD (pass empty config);
        # Classic agent is primarily used with whisper-api here.
        if model_name == "whisper-api":
            super().__init__(model_name, audio_config={})
        else:
            super().__init__(model_name, audio_config=audio_config)

    def load_model(self):
        self.ASR_model = ASR.create(self.model_name)
        
    def segment_audio(self, audio_path, cache_dir):
        # Whisper API path: do not perform VAD; create a single placeholder segment
        srt = SrtScript(src_lang=self.audio_config.get("src_lang", "en") if self.audio_config else "en",
                        tgt_lang=self.audio_config.get("tgt_lang", "zh") if self.audio_config else "zh")
        # single segment covering whole file; exact end not needed since ASR returns timestamped segments
        placeholder = SrtSegment(src_lang=srt.src_lang, tgt_lang=srt.tgt_lang,
                                    src_text="", translation="", speaker="",
                                    start_time=0.0, end_time=0.0, idx=0)
        placeholder.audio_path = audio_path
        srt.segments.append(placeholder)
        self.segments = srt
        return srt
        # Fallback to base behavior (use VAD)

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
                        segments.append({"start": start_str, "end": end_str, "text": text})
                except Exception:
                    # ignore malformed entries
                    pass
            else:
                i += 1
        return segments

    def _srt_time_to_seconds(self, s: str) -> float:
        # HH:MM:SS,mmm
        try:
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
        result = self.ASR_model.get_transcript(audio_path)
        # If Whisper API returns SRT text, keep as SRT time strings
        if isinstance(result, str):
            return self._parse_srt_to_segments(result)
        # If list of dicts with numeric times, convert to SRT strings
        norm = []
        for seg in (result or []):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")
            # Some implementations may already return HH:MM:SS,mmm strings
            if isinstance(start, str) and "-->" not in start:
                start_srt = start
            else:
                start_srt = self._seconds_to_srt_time(start if not isinstance(start, str) else 0.0)
            if isinstance(end, str) and "-->" not in end:
                end_srt = end
            else:
                end_srt = self._seconds_to_srt_time(end if not isinstance(end, str) else 0.0)
            norm.append({"start": start_srt, "end": end_srt, "text": text})
        return norm


class QwenAudioAgent(AudioAgent):
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct"):
        super().__init__(model_name)

    def load_model(self):
        from transformers import Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_name, device_map="auto")

    def transcribe(self, audio_path, visual_cues=None):
        audios = [librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[0]]

        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES.format(visual_cues=visual_cues) if visual_cues else AUDIO_TRANSCRIBE_PROMPT}
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        # Move all tensors to CUDA while preserving the BatchEncoding structure
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=5000) # TODO: The value of max_length may need to be adjusted.
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response


    def analyze_audio(self, audio_path):

        audios = [librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[0]]

        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": AUDIO_ANALYZE_PROMPT}
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        # Move all tensors to CUDA while preserving the BatchEncoding structure
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=5000) # TODO: The value of max_length may need to be adjusted.
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response



class GeminiAudioAgent(AudioAgent):
    def __init__(self, model_name="gemini-2.5-flash",audio_config: dict=None):
        super().__init__(model_name,audio_config)
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None

    def set_agent_history_logger(self, logger):
        """Set the agent history logger from task"""
        self.agent_history_logger = logger

    def load_model(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided either directly or via GEMINI_API_KEY environment variable")
        
        self.model = genai.Client(api_key=api_key)
    
    def parse_response(self, response, parsing_retries=5):
        for retry in range(parsing_retries):
            try:
                # Check if response is wrapped in code block markers
                if "```json" in response:
                    # Extract the actual JSON content from between the code block markers
                    json_content = response.split("```json", 1)[1].split("```", 1)[0].strip()
                    gemini_results = json.loads(json_content)
                    return gemini_results
                else:
                    # If no code block markers, try parsing directly
                    gemini_results = json.loads(response)
                    return gemini_results
                    
            except Exception as e:
                if retry < parsing_retries - 1:
                    print(f"Failed to parse response (attempt {retry + 1}/{parsing_retries}): {str(e)}")
                    print("Retrying with new response...")
                    continue
                else:
                    print(f"Failed to parse response after {parsing_retries} attempts")
                    return None
    
    def transcribe(self, audio_path, visual_cues=None):
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "audio_agent", "message": "Audio loaded, let me give it a good ol\' chomp... chomp chomp! 🎧"}')
        
        with open(audio_path, "rb") as audio:
            audio_data = audio.read()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES.format(visual_cues=visual_cues) if visual_cues else AUDIO_TRANSCRIBE_PROMPT),
                    types.Part.from_bytes(
                        data=audio_data,
                        mime_type="audio/wav"
                    )
                ]
            )
        ]

        resp = None

        for retry in range(5):
            if resp is None:
                response = self.model.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                resp = self.parse_response(response.text)
            else:
                if self.agent_history_logger:
                    self.agent_history_logger.info('{"role": "audio_agent", "message": "Transcription done! If I missed a beat, blame the waveform, not me 😜"}')
                return resp
        
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "audio_agent", "message": "Oops, 5 tries and still no luck. Even the best of us have off days!"}')
        self.logger.error("Failed to transcribe audio after 5 retries.")
        return resp

    def analyze_audio(self, audio_path):
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "audio_agent", "message": "Analyzing audio... let me put on my sonic goggles! 🥽"}')
        
        with open(audio_path, "rb") as audio:
            audio_data = audio.read()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=AUDIO_ANALYZE_PROMPT),
                    types.Part.from_bytes(
                        data=audio_data,
                        mime_type="audio/wav"
                    )
                ]
            )
        ]

        response = self.model.models.generate_content(
            model=self.model_name,
            contents=contents,
        )

        result = self.parse_json_response(response.text)
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "audio_agent", "message": "Audio analysis complete! If I missed a note, it was jazz."}')
        return result


if __name__ == "__main__":
    agent = QwenAudioAgent()
    visual_cues = "a man named lowko is talking about the game of starcraft"
    print(agent.transcribe("C:\\Work\\GitRepos\\task.mp3"))
    print(agent.transcribe("C:\\Work\\GitRepos\\task.mp3", visual_cues))
    print(agent.analyze_audio("C:\\Work\\GitRepos\\task.mp3"))
