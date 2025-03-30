from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch
from abc import ABC, abstractmethod
import base64
import os
from google import genai
from google.genai import types
from google.genai.types import Part, HttpOptions
from audio_prompt import AUDIO_TRANSCRIBE_PROMPT, AUDIO_ANALYZE_PROMPT, AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES
import json
from ASR import ASR
from VAD import VAD
import librosa


class AudioAgent(ABC):
    def __init__(self, model_name, audio_config: dict=None):
        self.model_name = model_name
        self.audio_config = audio_config
        self.device = None
        self.load_model()
    
    def segment_audio(self, audio_path):
        vad = VAD(model_name=self.audio_config["vad_model_name"], src_lang=self.audio_config["src_lang"], tgt_lang=self.audio_config["tgt_lang"])
        self.segments = vad.get_speaker_segments(audio_path)
        VAD.clip_audio_and_save(self.segments, audio_path, self.audio_config["cache_dir"])
        return self.segments

    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def transcribe(self, audio_path, visual_cues=None):
        pass
    
    @abstractmethod
    def analyze_audio(self, audio_path):
        pass

class ClassicAudioAgent(AudioAgent):
    def __init__(self, model_name="whisper-api"):
        super().__init__(model_name)

    def load_model(self):
        self.ASR_model = ASR.create(self.model_name)
        
    def transcribe(self, audio_path):
        return self.ASR_model.get_transcript(audio_path)


# TODO: @George please implement this
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
    def __init__(self, model_name="gemini-1.5-pro"):
        super().__init__(model_name)

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

        response = self.model.models.generate_content(
            model=self.model_name,
            contents=contents,
        )

        return self.parse_response(response.text)
    
    def analyze_audio(self, audio_path):
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

        return self.parse_response(response.text)


if __name__ == "__main__":
    agent = QwenAudioAgent()
    visual_cues = "a man named lowko is talking about the game of starcraft"
    print(agent.transcribe("C:\\Work\\GitRepos\\task.mp3"))
    print(agent.transcribe("C:\\Work\\GitRepos\\task.mp3", visual_cues))
    print(agent.analyze_audio("C:\\Work\\GitRepos\\task.mp3"))
