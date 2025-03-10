from abc import ABC, abstractmethod
from pathlib import Path
import logging
import torch
import stable_whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI

class ASR(ABC):
    """Abstract base class for all ASR implementations"""
    
    def __init__(self, device=None, logger=None):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logger
    
    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

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
        if method == "whisper-api":
            return WhisperAPIASR(**kwargs)
        elif "stable" in method:
            whisper_model = method.split("-")[2]
            return StableWhisperASR(whisper_model=whisper_model, **kwargs)
        else:
            return HuggingfaceWhisperASR(model_id=method, **kwargs)



class WhisperAPIASR(ASR):
    """Implementation of ASR using OpenAI's Whisper API"""
    
    def __init__(self, client=None, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        if self.client is None:
            self.client = OpenAI()
        
    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        with open(audio_path, 'rb') as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="srt", 
                language=source_lang.lower() if source_lang else None, 
                prompt=init_prompt or ""
            )
        return transcript


class StableWhisperASR(ASR):
    """Implementation of ASR using Stable Whisper"""
    
    def __init__(self, whisper_model="large-v2", pre_load_model=None, **kwargs):
        super().__init__(**kwargs)
        self.whisper_model = whisper_model
        self.model = pre_load_model
        
    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        if self.model is None:
            self.model = stable_whisper.load_model(self.whisper_model, self.device)
            
        transcript = self.model.transcribe(
            str(audio_path), 
            regroup=False, 
            initial_prompt=init_prompt or ""
        )
        
        # Process the transcript
        (
            transcript
            .split_by_punctuation(['.', '。', '?'])
            .merge_by_gap(.15, max_words=3)
            .merge_by_punctuation([' '])
            .split_by_punctuation(['.', '。', '?'])
        )
        
        transcript = transcript.to_dict()
        transcript = transcript['segments']
        
        # Release GPU resources
        torch.cuda.empty_cache()
        
        return transcript


class HuggingfaceWhisperASR(ASR):
    """Implementation of ASR using Whisper models from Huggingface"""
    
    def __init__(self, model_id="openai/whisper-large-v3", pre_load_model=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Initialize model and processor during initialization
        if pre_load_model:
            self.model = pre_load_model
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, 
                torch_dtype=self.torch_dtype
            )
            self.model.to(self.device)
            
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        # Apply language and prompt if provided
        self.log(f"Transcribing audio file: {audio_path}")
        generate_kwargs = {}
        if init_prompt:
            generate_kwargs["prompt"] = init_prompt
        
        transcript_whisper = self.pipe(str(audio_path), return_timestamps=True)
        
        # Convert format
        transcript = []
        for i in transcript_whisper['chunks']:
            transcript.append({
                'start': i['timestamp'][0], 
                'end': i['timestamp'][1], 
                'text': i['text']
            })
            
        # Release GPU resources
        torch.cuda.empty_cache()
            
        return transcript