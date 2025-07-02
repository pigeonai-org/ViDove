from abc import ABC, abstractmethod
import torch
import stable_whisper
import whisper
import traceback
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
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
        if method == "whisper-api":
            return WhisperAPIASR(**kwargs)
        elif "stable" in method:
            whisper_model = method.split("-")[2]
            return StableWhisperASR(whisper_model=whisper_model, **kwargs)
        elif "oai" in method:
            model_id = method.split("-")[2] if "-" in method else "large-v3"
            return OAIWhisperASR(model_id=model_id, **kwargs)
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


class OAIWhisperASR(ASR):
    """Implementation of ASR using OpenAI's Whisper model"""
    
    def __init__(self, model_id="large-v3", pre_load_model=None, **kwargs):
        super().__init__(**kwargs)
        try:
            self.model = whisper.load_model(name=model_id, device= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.model_name = model_id
            print(f"Loaded local Whisper model: {model_id}")
        except ImportError:
            raise ImportError("Please install whisper: pip install openai-whisper")
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
                audio_file,
                language=source_lang,
                word_timestamps=True
            )
            
            # Convert to our expected format
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment.get('start', 0.0),
                    'end': segment.get('end', 0.0),
                    'text': segment.get('text', '').strip()
                })
            
            return segments if segments else None
            
        except Exception as e:
            print(f"Error in HuggingfaceWhisperASR transcription: {e}")
            traceback.print_exc()
            return None

    
    

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
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            self.model.to(self.device)
            
        self.processor = WhisperProcessor.from_pretrained(self.model_id, sample_rate=16000)

    
    def get_transcript(self, audio_path, source_lang=None, init_prompt=None):
        try:
            # Apply language and prompt if provided
            self.log(f"Transcribing audio file: {audio_path}")
            input_speech = self.load_audio(audio_path)
            input_features = self.processor(input_speech, return_tensors="pt", sample_rate=16000).input_features
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
            if hasattr(outputs, 'sequences'):
                transcript_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
            else:
                transcript_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # For now, return a simple segment format (this is a basic implementation)
            # In a full implementation, you would parse timestamps from the model output
            transcript = [{
                'start': 0.0,
                'end': 30.0,  # Placeholder duration
                'text': transcript_text
            }]
            
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

    asr.get_transcript("/home/mlp/eason/ViDove/src/VAD/0a4b82fc-fff5-4254-a7b1-1c6c88ff538a.wav")