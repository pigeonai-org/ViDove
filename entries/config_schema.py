from typing import Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class VideoDownloadConfig(BaseModel):
    """YouTube download configuration"""
    resolution: Union[int, Literal["best"]] = Field(
        default=480, 
        description="Video resolution: 360, 480, 720, best(best available)"
    )


class MemoryConfig(BaseModel):
    """Memory and knowledge base configuration"""
    enable_local_knowledge: bool = Field(
        default=True, 
        description="Whether to enable local knowledge base"
    )
    enable_vision_knowledge: bool = Field(
        default=True, 
        description="Whether to enable vision knowledge"
    )
    enable_web_search: bool = Field(
        default=False, 
        description="Whether to enable web search"
    )
    local_knowledge_path: str = Field(
        default="/home/macrodove/ViDove/domain_dict",
        description="Local knowledge base path"
    )


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    enable_audio: bool = Field(
        default=True, 
        description="Whether to enable audio processing"
    )
    audio_agent: Literal["GeminiAudioAgent", "WhisperAudioAgent", "QwenAudioAgent", "GPT4oAudioAgent"] = Field(
        default="GeminiAudioAgent",
        description="Audio agent: GeminiAudioAgent, WhisperAudioAgent, QwenAudioAgent, GPT4oAudioAgent"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Model path, replace with your own model path"
    )
    VAD_model: Literal["pyannote/speaker-diarization-3.1", "API"] = Field(
        default="API",
        description="Voice Activity Detection model: pyannote/speaker-diarization-3.1 or pyannote API"
    )
    src_lang: str = Field(
        default="en",
        description="Source language"
    )
    tgt_lang: str = Field(
        default="zh",
        description="Target language"
    )


class VisionConfig(BaseModel):
    """Vision processing configuration"""
    enable_vision: bool = Field(
        default=True, 
        description="Whether to enable vision processing"
    )
    vision_model: Literal["CLIP", "gpt-4o", "gpt-4o-mini"] = Field(
        default="gpt-4o",
        description="Vision model: CLIP or gpt-4o"
    )
    model_path: str = Field(
        default="./ViDove/vision_model/clip-vit-base-patch16",
        description="Model path, replace with your own model path"
    )
    frame_cache_dir: str = Field(
        default="./cache",
        description="Frame cache directory, should be cleared after task finished"
    )
    frame_per_seg: int = Field(
        default=4,
        description="Number of frames extracted from segment"
    )


class PreProcessConfig(BaseModel):
    """Pre-processing module configuration"""
    sentence_form: bool = Field(
        default=True,
        description="Whether to perform sentence formatting"
    )
    spell_check: bool = Field(
        default=False,
        description="Whether to perform spell checking"
    )
    term_correct: bool = Field(
        default=True,
        description="Whether to perform term correction"
    )


class TranslationConfig(BaseModel):
    """Translation module configuration"""
    model: Literal["gpt-4", "gpt-4o-mini", "gpt-4o", "Assistant", "Multiagent", "RAG", "gpt-5", "gpt-5-mini"] = Field(
        default="gpt-4o",
        description="Translation model: gpt-4, gpt-4o-mini, gpt-4o, Assistant, Multiagent, RAG"
    )
    chunk_size: int = Field(
        default=2000,
        description="Translation chunk size"
    )
    use_history: bool = Field(
        default=True,
        description="Include recent translation history in each request (may reduce throughput)"
    )
    max_retries: int = Field(
        default=1,
        description="Max retries per chunk for transient API errors"
    )
    # Note: Parallelism is globally controlled by TaskConfig.num_workers (>1 enables parallel). 
    # The deprecated fields 'parallel' and 'workers' are intentionally removed.


class PostProcessConfig(BaseModel):
    """Post-processing module configuration"""
    enable_post_process: bool = Field(
        default=True,
        description="Whether to enable post-processing"
    )
    check_len_and_split: bool = Field(
        default=True,
        description="Whether to check length and split sentences"
    )
    remove_trans_punctuation: bool = Field(
        default=True,
        description="Whether to remove translation punctuation"
    )


class ProofreaderConfig(BaseModel):
    """Proofreader configuration"""
    enable_proofreading: bool = Field(
        default=True,
        description="Whether to enable proofreading"
    )
    window_size: int = Field(
        default=5,
        description="Proofreading window size, number of sentences per proofreading chunk"
    )
    short_term_memory_len: int = Field(
        default=5,
        description="Maximum number of sentences stored in short term memory"
    )
    enable_short_term_memory: bool = Field(
        default=False,
        description="Whether to use short term memory for proofreading"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to print the proofreading process"
    )


class EditorConfig(BaseModel):
    """Editor configuration"""
    enable_editor: bool = Field(
        default=True,
        description="Whether to enable editor"
    )
    user_instruction: Literal["none", "formal", "casual", "technical"] = Field(
        default="none",
        description="User instruction style for the editor"
    )
    editor_context_window: int = Field(
        default=10,
        description="Editor context window size, number of sentences to be provided as context"
    )
    history_length: int = Field(
        default=5,
        description="Editor history length, number of sentences to be provided as history"
    )


class OutputTypeConfig(BaseModel):
    """Output type configuration"""
    subtitle: Literal["srt", "ass"] = Field(
        default="srt",
        description="Subtitle format: srt or ass"
    )
    video: bool = Field(
        default=True,
        description="Whether to output video"
    )
    bilingual: bool = Field(
        default=True,
        description="Whether to output bilingual subtitles"
    )


class TaskConfig(BaseModel):
    """Main task configuration class"""
    # Basic configuration
    source_lang: Literal["EN", "ZH", "ES", "FR", "DE", "RU", "JA", "AR", "KR"] = Field(
        default="EN",
        description="Source language"
    )
    num_workers: int = Field(
        default=8,
        description="Global number of worker threads for VAD, proofreading, and editing"
    )
    target_lang: Literal["EN", "ZH", "ES", "FR", "DE", "RU", "JA", "AR", "KR"] = Field(
        default="ZH",
        description="Target language"
    )
    domain: Literal["General", "SC2", "CS:GO"] = Field(
        default="General",
        description="Domain"
    )
    
    # User instructions
    instructions: Optional[list[str]] = Field(
        default=None,
        description="List of user instructions for the editor agent"
    )
    
    # Module configurations
    video_download: VideoDownloadConfig = Field(
        default_factory=VideoDownloadConfig,
        description="YouTube download configuration"
    )
    MEMEORY: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory and knowledge base configuration"
    )
    audio: AudioConfig = Field(
        default_factory=AudioConfig,
        description="Audio processing configuration"
    )
    vision: VisionConfig = Field(
        default_factory=VisionConfig,
        description="Vision processing configuration"
    )
    pre_process: PreProcessConfig = Field(
        default_factory=PreProcessConfig,
        description="Pre-processing module configuration"
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig,
        description="Translation module configuration"
    )
    post_process: PostProcessConfig = Field(
        default_factory=PostProcessConfig,
        description="Post-processing module configuration"
    )
    proofreader: ProofreaderConfig = Field(
        default_factory=ProofreaderConfig,
        description="Proofreader configuration"
    )
    editor: EditorConfig = Field(
        default_factory=EditorConfig,
        description="Editor configuration"
    )
    output_type: OutputTypeConfig = Field(
        default_factory=OutputTypeConfig,
        description="Output type configuration"
    )
    
    # Runtime configuration (may be added by run.py)
    api_source: Optional[str] = Field(
        default=None,
        description="API source: openai or azure"
    )
    is_assistant: Optional[bool] = Field(
        default=None,
        description="Whether it is assistant mode"
    )

    @field_validator('source_lang', 'target_lang')
    def validate_language_codes(cls, v):
        """Validate language code format"""
        valid_codes = ['EN', 'ZH', 'ES', 'FR', 'DE', 'RU', 'JA', 'AR', 'KR', 'IT', 'PT']
        if v not in valid_codes:
            raise ValueError(f'Language code must be one of: {valid_codes}')
        return v

    @field_validator('video_download')
    def validate_video_download(cls, v):
        """Validate video download configuration"""
        if isinstance(v.resolution, int) and v.resolution not in [360, 480, 720]:
            raise ValueError('Resolution must be 360, 480, 720 or "best"')
        return v

    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> 'TaskConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return self.dict()

    def to_yaml(self) -> str:
        """Convert to YAML format string"""
        import yaml
        return yaml.dump(self.dict(), default_flow_style=False, allow_unicode=True)

    def to_flat_dict(self) -> dict:
        """Convert to flattened dictionary format for web interface"""
        flat_dict: dict = {}
        config_dict = self.dict()

        # Top-level fields
        flat_dict["source_lang"] = config_dict["source_lang"]
        flat_dict["target_lang"] = config_dict["target_lang"]
        flat_dict["domain"] = config_dict["domain"]
        flat_dict["num_workers"] = config_dict["num_workers"]

        # Flatten nested configurations
        for section_name, section_data in config_dict.items():
            if isinstance(section_data, dict) and section_name not in [
                "source_lang",
                "target_lang",
                "domain",
                "instructions",
                "api_source",
                "is_assistant",
                "num_workers",
            ]:
                for field_name, field_value in section_data.items():
                    flat_dict[f"{section_name}.{field_name}"] = field_value

        # Handle special cases
        if config_dict.get("instructions"):
            flat_dict["instructions"] = config_dict["instructions"]
        if config_dict.get("api_source"):
            flat_dict["api_source"] = config_dict["api_source"]
        if config_dict.get("is_assistant") is not None:
            flat_dict["is_assistant"] = config_dict["is_assistant"]

        return flat_dict
    
    @classmethod
    def from_flat_dict(cls, flat_dict: dict) -> 'TaskConfig':
        """Create TaskConfig from flattened dictionary format"""
        nested_dict = {}
        
        # Handle top-level fields
        for key in ['source_lang', 'target_lang', 'domain', 'instructions', 'api_source', 'is_assistant', 'num_workers']:
            if key in flat_dict:
                nested_dict[key] = flat_dict[key]
        
        # Group flattened fields back into sections
        sections = {}
        for key, value in flat_dict.items():
            if '.' in key:
                section, field = key.split('.', 1)
                if section not in sections:
                    sections[section] = {}
                sections[section][field] = value
        
        # Add sections to nested dict
        nested_dict.update(sections)
        
        return cls(**nested_dict)


# Convenience functions
def load_task_config(config_path: Union[str, Path] = "./configs/task_config.yaml") -> TaskConfig:
    """Convenience function to load task configuration"""
    return TaskConfig.from_yaml_file(config_path)


def validate_task_config(config_dict: dict) -> TaskConfig:
    """Validate configuration dictionary and return TaskConfig instance"""
    return TaskConfig(**config_dict)
