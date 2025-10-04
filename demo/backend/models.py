"""
Pydantic models and type definitions for the ViDove web backend.
"""
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class AgentConversationMessage(BaseModel):
    role: str
    message: str
    timestamp: Optional[str] = None


class AgentConversationResponse(BaseModel):
    task_id: str
    conversation: List[AgentConversationMessage]
    is_live: bool  # Whether this is from a live task or saved conversation


class ConfigurationValue(BaseModel):
    type: Literal["select", "boolean", "number"]
    options: Optional[List[Union[str, int]]] = None
    range: Optional[List[int]] = None
    default: Union[str, int, bool]
    description: str


class SessionConfig(BaseModel):
    source_lang: Literal["EN", "ZH", "ES", "FR", "DE", "RU", "JA", "AR", "KR"] = "EN"
    target_lang: Literal["EN", "ZH", "ES", "FR", "DE", "RU", "JA", "AR", "KR"] = "ZH"
    domain: Literal["General", "SC2", "CS:GO"] = "General"
    num_workers: int = Field(default=8, description="Global number of worker threads for VAD, proofreading, and editing")
    video_download_resolution: Union[Literal[360, 480, 720], Literal["best"]] = Field(default=480, alias="video_download.resolution")
    translation_model: Literal["gpt-4", "gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini"] = Field(default="gpt-5", alias="translation.model")
    translation_chunk_size: int = Field(default=2000, alias="translation.chunk_size")
    translation_use_history: bool = Field(default=True, alias="translation.use_history", description="Include recent translation history in each request")
    translation_max_retries: int = Field(default=1, alias="translation.max_retries", description="Max retries per chunk for transient API errors")
    # Audio is always enabled, agent selection is required
    audio_audio_agent: Literal["GeminiAudioAgent", "WhisperAudioAgent", "QwenAudioAgent", "GPT4oAudioAgent"] = Field(default="WhisperAudioAgent", alias="audio.audio_agent", description="Audio agent for transcription (always enabled)")
    audio_model_path: Optional[str] = Field(default=None, alias="audio.model_path")
    audio_VAD_model: Literal["pyannote/speaker-diarization-3.1", "API"] = Field(default="API", alias="audio.VAD_model")
    audio_src_lang: str = Field(default="en", alias="audio.src_lang")
    audio_tgt_lang: str = Field(default="zh", alias="audio.tgt_lang")
    vision_enable_vision: bool = Field(default=False, alias="vision.enable_vision")
    vision_vision_model: Literal["CLIP", "gpt-4o", "gpt-4o-mini"] = Field(default="gpt-4o", alias="vision.vision_model")
    vision_model_path: str = Field(default="./ViDove/vision_model/clip-vit-base-patch16", alias="vision.model_path")
    vision_frame_cache_dir: str = Field(default="./cache", alias="vision.frame_cache_dir")
    vision_frame_per_seg: int = Field(default=4, alias="vision.frame_per_seg")
    pre_process_sentence_form: bool = Field(default=True, alias="pre_process.sentence_form")
    pre_process_spell_check: bool = Field(default=False, alias="pre_process.spell_check")
    pre_process_term_correct: bool = Field(default=True, alias="pre_process.term_correct")
    post_process_enable_post_process: bool = Field(default=True, alias="post_process.enable_post_process")
    post_process_check_len_and_split: bool = Field(default=True, alias="post_process.check_len_and_split")
    post_process_remove_trans_punctuation: bool = Field(default=True, alias="post_process.remove_trans_punctuation")
    proofreader_enable_proofreading: bool = Field(default=True, alias="proofreader.enable_proofreading")
    proofreader_window_size: int = Field(default=5, alias="proofreader.window_size")
    proofreader_short_term_memory_len: int = Field(default=5, alias="proofreader.short_term_memory_len")
    proofreader_enable_short_term_memory: bool = Field(default=False, alias="proofreader.enable_short_term_memory")
    proofreader_verbose: bool = Field(default=True, alias="proofreader.verbose")
    editor_enable_editor: bool = Field(default=True, alias="editor.enable_editor")
    editor_user_instruction: Literal["none", "formal", "casual", "technical"] = Field(default="none", alias="editor.user_instruction")
    editor_editor_context_window: int = Field(default=10, alias="editor.editor_context_window")
    editor_history_length: int = Field(default=5, alias="editor.history_length")
    MEMEORY_enable_local_knowledge: bool = Field(default=True, alias="MEMEORY.enable_local_knowledge")
    MEMEORY_enable_web_search: bool = Field(default=False, alias="MEMEORY.enable_web_search")
    MEMEORY_enable_vision_knowledge: bool = Field(default=True, alias="MEMEORY.enable_vision_knowledge")
    MEMEORY_local_knowledge_path: str = Field(default="/home/macrodove/ViDove/domain_dict", alias="MEMEORY.local_knowledge_path")
    output_type_video: bool = Field(default=False, alias="output_type.video")
    output_type_bilingual: bool = Field(default=True, alias="output_type.bilingual")
    output_type_subtitle: Literal["srt", "ass"] = Field(default="srt", alias="output_type.subtitle")
    # Custom user instructions for translation habits/jargon
    instructions: Optional[List[str]] = Field(default=None, description="List of user instructions for the editor agent")
    # File upload tracking
    uploaded_file_path: Optional[str] = None
    uploaded_file_name: Optional[str] = None
    youtube_url: Optional[str] = None
    input_type: Optional[Literal["youtube", "video", "audio", "srt"]] = None

    class Config:
        # Allow field names with dots and underscores
        populate_by_name = True


class ConfigurationSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    current_config: SessionConfig
    is_complete: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class StartSessionResponse(BaseModel):
    session_id: str
    message: str
    current_config: SessionConfig


class SendMessageResponse(BaseModel):
    message: str
    config_updates: Dict[str, Any]
    current_config: SessionConfig
    is_complete: bool
    auto_created_task_id: Optional[str] = None


class TaskRequest(BaseModel):
    session_id: str
    input_type: Literal["youtube", "video", "audio", "srt"]
    input_data: str


class TaskStatus(BaseModel):
    task_id: str
    session_id: str  # Add session_id to associate task with user session
    status: Literal["CREATED", "RUNNING", "COMPLETED", "FAILED"]
    input_type: Literal["youtube", "video", "audio", "srt"]
    progress: Optional[int] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    agent_conversation: Optional[List[AgentConversationMessage]] = None  # Saved conversation history
    working_directory: Optional[str] = None  # Temporary working directory path for live tasks


class TaskInfo(BaseModel):
    task_id: str
    status: str
    created_at: str
    input_type: str


class CreateTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class UploadFileResponse(BaseModel):
    filename: str
    file_path: str
    size: int


class YouTubeUrlRequest(BaseModel):
    youtube_url: str


class YouTubeUrlResponse(BaseModel):
    youtube_url: str
    message: str


class ConfigResponse(BaseModel):
    config: SessionConfig
    is_complete: Optional[bool] = None


class ErrorResponse(BaseModel):
    detail: str


class FileDownloadInfo(BaseModel):
    filename: str
    file_type: Literal["subtitle", "video", "log", "unknown"]
    size_bytes: int
    created_at: float


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    has_results: bool
    files: List[FileDownloadInfo]
    error: Optional[str] = None
