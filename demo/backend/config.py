"""
Configuration schema and constants for the ViDove web backend.
"""
from models import ConfigurationValue

# Memory management settings
SESSION_CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour
SESSION_MAX_AGE_HOURS = 24
TASK_MAX_AGE_HOURS = 48
MAX_SESSIONS_IN_MEMORY = 1000
MAX_TASKS_IN_MEMORY = 500
MAX_CONCURRENT_TASKS = 3

# Rate limiting and bot protection
MAX_SESSIONS_PER_IP = 10  # Maximum sessions per IP address
MAX_TASKS_PER_SESSION = 20  # Maximum tasks per session
RATE_LIMIT_REQUESTS_PER_MINUTE = 60  # Max requests per IP per minute
RATE_LIMIT_SESSION_CREATE_PER_HOUR = 20  # Max session creations per IP per hour
RATE_LIMIT_TASK_CREATE_PER_HOUR = 10  # Max task creations per IP per hour
RATE_LIMIT_FILE_UPLOAD_PER_HOUR = 10  # Max file uploads per IP per hour

# Emergency memory protection
MEMORY_EMERGENCY_THRESHOLD_PERCENT = 90  # Stop accepting new requests at 90% memory
MEMORY_WARNING_THRESHOLD_PERCENT = 80  # Start aggressive cleanup at 80% memory
ENABLE_EMERGENCY_PROTECTION = True  # Enable emergency memory protection


# Configuration schema with proper typing
CONFIGURATION_SCHEMA = {
    "source_lang": ConfigurationValue(
        type="select",
        options=["EN", "ZH", "ES", "FR", "DE", "RU", "JA", "AR", "KR"],
        default="EN",
        description="Source language of the video content"
    ),
    "target_lang": ConfigurationValue(
        type="select", 
        options=["EN", "ZH", "ES", "FR", "DE", "RU", "JA", "AR", "KR"],
        default="ZH",
        description="Target language for translation"
    ),
    "domain": ConfigurationValue(
        type="select",
        options=["General", "SC2", "CS:GO"],
        default="General",
        description="Domain/field of the content for specialized translation"
    ),
    "num_workers": ConfigurationValue(
        type="number",
        range=[1, 16],
        default=8,
        description="Global number of worker threads for parallel processing"
    ),
    "video_download.resolution": ConfigurationValue(
        type="select",
        options=[360, 480, 720, "best"],
        default=480,
        description="Video resolution for download"
    ),
    "translation.model": ConfigurationValue(
        type="select",
        options=["gpt-4", "gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini"],
        default="gpt-4o",
        description="LLM model for translation"
    ),
    "translation.chunk_size": ConfigurationValue(
        type="number",
        range=[100, 5000],
        default=2000,
        description="Text chunk size for translation"
    ),
    "translation.use_history": ConfigurationValue(
        type="boolean",
        default=True,
        description="Include recent translation history in each request (may reduce throughput)"
    ),
    "translation.max_retries": ConfigurationValue(
        type="number",
        range=[0, 5],
        default=1,
        description="Max retries per chunk for transient API errors"
    ),
    "audio.audio_agent": ConfigurationValue(
        type="select",
        options=["GeminiAudioAgent", "WhisperAudioAgent", "QwenAudioAgent", "GPT4oAudioAgent"],
        default="WhisperAudioAgent",
        description="Audio processing agent for transcription"
    ),
    "audio.VAD_model": ConfigurationValue(
        type="select",
        options=["pyannote/speaker-diarization-3.1", "API"],
        default="API",
        description="Voice Activity Detection model"
    ),
    "audio.src_lang": ConfigurationValue(
        type="select",
        options=["en", "zh", "es", "fr", "de", "ru", "ja", "ar", "kr"],
        default="en",
        description="Source language code for audio processing"
    ),
    "audio.tgt_lang": ConfigurationValue(
        type="select",
        options=["en", "zh", "es", "fr", "de", "ru", "ja", "ar", "kr"],
        default="zh",
        description="Target language code for audio processing"
    ),
    "vision.enable_vision": ConfigurationValue(
        type="boolean",
        default=False,
        description="Enable vision processing for visual content analysis"
    ),
    "vision.vision_model": ConfigurationValue(
        type="select",
        options=["CLIP", "gpt-4o", "gpt-4o-mini"],
        default="gpt-4o",
        description="Vision model for visual content analysis"
    ),
    "vision.frame_per_seg": ConfigurationValue(
        type="number",
        range=[1, 10],
        default=4,
        description="Number of frames extracted per segment for vision analysis"
    ),
    "pre_process.sentence_form": ConfigurationValue(
        type="boolean",
        default=True,
        description="Normalize sentence structure before translation"
    ),
    "pre_process.spell_check": ConfigurationValue(
        type="boolean",
        default=False,
        description="Check and correct spelling errors"
    ),
    "pre_process.term_correct": ConfigurationValue(
        type="boolean",
        default=True,
        description="Apply domain-specific terminology corrections"
    ),
    "post_process.enable_post_process": ConfigurationValue(
        type="boolean",
        default=True,
        description="Enable post-processing module"
    ),
    "post_process.check_len_and_split": ConfigurationValue(
        type="boolean",
        default=True,
        description="Check subtitle length and split if necessary"
    ),
    "post_process.remove_trans_punctuation": ConfigurationValue(
        type="boolean",
        default=True,
        description="Remove translation artifacts and extra punctuation"
    ),
    "proofreader.enable_proofreading": ConfigurationValue(
        type="boolean",
        default=True,
        description="Enable proofreading of translations"
    ),
    "proofreader.window_size": ConfigurationValue(
        type="number",
        range=[1, 20],
        default=5,
        description="Number of sentences per proofreading chunk"
    ),
    "proofreader.short_term_memory_len": ConfigurationValue(
        type="number",
        range=[1, 20],
        default=5,
        description="Maximum number of sentences stored in short term memory"
    ),
    "proofreader.enable_short_term_memory": ConfigurationValue(
        type="boolean",
        default=False,
        description="Whether to use short term memory for proofreading"
    ),
    "proofreader.verbose": ConfigurationValue(
        type="boolean",
        default=True,
        description="Whether to print the proofreading process"
    ),
    "editor.enable_editor": ConfigurationValue(
        type="boolean",
        default=True,
        description="Enable editor module for translation improvement"
    ),
    "editor.user_instruction": ConfigurationValue(
        type="select",
        options=["none", "formal", "casual", "technical"],
        default="none",
        description="Additional instructions for the editor (none means no specific instruction)"
    ),
    "editor.editor_context_window": ConfigurationValue(
        type="number",
        range=[1, 20],
        default=10,
        description="Number of sentences to provide as context for the editor"
    ),
    "editor.history_length": ConfigurationValue(
        type="number",
        range=[1, 20],
        default=5,
        description="Number of sentences to provide as history for the editor"
    ),
    "MEMEORY.enable_local_knowledge": ConfigurationValue(
        type="boolean",
        default=True,
        description="Enable local knowledge base"
    ),
    "MEMEORY.enable_web_search": ConfigurationValue(
        type="boolean",
        default=False,
        description="Enable web search for additional context"
    ),
    "MEMEORY.enable_vision_knowledge": ConfigurationValue(
        type="boolean",
        default=False,
        description="Enable vision-based knowledge extraction"
    ),
    "output_type.video": ConfigurationValue(
        type="boolean",
        default=False,
        description="Generate video with embedded subtitles"
    ),
    "output_type.bilingual": ConfigurationValue(
        type="boolean",
        default=True,
        description="Create bilingual subtitles"
    ),
    "output_type.subtitle": ConfigurationValue(
        type="select",
        options=["srt", "ass"],
        default="srt",
        description="Subtitle file format"
    )
}


# Welcome message for new chat sessions
WELCOME_MESSAGE = """Hello! I'm your ViDove translation assistant. I'll help you configure your video translation task through a friendly conversation.

To get started, you can:
📁 **Upload a file**: Drag and drop a video, audio, or SRT file into the chat, or use the upload button. Here's a [demo video](https://drive.google.com/file/d/1gyaAg2jMRfo8L5zg6FpvBzOHIJIV_r7U/view?usp=sharing) you can try out.
💬 **Tell me your preferences**: What languages do you want to translate between?

You can also ask me about:
- Available languages and models
- Video quality settings
- Output format options
- Processing preferences

What would you like to translate today?"""


# File type mappings
FILE_TYPE_MAPPINGS = {
    'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
    'audio': ['.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg'],
    'srt': ['.srt']
}
