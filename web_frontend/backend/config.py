"""
Configuration schema and constants for the ViDove web backend.
"""
from models import ConfigurationValue


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
    "video_download.resolution": ConfigurationValue(
        type="select",
        options=[360, 480, 720, "best"],
        default=480,
        description="Video resolution for download"
    ),
    "translation.model": ConfigurationValue(
        type="select",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-4o", "Assistant", "Multiagent"],
        default="gpt-4o",
        description="LLM model for translation"
    ),
    "translation.chunk_size": ConfigurationValue(
        type="number",
        range=[100, 5000],
        default=2000,
        description="Text chunk size for translation"
    ),
    "audio.audio_agent": ConfigurationValue(
        type="select",
        options=["GeminiAudioAgent"],
        default="GeminiAudioAgent",
        description="Audio processing agent for transcription"
    ),
    "audio.VAD_model": ConfigurationValue(
        type="select",
        options=["pyannote/speaker-diarization-3.1", "silero"],
        default="pyannote/speaker-diarization-3.1",
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
    "vision.vision_model": ConfigurationValue(
        type="select",
        options=["CLIP", "gpt-4o"],
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
        default=False,
        description="Enable local knowledge base"
    ),
    "MEMEORY.enable_web_search": ConfigurationValue(
        type="boolean",
        default=False,
        description="Enable web search for additional context"
    ),
    "MEMEORY.enable_vision_knowledge": ConfigurationValue(
        type="boolean",
        default=True,
        description="Enable vision-based knowledge extraction"
    ),
    "output_type.video": ConfigurationValue(
        type="boolean",
        default=True,
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
📁 **Upload a file**: Drag and drop a video, audio, or SRT file into the chat, or use the upload button
🎬 **Share a YouTube URL**: Just paste any YouTube link directly in the chat
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
