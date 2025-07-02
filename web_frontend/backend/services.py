"""
Business logic and services for the ViDove web backend.
"""
import os
import json
import threading
import traceback
import subprocess
import tempfile
import yaml
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, cast, List as ListType
from uuid import uuid4

from openai import AsyncOpenAI

from models import ChatMessage, SessionConfig, TaskStatus
from config import CONFIGURATION_SCHEMA


# OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ViDove configuration
VIDOVE_ROOT = Path(__file__).parent.parent.parent
VIDOVE_PYTHON = VIDOVE_ROOT / "venv" / "bin" / "python"  # Adjust path to ViDove's venv
VIDOVE_ENTRY = VIDOVE_ROOT / "entries" / "run.py"


async def generate_llm_response(
    messages: List[ChatMessage], 
    current_config: SessionConfig
) -> Dict[str, Any]:
    """Generate LLM response with type safety"""
    try:
        uploaded_file_info = ""
        if current_config.uploaded_file_path and current_config.uploaded_file_name:
            uploaded_file_info = f"""

UPLOADED FILE DETECTED:
- File: {current_config.uploaded_file_name}
- Type: {current_config.input_type or 'Unknown'}
- Path: {current_config.uploaded_file_path}

The user has already uploaded a file for translation. You can reference this in your responses."""

        # Prepare current configuration and schema for the LLM
        current_config_json = json.dumps(current_config.dict(), indent=2)
        available_options_json = json.dumps({k: v.dict() for k, v in CONFIGURATION_SCHEMA.items()}, indent=2)
        
        system_prompt = f"""You are a helpful video translation configuration assistant. Your job is to help users configure their video translation settings through natural conversation.

CURRENT CONFIGURATION:
{current_config_json}

AVAILABLE OPTIONS AND CONSTRAINTS:
{available_options_json}

{uploaded_file_info}

INSTRUCTIONS:
1. Help users understand and configure their translation settings
2. Extract configuration updates from user messages based on the available options
3. Provide friendly, helpful responses explaining the changes
4. Use the exact option values from the schema (e.g., "EN", "ZH" for languages)
5. Do NOT determine when configuration is complete - the user will decide when to launch
6. If the user provides a YouTube URL (containing youtube.com, youtu.be, etc.), set youtube_url in config_updates
7. When a YouTube URL is detected, also set input_type to "youtube" and clear any uploaded file references
8. When user provides custom instructions, set them in the configuration under the 'instructions' field as an array

RESPONSE FORMAT - Always respond with ONLY valid JSON:
{{
  "message": "Your friendly response to the user",
  "config_updates": {{
    "key": "value"
  }}
}}

EXAMPLES:
- User: "translate from english to chinese" -> {{"source_lang": "EN", "target_lang": "ZH"}}
- User: "use gpt-4 model" -> {{"translation.model": "gpt-4"}}
- User: "set resolution to 720p" -> {{"video_download.resolution": 720}}
- User: "https://www.youtube.com/watch?v=abc123" -> {{"youtube_url": "https://www.youtube.com/watch?v=abc123", "input_type": "youtube", "uploaded_file_path": null, "uploaded_file_name": null}}
- User: "enable bilingual subtitles" -> {{"output_type.bilingual": true}}
- User: "I want formal tone, avoid slang, use technical terms" -> {{"instructions": ["Use formal tone", "Avoid slang", "Use technical terms"]}}
- User: "please translate gaming terms accurately" -> {{"instructions": ["Translate gaming terms accurately"]}}

Remember: NO markdown, NO code blocks, ONLY the JSON object."""

        # Type-safe message formatting for OpenAI API
        message_history: ListType[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        message_history.extend([{"role": msg.role, "content": msg.content} for msg in messages])

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=cast(ListType, message_history),
            temperature=0.7,
            max_tokens=1000,
        )

        content = response.choices[0].message.content
        
        if content is None:
            return {
                "message": "I apologize, but I didn't receive a proper response. Please try again.",
                "config_updates": {},
                "is_complete": False
            }
        
        try:
            # Clean up the content - remove markdown code blocks if present
            clean_content = content.strip()
            if clean_content.startswith("```"):
                clean_content = clean_content[3:]
                if clean_content.startswith("json"):
                    clean_content = clean_content[4:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()
            
            parsed_response: Dict[str, Any] = json.loads(clean_content)
            
            return {
                "message": parsed_response.get("message", content),
                "config_updates": parsed_response.get("config_updates", {}),
                "is_complete": False  # Remove automatic completion logic - user decides when to launch
            }
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {content}")
            # Just return the content as message if JSON parsing fails
            return {
                "message": content,
                "config_updates": {},
                "is_complete": False
            }

    except Exception as e:
        return {
            "message": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
            "config_updates": {},
            "is_complete": False
        }


def create_task_config_file(session_config: Any, temp_dir: Path) -> Path:
    """Create a temporary task configuration file for ViDove"""
    task_config = {
        "source_lang": session_config.source_lang,
        "target_lang": session_config.target_lang,
        "domain": session_config.domain,
        "instructions": getattr(session_config, 'instructions', None),
        "video_download": {
            "resolution": session_config.video_download_resolution
        },
        "translation": {
            "model": session_config.translation_model,
            "chunk_size": session_config.translation_chunk_size
        },
        "audio": {
            "enable_audio": True,
            "audio_agent": session_config.audio_audio_agent,
            "model_path": None,
            "VAD_model": session_config.audio_VAD_model,
            "src_lang": session_config.audio_src_lang,
            "tgt_lang": session_config.audio_tgt_lang
        },
        "vision": {
            "enable_vision": True,
            "vision_model": session_config.vision_vision_model,
            "model_path": None,
            "frame_cache_dir": ".cache/frames",
            "frame_per_seg": session_config.vision_frame_per_seg
        },
        "MEMEORY": {
            "enable_local_knowledge": session_config.MEMEORY_enable_local_knowledge,
            "enable_web_search": session_config.MEMEORY_enable_web_search,
            "enable_vision_knowledge": session_config.MEMEORY_enable_vision_knowledge,
            "local_knowledge_path": "./domain_dict"
        },
        "pre_process": {
            "sentence_form": session_config.pre_process_sentence_form,
            "spell_check": session_config.pre_process_spell_check,
            "term_correct": session_config.pre_process_term_correct
        },
        "post_process": {
            "enable_post_process": session_config.post_process_enable_post_process,
            "check_len_and_split": session_config.post_process_check_len_and_split,
            "remove_trans_punctuation": session_config.post_process_remove_trans_punctuation
        },
        "proofreader": {
            "enable_proofreading": session_config.proofreader_enable_proofreading,
            "window_size": session_config.proofreader_window_size,
            "short_term_memory_len": session_config.proofreader_short_term_memory_len,
            "enable_short_term_memory": session_config.proofreader_enable_short_term_memory,
            "verbose": session_config.proofreader_verbose
        },
        "editor": {
            "enable_editor": session_config.editor_enable_editor,
            "editor_context_window": session_config.editor_editor_context_window,
            "history_length": session_config.editor_history_length
        },
        "output_type": {
            "subtitle": session_config.output_type_subtitle,
            "video": session_config.output_type_video,
            "bilingual": session_config.output_type_bilingual
        },
        "api_source": "openai"
    }
    
    config_file = temp_dir / "task_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(task_config, f, default_flow_style=False, allow_unicode=True)
    
    return config_file


def create_launch_config_file(temp_dir: Path, task_id: str) -> Path:
    """Create a temporary launch configuration file for ViDove"""
    launch_config = {
        "local_dump": str(temp_dir / "local_dump"),
        "environ": "local",
        "api_source": "openai"
    }
    
    config_file = temp_dir / "launch_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(launch_config, f, default_flow_style=False)
    
    return config_file


def convert_web_config_to_task_config(session_config: SessionConfig) -> Dict[str, Any]:
    """Convert web interface configuration to ViDove task configuration format"""
    return {
        "source_lang": session_config.source_lang,
        "target_lang": session_config.target_lang,
        "domain": session_config.domain,
        "instructions": getattr(session_config, 'instructions', None),
        "video_download": {
            "resolution": session_config.video_download_resolution
        },
        "translation": {
            "model": session_config.translation_model,
            "chunk_size": session_config.translation_chunk_size
        },
        "audio": {
            "enable_audio": True,
            "audio_agent": session_config.audio_audio_agent,
            "model_path": None,
            "VAD_model": session_config.audio_VAD_model,
            "src_lang": session_config.audio_src_lang,
            "tgt_lang": session_config.audio_tgt_lang
        },
        "vision": {
            "enable_vision": True,
            "vision_model": session_config.vision_vision_model,
            "model_path": None,
            "frame_cache_dir": ".cache/frames",
            "frame_per_seg": session_config.vision_frame_per_seg
        },
        "MEMEORY": {
            "enable_local_knowledge": session_config.MEMEORY_enable_local_knowledge,
            "enable_web_search": session_config.MEMEORY_enable_web_search,
            "enable_vision_knowledge": session_config.MEMEORY_enable_vision_knowledge,
            "local_knowledge_path": "./domain_dict"
        },
        "pre_process": {
            "sentence_form": session_config.pre_process_sentence_form,
            "spell_check": session_config.pre_process_spell_check,
            "term_correct": session_config.pre_process_term_correct
        },
        "post_process": {
            "enable_post_process": session_config.post_process_enable_post_process,
            "check_len_and_split": session_config.post_process_check_len_and_split,
            "remove_trans_punctuation": session_config.post_process_remove_trans_punctuation
        },
        "proofreader": {
            "enable_proofreading": session_config.proofreader_enable_proofreading,
            "window_size": session_config.proofreader_window_size,
            "short_term_memory_len": session_config.proofreader_short_term_memory_len,
            "enable_short_term_memory": session_config.proofreader_enable_short_term_memory,
            "verbose": session_config.proofreader_verbose
        },
        "editor": {
            "enable_editor": session_config.editor_enable_editor,
            "editor_context_window": session_config.editor_editor_context_window,
            "history_length": session_config.editor_history_length
        },
        "output_type": {
            "subtitle": session_config.output_type_subtitle,
            "video": session_config.output_type_video,
            "bilingual": session_config.output_type_bilingual
        },
        "api_source": "openai"
    }


def run_vidove_task(
    task_id: str, 
    task_config: Dict[str, Any], 
    input_type: str, 
    input_data: str,
    tasks: Dict[str, TaskStatus],
    running_tasks: Dict[str, threading.Thread]
) -> None:
    """Run the actual ViDove task via subprocess with proper resource management"""
    temp_dir = None
    process = None
    original_cwd = os.getcwd()  # Store original working directory at start
    
    try:
        # Update task status to running
        if task_id in tasks:
            tasks[task_id].status = "RUNNING"
        
        # Create temporary directory for this task
        temp_dir = Path(tempfile.mkdtemp(prefix=f"vidove_task_{task_id}_"))
        
        # Store the working directory path in task status for conversation access
        if task_id in tasks:
            tasks[task_id].working_directory = str(temp_dir)
        
        # Create local_dump directory structure
        local_dump_dir = temp_dir / "local_dump"
        local_dump_dir.mkdir(parents=True, exist_ok=True)
        
        # Create task-specific directory (ViDove will create its own with internal task_id)
        task_dir = local_dump_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        results_dir = task_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if ViDove environment exists
        if not VIDOVE_PYTHON.exists():
            raise RuntimeError(f"ViDove Python environment not found at {VIDOVE_PYTHON}. Please ensure ViDove virtual environment is properly set up.")
        
        if not VIDOVE_ENTRY.exists():
            raise RuntimeError(f"ViDove entry script not found at {VIDOVE_ENTRY}. Please ensure ViDove is properly installed.")
        
        # Convert task_config to SessionConfig-like object for file generation
        # This is a bit hacky but necessary for the config file generation
        class TempSessionConfig:
            def __init__(self, config_dict):
                self.__dict__.update(config_dict)
                # Set defaults for any missing values
                self.source_lang = config_dict.get("source_lang", "EN")
                self.target_lang = config_dict.get("target_lang", "ZH")
                self.domain = config_dict.get("domain", "General")
                self.instructions = config_dict.get("instructions", None)
                self.video_download_resolution = config_dict.get("video_download", {}).get("resolution", 480)
                self.translation_model = config_dict.get("translation", {}).get("model", "gpt-4o")
                self.translation_chunk_size = config_dict.get("translation", {}).get("chunk_size", 2000)
                self.audio_audio_agent = config_dict.get("audio", {}).get("audio_agent", "GeminiAudioAgent")
                self.audio_model_path = config_dict.get("audio", {}).get("model_path", None)
                self.audio_VAD_model = config_dict.get("audio", {}).get("VAD_model", "pyannote/speaker-diarization-3.1")
                self.audio_src_lang = config_dict.get("audio", {}).get("src_lang", "en")
                self.audio_tgt_lang = config_dict.get("audio", {}).get("tgt_lang", "zh")
                self.vision_vision_model = config_dict.get("vision", {}).get("vision_model", "gpt-4o")
                self.vision_model_path = config_dict.get("vision", {}).get("model_path", "./ViDove/vision_model/clip-vit-base-patch16")
                self.vision_frame_cache_dir = config_dict.get("vision", {}).get("frame_cache_dir", "./cache")
                self.vision_frame_per_seg = config_dict.get("vision", {}).get("frame_per_seg", 4)
                self.pre_process_sentence_form = config_dict.get("pre_process", {}).get("sentence_form", True)
                self.pre_process_spell_check = config_dict.get("pre_process", {}).get("spell_check", False)
                self.pre_process_term_correct = config_dict.get("pre_process", {}).get("term_correct", True)
                self.post_process_enable_post_process = config_dict.get("post_process", {}).get("enable_post_process", True)
                self.post_process_check_len_and_split = config_dict.get("post_process", {}).get("check_len_and_split", True)
                self.post_process_remove_trans_punctuation = config_dict.get("post_process", {}).get("remove_trans_punctuation", False)
                self.proofreader_enable_proofreading = config_dict.get("proofreader", {}).get("enable_proofreading", True)
                self.proofreader_window_size = config_dict.get("proofreader", {}).get("window_size", 5)
                self.proofreader_short_term_memory_len = config_dict.get("proofreader", {}).get("short_term_memory_len", 5)
                self.proofreader_enable_short_term_memory = config_dict.get("proofreader", {}).get("enable_short_term_memory", False)
                self.proofreader_verbose = config_dict.get("proofreader", {}).get("verbose", True)
                self.editor_enable_editor = config_dict.get("editor", {}).get("enable_editor", True)
                self.editor_user_instruction = config_dict.get("editor", {}).get("user_instruction", "none")
                self.editor_editor_context_window = config_dict.get("editor", {}).get("editor_context_window", 10)
                self.editor_history_length = config_dict.get("editor", {}).get("history_length", 5)
                self.MEMEORY_enable_local_knowledge = config_dict.get("MEMEORY", {}).get("enable_local_knowledge", False)
                self.MEMEORY_local_knowledge_path = config_dict.get("MEMEORY", {}).get("local_knowledge_path", "/home/macrodove/ViDove/domain_dict")
                self.MEMEORY_enable_web_search = config_dict.get("MEMEORY", {}).get("enable_web_search", False)
                self.MEMEORY_enable_vision_knowledge = config_dict.get("MEMEORY", {}).get("enable_vision_knowledge", True)
                self.output_type_video = config_dict.get("output_type", {}).get("video", True)
                self.output_type_bilingual = config_dict.get("output_type", {}).get("bilingual", True)
                self.output_type_subtitle = config_dict.get("output_type", {}).get("subtitle", "srt")
        
        temp_session_config = TempSessionConfig(task_config)
        
        # Create configuration files
        task_config_file = create_task_config_file(temp_session_config, temp_dir)
        launch_config_file = create_launch_config_file(temp_dir, task_id)
        
        # Build command arguments based on input type
        cmd_args = [
            str(VIDOVE_PYTHON),
            str(VIDOVE_ENTRY),
            "--launch_cfg", str(launch_config_file),
            "--task_cfg", str(task_config_file)
        ]
        
        # Convert file paths to absolute paths to ensure they work regardless of working directory
        if input_type == "youtube":
            cmd_args.extend(["--link", input_data])
        elif input_type == "video":
            # Convert to absolute path to ensure it can be found from any working directory
            absolute_video_path = str(Path(input_data).resolve())
            if not Path(absolute_video_path).exists():
                raise FileNotFoundError(f"Video file not found: {input_data}")
            cmd_args.extend(["--video_file", absolute_video_path])
        elif input_type == "audio":
            # Convert to absolute path to ensure it can be found from any working directory
            absolute_audio_path = str(Path(input_data).resolve())
            if not Path(absolute_audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {input_data}")
            cmd_args.extend(["--audio_file", absolute_audio_path])
        elif input_type == "srt":
            # Convert to absolute path to ensure it can be found from any working directory
            absolute_srt_path = str(Path(input_data).resolve())
            if not Path(absolute_srt_path).exists():
                raise FileNotFoundError(f"SRT file not found: {input_data}")
            cmd_args.extend(["--srt_file", absolute_srt_path])
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        # Change to ViDove root directory for execution
        print(f"Changing working directory from {original_cwd} to {VIDOVE_ROOT}")
        os.chdir(VIDOVE_ROOT)
        
        # Verify we're in the correct directory
        current_cwd = os.getcwd()
        if current_cwd != str(VIDOVE_ROOT):
            raise RuntimeError(f"Failed to change to ViDove root directory. Expected: {VIDOVE_ROOT}, Current: {current_cwd}")
        print(f"Successfully changed to ViDove root directory: {current_cwd}")
        
        # Run the ViDove command
        print(f"Running ViDove command: {' '.join(cmd_args)}")
        print(f"Working directory for execution: {os.getcwd()}")
        process = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=True,
            env=os.environ.copy()  # Inherit current environment variables
        )
        
        print(f"ViDove task {task_id} completed successfully")
        print(f"STDOUT: {process.stdout}")
        if process.stderr:
            print(f"STDERR: {process.stderr}")
        
        # Give ViDove a moment to finish writing files
        time.sleep(2)
        
        # Find the result files in ViDove's output structure
        target_lang = task_config.get("target_lang", "EN")
        source_lang = task_config.get("source_lang", "EN")
        print(f"Searching for result files. Source: {source_lang}, Target: {target_lang}")
        print(f"Local dump directory: {local_dump_dir}")
        
        # ViDove creates its own internal task_id and creates a directory like task_{internal_id}
        # We need to find all task directories and search for result files in them
        task_directories = []
        for item in local_dump_dir.glob("task_*"):
            if item.is_dir():
                task_directories.append(item)
        
        print(f"Found task directories: {[d.name for d in task_directories]}")
        
        # Search for result files in all task directories
        result_files = []
        all_results_dirs = []
        
        for task_dir in task_directories:
            results_dir = task_dir / "results"
            if results_dir.exists():
                all_results_dirs.append(results_dir)
                print(f"Checking results directory: {results_dir}")
                
                # List all files in the results directory
                print(f"Files in {results_dir.name}:")
                for file in results_dir.glob("*"):
                    print(f"  {file.name} ({'file' if file.is_file() else 'dir'})")
                
                # Look for all .srt files in this results directory
                srt_files = list(results_dir.glob("*.srt"))
                print(f"Found SRT files in {results_dir.name}: {[f.name for f in srt_files]}")
                
                if srt_files:
                    # Sort by modification time, newest first
                    srt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Find the main translation file (should match pattern *_{target_lang}.srt)
                    for srt_file in srt_files:
                        filename = srt_file.name
                        print(f"Examining file: {filename}")
                        
                        # Check if it's the main translation file
                        if filename.endswith(f"_{target_lang}.srt"):
                            result_files.append(srt_file)
                            print(f"Found main translation file: {filename}")
                        
                        # Check if it's a bilingual file
                        elif filename.endswith(f"_{source_lang}_{target_lang}.srt"):
                            result_files.append(srt_file)
                            print(f"Found bilingual file: {filename}")
                    
                    # If no specific pattern match, add all SRT files
                    if not result_files:
                        result_files.extend(srt_files)
                        print(f"No pattern match, added all SRT files: {[f.name for f in srt_files]}")
                
                # Also look for video files if video output is enabled
                if task_config.get("output_type", {}).get("video", False):
                    print("Video output enabled, searching for video files...")
                    for video_ext in ['.mp4', '.avi', '.mkv', '.mov']:
                        video_files = list(results_dir.glob(f"*{video_ext}"))
                        if video_files:
                            result_files.extend(video_files)
                            print(f"Found {video_ext} files: {[f.name for f in video_files]}")
        
        # Look for log files in task directories (one level up from results)
        for task_dir in task_directories:
            log_files = list(task_dir.glob("*.log"))
            if log_files:
                log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                result_files.extend(log_files)
                print(f"Found log files in {task_dir.name}: {[f.name for f in log_files]}")
        
        # If no SRT files found in results directories, check for other subtitle formats
        if not any(f.suffix.lower() == '.srt' for f in result_files):
            for results_dir in all_results_dirs:
                other_files = list(results_dir.glob("*.ass")) + list(results_dir.glob("*.vtt"))
                if other_files:
                    other_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    result_files.extend(other_files)
                    print(f"Added other subtitle files: {[f.name for f in other_files]}")
        
        # Set result_path to the first (main) result file for backward compatibility
        result_path = str(result_files[0]) if result_files else None
        
        print(f"Final result path: {result_path}")
        print(f"All result files to copy: {[str(f) for f in result_files]}")
        
        # Copy result files to permanent location before cleanup
        permanent_result_paths = []
        if result_files:
            try:
                # Create a permanent results directory in the web backend
                permanent_results_dir = Path(__file__).parent / "results"
                permanent_results_dir.mkdir(exist_ok=True)
                print(f"Created permanent results directory: {permanent_results_dir}")
                
                # Copy all result files
                for result_file in result_files:
                    if result_file.exists():
                        # Generate filename for the permanent copy
                        original_filename = result_file.name
                        # Ensure task_id is in the filename for uniqueness
                        if task_id not in original_filename:
                            name_parts = original_filename.rsplit('.', 1)
                            if len(name_parts) == 2:
                                permanent_filename = f"{task_id}_{name_parts[0]}.{name_parts[1]}"
                            else:
                                permanent_filename = f"{task_id}_{original_filename}"
                        else:
                            permanent_filename = original_filename
                        
                        permanent_result_path = permanent_results_dir / permanent_filename
                        
                        # Copy the result file to permanent location
                        shutil.copy2(result_file, permanent_result_path)
                        permanent_result_paths.append(permanent_result_path)
                        print(f"Copied result file from {result_file} to {permanent_result_path}")
                
                # Video files should already be included in result_files if video output was enabled
                # No need for separate video file copying logic here
                
            except Exception as copy_error:
                print(f"Warning: Failed to copy result files to permanent location: {copy_error}")
                print(f"Copy error traceback: {traceback.format_exc()}")
        else:
            print(f"No result files found to copy.")
            # Let's also check if there are ANY files in the task directories
            for task_dir in task_directories:
                results_dir = task_dir / "results"
                if results_dir.exists():
                    all_files_results = list(results_dir.glob("*"))
                    print(f"All files in {task_dir.name}/results: {[str(f) for f in all_files_results]}")
                all_files_task = list(task_dir.glob("*"))
                print(f"All files in {task_dir.name}: {[str(f) for f in all_files_task]}")
        
        # Update task status to completed
        if task_id in tasks:
            tasks[task_id].status = "COMPLETED"
            if permanent_result_paths:
                # Use the first (main) result file as the primary result path
                tasks[task_id].result_path = str(permanent_result_paths[0])
                print(f"Task completed with permanent result: {permanent_result_paths[0]}")
            elif result_path:
                tasks[task_id].result_path = result_path
                print(f"Task completed with original result: {result_path}")
            else:
                tasks[task_id].result_path = "No result file found"
                print("Task completed but no result file found")
            
            # Save agent conversation when task completes
            try:
                # Import here to avoid circular imports
                from endpoints import save_conversation_on_completion
                save_conversation_on_completion(task_id)
            except Exception as conv_error:
                print(f"Warning: Failed to save conversation for task {task_id}: {conv_error}")
            
            # Clear working directory reference after saving conversation
            tasks[task_id].working_directory = None
            
    except subprocess.TimeoutExpired:
        error_msg = f"Task {task_id} timed out after 1 hour"
        if task_id in tasks:
            tasks[task_id].status = "FAILED"
            tasks[task_id].error = error_msg
            tasks[task_id].working_directory = None  # Clear working directory reference
        print(error_msg)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ViDove process failed with return code {e.returncode}"
        if task_id in tasks:
            tasks[task_id].status = "FAILED"
            tasks[task_id].error = f"{error_msg}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            tasks[task_id].working_directory = None  # Clear working directory reference
        
        print(f"Task {task_id} failed: {error_msg}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
    except Exception as e:
        # Update task status to failed with detailed error information
        error_msg = str(e)
        if task_id in tasks:
            tasks[task_id].status = "FAILED"
            tasks[task_id].error = error_msg
            tasks[task_id].working_directory = None  # Clear working directory reference
        
        # Log the error with full traceback
        full_error = traceback.format_exc()
        print(f"Task {task_id} failed: {error_msg}")
        print(f"Full traceback: {full_error}")
            
    finally:
        # Cleanup section
        try:
            # 1. Restore original working directory
            try:
                os.chdir(original_cwd)
                print(f"Restored working directory to: {os.getcwd()}")
            except Exception as cwd_error:
                print(f"Warning: Could not restore working directory to {original_cwd}: {cwd_error}")
            
            # 2. Clean up from running tasks tracking
            if task_id in running_tasks:
                del running_tasks[task_id]
            
            # 3. Process cleanup is handled automatically by subprocess.run()
            # since we're using subprocess.run() with check=True, the process
            # is already completed when we reach this point
            
            # 4. Clean up temporary directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary directory {temp_dir}: {cleanup_error}")
            
        except Exception as cleanup_error:
            print(f"Error during cleanup for task {task_id}: {cleanup_error}")
            # Don't raise here - we don't want cleanup errors to mask the original error


def determine_file_type(file_extension: str) -> Optional[Literal["video", "audio", "srt"]]:
    """Determine input type based on file extension"""
    from config import FILE_TYPE_MAPPINGS
    
    file_extension = file_extension.lower()
    for input_type, extensions in FILE_TYPE_MAPPINGS.items():
        if file_extension in extensions:
            # Type cast since we know input_type is one of the valid literals
            return input_type  # type: ignore
    return None


def get_result_file_path(task_id: str) -> Optional[Path]:
    """Get the path to the result file for a given task ID"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        return None
    
    # Look for files with the task_id in the filename
    for file_path in results_dir.glob(f"{task_id}_*"):
        if file_path.is_file():
            return file_path
    
    return None


def list_result_files(task_id: str) -> List[Path]:
    """List all result files for a given task ID"""
    results_dir = Path(__file__).parent / "results"
    result_files = []
    
    if results_dir.exists():
        # Look for all files with the task_id in the filename
        for file_path in results_dir.glob(f"{task_id}_*"):
            if file_path.is_file():
                result_files.append(file_path)
    
    return result_files


def cleanup_old_result_files(max_age_hours: int = 24) -> None:
    """Clean up result files older than specified hours"""
    
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in results_dir.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    print(f"Cleaned up old result file: {file_path}")
                except Exception as e:
                    print(f"Failed to clean up file {file_path}: {e}")


def get_task_result_info(task_id: str) -> Dict[str, Any]:
    """Get comprehensive information about task results"""
    result_files = list_result_files(task_id)
    
    result_info: Dict[str, Any] = {
        "task_id": task_id,
        "has_results": len(result_files) > 0,
        "files": []
    }
    
    for file_path in result_files:
        file_info = {
            "filename": file_path.name,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "created_at": file_path.stat().st_mtime,
            "file_type": "subtitle" if file_path.suffix.lower() == ".srt" else "video" if file_path.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"] else "log" if file_path.suffix.lower() == ".log" else "unknown"
        }
        result_info["files"].append(file_info)
    
    return result_info


def validate_task_access(task_id: str, tasks: Dict[str, TaskStatus]) -> bool:
    """Validate that a task exists and is accessible"""
    return task_id in tasks and tasks[task_id].status == "COMPLETED"
