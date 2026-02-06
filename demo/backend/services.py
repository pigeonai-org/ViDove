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
import signal
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
    
    config_file = temp_dir / "task_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(session_config, f, default_flow_style=False, allow_unicode=True)
    
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
        "num_workers": getattr(session_config, 'num_workers', 8),
        "instructions": getattr(session_config, 'instructions', None),
        "video_download": {
            "resolution": session_config.video_download_resolution
        },
        "translation": {
            "model": session_config.translation_model,
            "chunk_size": session_config.translation_chunk_size,
            "use_history": getattr(session_config, 'translation_use_history', True),
            "max_retries": getattr(session_config, 'translation_max_retries', 1)
        },
        "audio": {
            "enable_audio": True,  # Audio is always enabled
            "audio_agent": session_config.audio_audio_agent,
            "model_path": session_config.audio_model_path,  # Can be None
            "VAD_model": session_config.audio_VAD_model,
            "src_lang": session_config.audio_src_lang,
            "tgt_lang": session_config.audio_tgt_lang
        },
        "vision": {
            "enable_vision": session_config.vision_enable_vision,
            "vision_model": session_config.vision_vision_model,
            "model_path": session_config.vision_model_path,  # Required, has default
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
        
        # Check if ViDove environment exists
        if not VIDOVE_PYTHON.exists():
            raise RuntimeError(f"ViDove Python environment not found at {VIDOVE_PYTHON}. Please ensure ViDove virtual environment is properly set up.")
        
        if not VIDOVE_ENTRY.exists():
            raise RuntimeError(f"ViDove entry script not found at {VIDOVE_ENTRY}. Please ensure ViDove is properly installed.")
       
        
        task_config_file = create_task_config_file(task_config, temp_dir)
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
        
        # Use Popen for better process control during timeout
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()  # Inherit current environment variables
        )
        
        try:
            stdout, stderr = process.communicate(timeout=7200)  # 1 hour timeout
            
            # Check if process succeeded
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, cmd_args, stdout, stderr
                )
            
            print(f"ViDove task {task_id} completed successfully")
            print(f"STDOUT: {stdout}")
            if stderr:
                print(f"STDERR: {stderr}")
                
        except subprocess.TimeoutExpired:
            # Properly terminate the timed-out process
            print(f"Task {task_id} timed out, terminating process...")
            
            # Try to terminate process gracefully
            try:
                process.terminate()
                print(f"Sent SIGTERM to process {process.pid}")
            except (ProcessLookupError, OSError):
                # Process may already be dead
                pass
            
            # Give the process a chance to terminate gracefully
            try:
                process.wait(timeout=10)  # Wait up to 10 seconds for graceful termination
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                print("Process didn't terminate gracefully, force killing...")
                try:
                    process.kill()
                    print(f"Sent SIGKILL to process {process.pid}")
                except (ProcessLookupError, OSError):
                    # Process may already be dead
                    pass
                process.wait()  # Wait for the process to be fully cleaned up
            
            # Re-raise the timeout exception to be handled by the outer try-except
            raise
        
        # Search for result files in all task directories
        result_files = []
        # get the only child directory in local_dump
        local_dump_subdirs = [d for d in local_dump_dir.iterdir() if d.is_dir()]
        if not local_dump_subdirs:
            raise FileNotFoundError(f"No task directories found in {local_dump_dir}. ViDove may not have created any output.")

        task_dir = local_dump_subdirs[0]
        for subdir in task_dir.iterdir():
            if subdir.is_dir() and subdir.name == "results":
                # append all files in the results directory
                result_files.extend(subdir.glob("*"))
            if subdir.is_file() and subdir.name.endswith(".log"):
                result_files.append(subdir)

        permanent_result_path = Path(__file__).parent / "results" / task_id
        if not permanent_result_path.exists():
            permanent_result_path.mkdir(parents=True)
        for result_file in result_files:
            if result_file.is_file():
                destination = permanent_result_path / result_file.name
                shutil.copy(result_file, destination)
                print(f"Copied result file {result_file} to {destination}")

        if task_id in tasks:
            tasks[task_id].status = "COMPLETED"
            if permanent_result_path:
                # Use the first (main) result file as the primary result path
                tasks[task_id].result_path = str(permanent_result_path)
                print(f"Task completed with permanent result: {permanent_result_path}")
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
            tasks[task_id].error = f"{error_msg}\nSTDOUT: {e.stdout or 'No stdout'}\nSTDERR: {e.stderr or 'No stderr'}"
            tasks[task_id].working_directory = None  # Clear working directory reference
        
        print(f"Task {task_id} failed: {error_msg}")
        print(f"STDOUT: {e.stdout or 'No stdout'}")
        print(f"STDERR: {e.stderr or 'No stderr'}")
        
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
            
            # 3. Ensure subprocess is terminated if still running
            if process and process.poll() is None:
                print(f"Terminating remaining subprocess for task {task_id}")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                    try:
                        process.kill()
                        process.wait()
                    except (ProcessLookupError, OSError):
                        pass  # Process already terminated
                except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                    try:
                        # Force kill the entire process group
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
                    except (ProcessLookupError, OSError):
                        pass  # Process group already terminated
            
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
    results_dir = Path(__file__).parent / "results" / task_id
    result_files = []
    
    if results_dir.exists():
        # Look for all files with the task_id in the filename
        for file_path in results_dir.glob(f"*"):
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
    return task_id in tasks 
