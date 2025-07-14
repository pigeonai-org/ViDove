"""
FastAPI route handlers for the ViDove web backend.
"""
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from fastapi import HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse

from models import (
    ChatMessage, SessionConfig, ConfigurationSession, TaskStatus, TaskInfo,
    StartSessionResponse, SendMessageResponse, ConfigResponse, TaskRequest,
    CreateTaskResponse, UploadFileResponse, YouTubeUrlRequest, YouTubeUrlResponse, ErrorResponse,
    TaskResultResponse, FileDownloadInfo, AgentConversationResponse, AgentConversationMessage
)
from config import WELCOME_MESSAGE
from services import (
    generate_llm_response, convert_web_config_to_task_config, run_vidove_task, determine_file_type,
    get_task_result_info, validate_task_access, get_result_file_path, list_result_files
)


# Task access control functions
def validate_task_ownership(task_id: str, session_id: str) -> bool:
    """Validate that a task belongs to the given session"""
    if task_id not in tasks:
        return False
    return tasks[task_id].session_id == session_id


def get_task_with_access_control(task_id: str, session_id: str) -> TaskStatus:
    """Get task status with access control"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if not validate_task_ownership(task_id, session_id):
        raise HTTPException(status_code=403, detail="Access denied: Task does not belong to this session")
    
    return tasks[task_id]


# Global storage - in production, use proper database
sessions: Dict[str, ConfigurationSession] = {}
tasks: Dict[str, TaskStatus] = {}
running_tasks: Dict[str, threading.Thread] = {}


async def start_chat_session() -> StartSessionResponse:
    """Start a new configuration chat session"""
    session_id = str(uuid.uuid4())
    config = SessionConfig()
    
    session = ConfigurationSession(
        session_id=session_id,
        messages=[],
        current_config=config,
        is_complete=False
    )
    
    sessions[session_id] = session
    
    return StartSessionResponse(
        session_id=session_id,
        message=WELCOME_MESSAGE,
        current_config=config
    )


async def send_message(session_id: str, message: ChatMessage) -> SendMessageResponse:
    """Send a message to the chat session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session.messages.append(message)
    
    # Generate LLM response
    llm_response = await generate_llm_response(session.messages, session.current_config)
    
    # Update configuration
    config_updates = llm_response.get("config_updates", {})
    
    # Apply updates to current config
    for key, value in config_updates.items():
        attr_name = key.replace('.', '_')
        if hasattr(session.current_config, attr_name):
            # Special handling for instructions field
            if key == 'instructions' and isinstance(value, list):
                # Ensure instructions is a list of strings
                processed_instructions = [str(item) for item in value if item]
                setattr(session.current_config, attr_name, processed_instructions)
            else:
                setattr(session.current_config, attr_name, value)
        else:
            print(f"Warning: Unknown configuration field: {key}")
            # Skip unknown configuration fields
            pass
    
    session.is_complete = llm_response.get("is_complete", False)
    
    # Add assistant response to session
    assistant_message = ChatMessage(
        role="assistant",
        content=llm_response["message"],
        timestamp=datetime.now().isoformat()
    )
    session.messages.append(assistant_message)
    
    # Note: Removed auto-task creation logic - user will decide when to launch task
    response_data = {
        "message": llm_response["message"],
        "config_updates": config_updates,
        "current_config": session.current_config,
        "is_complete": session.is_complete
    }
    
    return SendMessageResponse(**response_data)


async def get_current_config(session_id: str) -> ConfigResponse:
    """Get current configuration for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return ConfigResponse(
        config=session.current_config,
        is_complete=session.is_complete
    )


async def update_config(session_id: str, config_updates: Dict[str, Any]) -> ConfigResponse:
    """Update configuration for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Define restricted fields that cannot be modified by users
    restricted_fields = {
        'uploaded_file_path',
        'uploaded_file_name', 
        'youtube_url',
        'input_type',
        'audio.audio_agent',  # System-managed
        # Add more restricted fields as needed
    }
    
    # Validate and apply updates
    for key, value in config_updates.items():
        # Check if field is restricted
        if key in restricted_fields:
            raise HTTPException(
                status_code=403, 
                detail=f"Field '{key}' is read-only and cannot be modified"
            )
        
        # Validate against schema if needed
        from config import CONFIGURATION_SCHEMA
        if key in CONFIGURATION_SCHEMA:
            schema_field = CONFIGURATION_SCHEMA[key]
            
            # Type validation
            if schema_field.type == "number" and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Field '{key}' must be a number"
                    )
            
            if schema_field.type == "boolean" and not isinstance(value, bool):
                # Handle string to boolean conversion
                if isinstance(value, str):
                    if value.lower() in ('true', '1', 'yes'):
                        value = True
                    elif value.lower() in ('false', '0', 'no'):
                        value = False
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Field '{key}' must be a boolean"
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Field '{key}' must be a boolean"
                    )
            
            if schema_field.type == "select" and schema_field.options:
                valid_values = [opt for opt in schema_field.options]
                if value not in valid_values:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Field '{key}' must be one of: {valid_values}"
                    )
            
            # Range validation for numbers
            if schema_field.type == "number" and schema_field.range:
                min_val, max_val = schema_field.range
                if value < min_val or value > max_val:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Field '{key}' must be between {min_val} and {max_val}"
                    )
        
        # Apply the update - convert dot notation to underscore
        attr_name = key.replace('.', '_')
        if hasattr(session.current_config, attr_name):
            # Type conversion for specific fields
            current_attr = getattr(session.current_config, attr_name)
            if isinstance(current_attr, int) and isinstance(value, str) and value.isdigit():
                value = int(value)
            elif isinstance(current_attr, float) and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            setattr(session.current_config, attr_name, value)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown configuration field: {key}"
            )
    
    return ConfigResponse(
        config=session.current_config,
        is_complete=session.is_complete
    )


async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks) -> CreateTaskResponse:
    """Create a new translation task"""
    if task_request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[task_request.session_id]
    if not session.is_complete:
        raise HTTPException(status_code=400, detail="Configuration is not complete. Please finish the conversation first.")
    
    task_id = str(uuid.uuid4())
    
    # Create task status
    task_status = TaskStatus(
        task_id=task_id,
        session_id=task_request.session_id,
        status="CREATED",
        input_type=task_request.input_type
    )
    tasks[task_id] = task_status
    
    # Use uploaded file info if available, otherwise use provided input data
    input_data = task_request.input_data
    input_type = task_request.input_type
    
    # Check if there's an uploaded file for this session
    if session.current_config.uploaded_file_path and not input_data.strip():
        input_data = session.current_config.uploaded_file_path
        if session.current_config.input_type:
            input_type = session.current_config.input_type
    
    if not input_data.strip():
        raise HTTPException(status_code=400, detail="No input data provided. Please upload a file or provide a URL.")
    
    # Convert web config to ViDove task config
    task_config = convert_web_config_to_task_config(session.current_config)
    
    # Start task execution in background
    thread = threading.Thread(
        target=run_vidove_task,
        args=(task_id, task_config, input_type, input_data, tasks, running_tasks)
    )
    thread.start()
    running_tasks[task_id] = thread
    
    return CreateTaskResponse(
        task_id=task_id,
        status="CREATED",
        message=f"Translation task created and started for {input_type} input"
    )


async def launch_task_from_session(session_id: str) -> CreateTaskResponse:
    """Launch a translation task from a completed configuration session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Check if we have enough information to launch
    if not session.current_config.source_lang or not session.current_config.target_lang:
        raise HTTPException(status_code=400, detail="Incomplete configuration. Please specify at least source and target languages.")
    
    # Check for input data - either file upload or YouTube URL
    input_data = None
    input_type: Optional[Literal["youtube", "video", "audio", "srt"]] = None
    
    if session.current_config.youtube_url:
        # YouTube URL is provided
        input_data = session.current_config.youtube_url
        input_type = "youtube"
    elif session.current_config.uploaded_file_path:
        # Uploaded file is provided
        input_data = session.current_config.uploaded_file_path
        input_type = session.current_config.input_type
    
    if not input_data:
        raise HTTPException(status_code=400, detail="No input provided. Please upload a file or provide a YouTube URL before launching the task.")
    
    if not input_type:
        raise HTTPException(status_code=400, detail="Unable to determine input type. Please re-upload the file or re-submit the YouTube URL.")
    
    # Create new task
    task_id = str(uuid.uuid4())
    
    # Create task status
    task_status = TaskStatus(
        task_id=task_id,
        session_id=session_id,
        status="CREATED",
        input_type=input_type
    )
    tasks[task_id] = task_status
    
    # Convert web config to ViDove task config
    task_config = convert_web_config_to_task_config(session.current_config)
    
    # Start task execution in background
    thread = threading.Thread(
        target=run_vidove_task,
        args=(task_id, task_config, input_type, input_data, tasks, running_tasks)
    )
    thread.start()
    running_tasks[task_id] = thread
    
    return CreateTaskResponse(
        task_id=task_id,
        status="CREATED",
        message=f"Translation task launched successfully for {input_type} input"
    )


async def get_task_status(task_id: str, session_id: str) -> TaskStatus:
    """Get status of a translation task with access control"""
    return get_task_with_access_control(task_id, session_id)


async def list_tasks(session_id: str) -> List[TaskInfo]:
    """List translation tasks for a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    task_list = []
    for task_id, task_status in tasks.items():
        # Only include tasks that belong to this session
        if task_status.session_id == session_id:
            task_info = TaskInfo(
                task_id=task_id,
                status=task_status.status,
                created_at=task_status.created_at.isoformat(),
                input_type=task_status.input_type
            )
            task_list.append(task_info)
    
    return task_list


async def cancel_task(task_id: str, session_id: str) -> Dict[str, str]:
    """Cancel a running task with access control"""
    task_status = get_task_with_access_control(task_id, session_id)
    
    # Update task status to failed/cancelled
    if task_status.status == "RUNNING":
        task_status.status = "FAILED"
        task_status.error = "Task cancelled by user"
        
        # Try to terminate the thread (note: this is not guaranteed to work)
        if task_id in running_tasks:
            thread = running_tasks[task_id]
            # We can't actually kill a thread in Python, but we mark it as cancelled
            # The task should check for cancellation periodically
    
    return {"message": f"Task {task_id} cancellation requested"}


async def upload_file(session_id: str, file: UploadFile = File(...)) -> UploadFileResponse:
    """Upload a file for translation and associate it with a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create uploads directory relative to current working directory
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(file.filename or "").suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_dir / unique_filename
    
    # Determine input type based on file extension
    input_type = determine_file_type(file_extension)
    
    # Save file using streaming to handle large files efficiently
    total_size = 0
    chunk_size = 8192  # 8KB chunks
    
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total_size += len(chunk)
                
                # Optional: Add a reasonable file size limit (e.g., 5GB)
                # You can adjust this based on your needs
                if total_size > 5 * 1024 * 1024 * 1024:  # 5GB limit
                    # Clean up partial file
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413, 
                        detail="File too large. Maximum file size is 5GB."
                    )
                    
    except Exception as e:
        # Clean up partial file on error
        file_path.unlink(missing_ok=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, 
            detail=f"Error uploading file: {str(e)}"
        )
    
    # Update session configuration with uploaded file info
    session = sessions[session_id]
    session.current_config.uploaded_file_path = str(file_path)
    session.current_config.uploaded_file_name = file.filename or unique_filename
    if input_type:
        session.current_config.input_type = input_type
    
    return UploadFileResponse(
        filename=file.filename or unique_filename,
        file_path=str(file_path),
        size=total_size
    )


async def submit_youtube_url(session_id: str, request: YouTubeUrlRequest) -> YouTubeUrlResponse:
    """Submit a YouTube URL for translation and associate it with a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Validate YouTube URL (basic validation)
    youtube_url = request.youtube_url.strip()
    if not youtube_url:
        raise HTTPException(status_code=400, detail="YouTube URL cannot be empty")
    
    # Basic YouTube URL validation
    valid_youtube_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
    if not any(domain in youtube_url.lower() for domain in valid_youtube_domains):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
    
    # Clear any previously uploaded file data since we're using YouTube now
    session.current_config.uploaded_file_path = None
    session.current_config.uploaded_file_name = None
    
    # Set YouTube URL and input type
    session.current_config.youtube_url = youtube_url
    session.current_config.input_type = "youtube"
    
    return YouTubeUrlResponse(
        youtube_url=youtube_url,
        message=f"YouTube URL successfully set for translation: {youtube_url}"
    )


async def get_task_results(task_id: str, session_id: str) -> TaskResultResponse:
    """Get information about task result files with access control"""
    task_status = get_task_with_access_control(task_id, session_id)
    
    # Get result file information
    result_info = get_task_result_info(task_id)
    
    # Convert file info to response format
    files = []
    for file_info in result_info["files"]:
        files.append(FileDownloadInfo(
            filename=file_info["filename"],
            file_type=file_info["file_type"],
            size_bytes=file_info["size_bytes"],
            created_at=file_info["created_at"]
        ))
    
    return TaskResultResponse(
        task_id=task_id,
        status=task_status.status,
        has_results=result_info["has_results"],
        files=files,
        error=task_status.error
    )


async def download_task_file(task_id: str, filename: str, session_id: str) -> FileResponse:
    """Download a specific result file for a task with access control"""
    task_status = get_task_with_access_control(task_id, session_id)
    
    # Validate task access
    if not validate_task_access(task_id, tasks):
        raise HTTPException(status_code=400, detail="Task is not completed or accessible")
    
    # Get all result files for this task
    result_files = list_result_files(task_id)
    
    # Find the requested file
    target_file = None
    for file_path in result_files:
        if file_path.name == filename:
            target_file = file_path
            break
    
    if not target_file or not target_file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    media_type = "application/octet-stream"
    if target_file.suffix.lower() == ".srt":
        media_type = "text/plain"
    elif target_file.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
        media_type = f"video/{target_file.suffix[1:]}"
    
    return FileResponse(
        path=str(target_file),
        filename=filename,
        media_type=media_type
    )


async def download_primary_result(task_id: str, session_id: str) -> FileResponse:
    """Download the primary result file (usually subtitle) for a task with access control"""
    task_status = get_task_with_access_control(task_id, session_id)
    
    # Validate task access
    if not validate_task_access(task_id, tasks):
        raise HTTPException(status_code=400, detail="Task is not completed or accessible")
    
    # Get the primary result file
    primary_file = get_result_file_path(task_id)
    
    if not primary_file or not primary_file.exists():
        raise HTTPException(status_code=404, detail="No result file found")
    
    # Determine media type
    media_type = "application/octet-stream"
    if primary_file.suffix.lower() == ".srt":
        media_type = "text/plain"
    elif primary_file.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
        media_type = f"video/{primary_file.suffix[1:]}"
    
    return FileResponse(
        path=str(primary_file),
        filename=primary_file.name,
        media_type=media_type
    )


async def get_task_conversation(task_id: str, session_id: str) -> AgentConversationResponse:
    """Get agent conversation history for a task"""
    task_status = get_task_with_access_control(task_id, session_id)
    
    conversation = []
    is_live = False
    
    # If task has saved conversation history, return it
    if task_status.agent_conversation:
        conversation = task_status.agent_conversation
        is_live = False
    else:
        # Try to read live conversation from jsonl file
        conversation = read_live_conversation(task_id)
        is_live = len(conversation) > 0
    
    response = AgentConversationResponse(
        task_id=task_id,
        conversation=conversation,
        is_live=is_live
    )
    return response


def read_live_conversation(task_id: str) -> List[AgentConversationMessage]:
    """Read live conversation from the agent_history.jsonl file in task working directory"""
    import json
    from pathlib import Path
    
    conversation = []
    
    # Get the task's working directory from task status
    if task_id not in tasks:
        return conversation
    
    task_status = tasks[task_id]
    working_dir = task_status.working_directory
    
    if not working_dir:
        return conversation
    
    working_path = Path(working_dir)
    
    if not working_path.exists():
        return conversation
    
    # The agent_history.jsonl should be in the working directory root or 
    # in the local_dump/task_* subdirectories created by ViDove
    agent_history_locations = [
        working_path / "agent_history.jsonl",  # Direct in working directory
    ]
    
    # Also check in local_dump subdirectories
    local_dump_dir = working_path / "local_dump"
    if local_dump_dir.exists():
        for task_subdir in local_dump_dir.glob("task_*"):
            if task_subdir.is_dir():
                agent_history_locations.append(task_subdir / "agent_history.jsonl")
    
    # Try each possible location
    for agent_history_file in agent_history_locations:
        if agent_history_file.exists():
            try:
                with open(agent_history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            conversation.append(AgentConversationMessage(
                                role=data.get("role", "unknown"),
                                message=data.get("message", ""),
                                timestamp=data.get("timestamp")
                            ))
                break  # Found the file, stop looking
            except Exception as e:
                print(f"Error reading agent history from {agent_history_file}: {e}")
        else:
            pass
    
    return conversation


def save_conversation_on_completion(task_id: str) -> None:
    """Save conversation to task status when task completes"""
    if task_id in tasks:
        conversation = read_live_conversation(task_id)
        if conversation:
            tasks[task_id].agent_conversation = conversation


def cleanup_resources():
    """Cleanup function for application shutdown"""
    
    # Cancel all running tasks
    for task_id, thread in running_tasks.items():
        if thread.is_alive():
            if task_id in tasks:
                tasks[task_id].status = "FAILED"
                tasks[task_id].error = "Application shutdown"
    
    # Wait for threads to complete (with timeout)
    import time
    start_time = time.time()
    timeout = 10  # 10 seconds timeout
    
    for task_id, thread in running_tasks.items():
        remaining_time = max(0, timeout - (time.time() - start_time))
        if remaining_time > 0 and thread.is_alive():
            thread.join(timeout=remaining_time)
            
        if thread.is_alive():
            print(f"Warning: Task {task_id} did not complete within timeout")
    
    print("ViDove web backend shutdown complete.")