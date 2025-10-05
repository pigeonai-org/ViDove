"""
FastAPI route handlers for the ViDove web backend.
"""
import uuid
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime, timedelta
from collections import OrderedDict

from fastapi import HTTPException, UploadFile, File, BackgroundTasks, Request
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
# Using OrderedDict for LRU-style cleanup
sessions: Dict[str, ConfigurationSession] = OrderedDict()
tasks: Dict[str, TaskStatus] = OrderedDict()
running_tasks: Dict[str, threading.Thread] = {}

# Cleanup management
cleanup_lock = threading.Lock()
last_cleanup_time = time.time()

# Task concurrency limiting
MAX_CONCURRENT_TASKS = 3
task_semaphore = threading.Semaphore(MAX_CONCURRENT_TASKS)

# Rate limiting and bot protection
ip_request_tracking: Dict[str, List[float]] = {}  # IP -> [timestamps]
ip_session_tracking: Dict[str, List[str]] = {}  # IP -> [session_ids]
ip_action_tracking: Dict[str, Dict[str, List[float]]] = {}  # IP -> {action -> [timestamps]}
session_task_count: Dict[str, int] = {}  # session_id -> task_count
rate_limit_lock = threading.Lock()
emergency_mode = False


async def start_chat_session(request: Request) -> StartSessionResponse:
    """Start a new configuration chat session with rate limiting"""
    from config import RATE_LIMIT_SESSION_CREATE_PER_HOUR
    
    # Get client IP
    client_ip = get_client_ip(request)
    
    # Check memory pressure
    memory_ok, memory_percent = check_memory_pressure()
    if not memory_ok:
        emergency_cleanup()
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable due to high memory usage ({memory_percent:.1f}%). Please try again in a few minutes."
        )
    
    # Check rate limit for session creation
    if not check_rate_limit(client_ip, "session_create", RATE_LIMIT_SESSION_CREATE_PER_HOUR, 3600):
        raise HTTPException(
            status_code=429,
            detail="Too many session creation requests. Please try again later."
        )
    
    # Check session limit per IP
    if not check_session_limit_per_ip(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Maximum number of sessions reached for your IP. Please close existing sessions first."
        )
    
    session_id = str(uuid.uuid4())
    config = SessionConfig()
    
    session = ConfigurationSession(
        session_id=session_id,
        messages=[],
        current_config=config,
        is_complete=False
    )
    
    sessions[session_id] = session
    register_session_for_ip(client_ip, session_id)
    
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
        'audio.audio_agent',  # System-managed, always WhisperAudioAgent
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
    
    # Start task execution in background with concurrency limiting
    thread = threading.Thread(
        target=run_task_with_semaphore,
        args=(task_id, task_config, input_type, input_data, tasks, running_tasks)
    )
    thread.daemon = True  # Make thread a daemon so it doesn't prevent process shutdown
    thread.start()
    running_tasks[task_id] = thread
    
    return CreateTaskResponse(
        task_id=task_id,
        status="CREATED",
        message=f"Translation task created and started for {input_type} input"
    )


async def launch_task_from_session(session_id: str, request: Request) -> CreateTaskResponse:
    """Launch a translation task from a completed configuration session"""
    from config import RATE_LIMIT_TASK_CREATE_PER_HOUR
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get client IP and check rate limit
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip, "task_create", RATE_LIMIT_TASK_CREATE_PER_HOUR, 3600):
        raise HTTPException(
            status_code=429,
            detail="Too many task creation requests. Please try again later."
        )
    
    # Check task limit per session
    if not check_task_limit_per_session(session_id):
        raise HTTPException(
            status_code=429,
            detail="Maximum number of tasks reached for this session. Please create a new session."
        )
    
    # Check memory pressure
    memory_ok, memory_percent = check_memory_pressure()
    if not memory_ok:
        emergency_cleanup()
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable due to high memory usage ({memory_percent:.1f}%). Please try again in a few minutes."
        )
    
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
    
    # Start task execution in background with concurrency limiting
    thread = threading.Thread(
        target=run_task_with_semaphore,
        args=(task_id, task_config, input_type, input_data, tasks, running_tasks)
    )
    thread.daemon = True  # Make thread a daemon so it doesn't prevent process shutdown
    thread.start()
    running_tasks[task_id] = thread
    
    # Increment task count for session
    increment_session_task_count(session_id)
    
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


async def upload_file(session_id: str, file: UploadFile = File(...), request: Request = None) -> UploadFileResponse:
    """Upload a file for translation and associate it with a session"""
    from config import RATE_LIMIT_FILE_UPLOAD_PER_HOUR
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get client IP and check rate limit
    if request:
        client_ip = get_client_ip(request)
        if not check_rate_limit(client_ip, "file_upload", RATE_LIMIT_FILE_UPLOAD_PER_HOUR, 3600):
            raise HTTPException(
                status_code=429,
                detail="Too many file upload requests. Please try again later."
            )
    
    # Check memory pressure
    memory_ok, memory_percent = check_memory_pressure()
    if not memory_ok:
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable due to high memory usage ({memory_percent:.1f}%). Please try again later."
        )
    
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


def get_client_ip(request) -> str:
    """Extract client IP from request, handling proxies"""
    # Check X-Forwarded-For header (for proxied requests)
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fall back to direct client
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str, action: str, limit: int, window_seconds: int) -> bool:
    """Check if IP has exceeded rate limit for an action"""
    from config import ENABLE_EMERGENCY_PROTECTION
    
    if not ENABLE_EMERGENCY_PROTECTION:
        return True
    
    current_time = time.time()
    cutoff_time = current_time - window_seconds
    
    with rate_limit_lock:
        # Initialize tracking for this IP and action
        if ip not in ip_action_tracking:
            ip_action_tracking[ip] = {}
        if action not in ip_action_tracking[ip]:
            ip_action_tracking[ip][action] = []
        
        # Remove old timestamps
        ip_action_tracking[ip][action] = [
            ts for ts in ip_action_tracking[ip][action] if ts > cutoff_time
        ]
        
        # Check if limit exceeded
        if len(ip_action_tracking[ip][action]) >= limit:
            return False
        
        # Record this action
        ip_action_tracking[ip][action].append(current_time)
        return True


def check_session_limit_per_ip(ip: str) -> bool:
    """Check if IP has exceeded maximum sessions"""
    from config import MAX_SESSIONS_PER_IP, ENABLE_EMERGENCY_PROTECTION
    
    if not ENABLE_EMERGENCY_PROTECTION:
        return True
    
    with rate_limit_lock:
        if ip not in ip_session_tracking:
            ip_session_tracking[ip] = []
        
        # Remove sessions that no longer exist
        ip_session_tracking[ip] = [
            sid for sid in ip_session_tracking[ip] if sid in sessions
        ]
        
        return len(ip_session_tracking[ip]) < MAX_SESSIONS_PER_IP


def register_session_for_ip(ip: str, session_id: str):
    """Register a new session for an IP"""
    with rate_limit_lock:
        if ip not in ip_session_tracking:
            ip_session_tracking[ip] = []
        ip_session_tracking[ip].append(session_id)


def check_task_limit_per_session(session_id: str) -> bool:
    """Check if session has exceeded maximum tasks"""
    from config import MAX_TASKS_PER_SESSION, ENABLE_EMERGENCY_PROTECTION
    
    if not ENABLE_EMERGENCY_PROTECTION:
        return True
    
    task_count = session_task_count.get(session_id, 0)
    return task_count < MAX_TASKS_PER_SESSION


def increment_session_task_count(session_id: str):
    """Increment task count for a session"""
    session_task_count[session_id] = session_task_count.get(session_id, 0) + 1


def check_memory_pressure() -> tuple[bool, float]:
    """Check if system is under memory pressure"""
    from config import (MEMORY_EMERGENCY_THRESHOLD_PERCENT, 
                       MEMORY_WARNING_THRESHOLD_PERCENT,
                       ENABLE_EMERGENCY_PROTECTION)
    
    if not ENABLE_EMERGENCY_PROTECTION:
        return True, 0.0
    
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent >= MEMORY_EMERGENCY_THRESHOLD_PERCENT:
            return False, memory_percent  # Emergency - reject requests
        
        return True, memory_percent
    except ImportError:
        return True, 0.0


def emergency_cleanup():
    """Perform aggressive cleanup when memory pressure is high"""
    print("⚠️ EMERGENCY CLEANUP TRIGGERED - High memory pressure detected")
    
    # More aggressive cleanup - remove half of oldest sessions
    sessions_to_remove = len(sessions) // 2
    cleaned_sessions = 0
    
    with cleanup_lock:
        for _ in range(sessions_to_remove):
            if len(sessions) > 0:
                session_id, _ = sessions.popitem(last=False)
                cleaned_sessions += 1
    
    # Remove half of completed tasks
    completed_tasks = [(tid, t) for tid, t in tasks.items() 
                      if t.status in ["COMPLETED", "FAILED"]]
    tasks_to_remove = len(completed_tasks) // 2
    cleaned_tasks = 0
    
    with cleanup_lock:
        for task_id, _ in completed_tasks[:tasks_to_remove]:
            if task_id in tasks:
                tasks[task_id].agent_conversation = None
                del tasks[task_id]
                cleaned_tasks += 1
    
    print(f"Emergency cleanup completed: {cleaned_sessions} sessions, {cleaned_tasks} tasks removed")
    print(f"Remaining: {len(sessions)} sessions, {len(tasks)} tasks")


def cleanup_old_sessions(max_age_hours: int = 24, max_sessions: int = 1000) -> int:
    """Remove sessions older than max_age_hours or keep only max_sessions most recent"""
    cleaned = 0
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=max_age_hours)
    
    with cleanup_lock:
        # Remove old sessions
        sessions_to_remove = []
        for session_id, session in list(sessions.items()):
            if session.created_at < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del sessions[session_id]
            cleaned += 1
            print(f"Cleaned up old session: {session_id}")
        
        # If still too many, remove oldest
        while len(sessions) > max_sessions:
            session_id, _ = sessions.popitem(last=False)  # Remove oldest
            cleaned += 1
            print(f"Cleaned up excess session: {session_id}")
    
    return cleaned


def cleanup_old_tasks(max_age_hours: int = 48, max_tasks: int = 500) -> int:
    """Remove completed/failed tasks older than max_age_hours"""
    cleaned = 0
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=max_age_hours)
    
    with cleanup_lock:
        tasks_to_remove = []
        for task_id, task in list(tasks.items()):
            # Only clean up completed or failed tasks
            if task.status in ["COMPLETED", "FAILED"]:
                if task.created_at < cutoff_time:
                    # Clear agent conversation to free memory
                    task.agent_conversation = None
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del tasks[task_id]
            # Also remove from running_tasks if present
            if task_id in running_tasks:
                del running_tasks[task_id]
            cleaned += 1
            print(f"Cleaned up old task: {task_id}")
        
        # If still too many completed tasks, remove oldest completed ones
        completed_tasks = [(tid, t) for tid, t in tasks.items() 
                          if t.status in ["COMPLETED", "FAILED"]]
        if len(completed_tasks) > max_tasks:
            # Sort by creation time
            completed_tasks.sort(key=lambda x: x[1].created_at)
            excess_count = len(completed_tasks) - max_tasks
            for task_id, _ in completed_tasks[:excess_count]:
                del tasks[task_id]
                if task_id in running_tasks:
                    del running_tasks[task_id]
                cleaned += 1
                print(f"Cleaned up excess task: {task_id}")
    
    return cleaned


def periodic_cleanup():
    """Perform periodic cleanup - called from background thread"""
    global last_cleanup_time
    
    from config import (SESSION_CLEANUP_INTERVAL_SECONDS, SESSION_MAX_AGE_HOURS,
                       TASK_MAX_AGE_HOURS, MAX_SESSIONS_IN_MEMORY, MAX_TASKS_IN_MEMORY)
    
    current_time = time.time()
    if current_time - last_cleanup_time < SESSION_CLEANUP_INTERVAL_SECONDS:
        return
    
    last_cleanup_time = current_time
    
    print("Starting periodic cleanup...")
    sessions_cleaned = cleanup_old_sessions(
        max_age_hours=SESSION_MAX_AGE_HOURS,
        max_sessions=MAX_SESSIONS_IN_MEMORY
    )
    tasks_cleaned = cleanup_old_tasks(
        max_age_hours=TASK_MAX_AGE_HOURS,
        max_tasks=MAX_TASKS_IN_MEMORY
    )
    
    # Call the existing cleanup function for result files
    from services import cleanup_old_result_files
    cleanup_old_result_files(max_age_hours=TASK_MAX_AGE_HOURS)
    
    print(f"Cleanup completed: {sessions_cleaned} sessions, {tasks_cleaned} tasks")
    print(f"Current state: {len(sessions)} sessions, {len(tasks)} tasks in memory")


cleanup_thread = None

def start_cleanup_thread():
    """Start background cleanup thread"""
    global cleanup_thread
    
    from config import SESSION_CLEANUP_INTERVAL_SECONDS
    
    def cleanup_loop():
        while True:
            try:
                time.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
                periodic_cleanup()
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("Started cleanup thread")


def cleanup_orphaned_temp_dirs():
    """Clean up temporary directories left from crashed processes"""
    import tempfile
    import shutil
    import glob
    
    temp_dir = Path(tempfile.gettempdir())
    pattern = str(temp_dir / "vidove_task_*")
    
    cleaned = 0
    for temp_path in glob.glob(pattern):
        try:
            shutil.rmtree(temp_path)
            cleaned += 1
            print(f"Cleaned orphaned temp directory: {temp_path}")
        except Exception as e:
            print(f"Failed to clean {temp_path}: {e}")
    
    print(f"Startup cleanup: removed {cleaned} orphaned temp directories")


def run_task_with_semaphore(task_id, task_config, input_type, input_data, tasks, running_tasks):
    """Wrapper to run task with semaphore for concurrency limiting"""
    with task_semaphore:
        from services import run_vidove_task
        run_vidove_task(task_id, task_config, input_type, input_data, tasks, running_tasks)


def cleanup_resources():
    """Cleanup function for application shutdown"""
    
    # Cancel all running tasks
    for task_id, thread in running_tasks.items():
        if thread.is_alive():
            if task_id in tasks:
                tasks[task_id].status = "FAILED"
                tasks[task_id].error = "Application shutdown"
    
    # Wait for threads to complete (with timeout)
    start_time = time.time()
    timeout = 10  # 10 seconds timeout
    
    for task_id, thread in running_tasks.items():
        remaining_time = max(0, timeout - (time.time() - start_time))
        if remaining_time > 0 and thread.is_alive():
            thread.join(timeout=remaining_time)
            
        if thread.is_alive():
            print(f"Warning: Task {task_id} did not complete within timeout")
    
    print("ViDove web backend shutdown complete.")