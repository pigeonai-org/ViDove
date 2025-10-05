"""
Main FastAPI application entry point for the ViDove web backend.
"""
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, AsyncGenerator, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse



from models import (
    StartSessionResponse, SendMessageResponse, ConfigResponse, TaskRequest,
    CreateTaskResponse, TaskStatus, TaskInfo, UploadFileResponse, ChatMessage,
    YouTubeUrlRequest, YouTubeUrlResponse, TaskResultResponse, AgentConversationResponse
)
import endpoints


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    # Startup code
    endpoints.start_cleanup_thread()
    endpoints.cleanup_orphaned_temp_dirs()
    print("ViDove backend started with memory management enabled")
    
    yield
    
    # Shutdown code
    endpoints.cleanup_resources()


# Initialize FastAPI app
app = FastAPI(
    title="ViDove Translation Assistant API",
    description="LLM-powered conversational configuration for video translation tasks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production flexibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for task execution
executor = ThreadPoolExecutor(max_workers=8)


# Health check endpoints
@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {"message": "ViDove Translation Assistant API", "version": "1.0.0"}


@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """Health check endpoint for Docker healthcheck"""
    return {"status": "healthy", "service": "vidove-backend"}


@app.get("/api/system/memory")
async def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics for monitoring"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Count running tasks
        running_task_count = len([t for t in endpoints.tasks.values() if t.status == "RUNNING"])
        
        return {
            "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "process_memory_percent": round(process.memory_percent(), 2),
            "system_memory_total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
            "system_memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "system_memory_percent": psutil.virtual_memory().percent,
            "active_sessions": len(endpoints.sessions),
            "active_tasks": running_task_count,
            "total_tasks": len(endpoints.tasks),
            "max_concurrent_tasks": endpoints.MAX_CONCURRENT_TASKS,
            "available_task_slots": endpoints.MAX_CONCURRENT_TASKS - running_task_count
        }
    except ImportError:
        return {
            "error": "psutil not installed",
            "active_sessions": len(endpoints.sessions),
            "active_tasks": len([t for t in endpoints.tasks.values() if t.status == "RUNNING"]),
            "total_tasks": len(endpoints.tasks)
        }


# Chat and Configuration endpoints
@app.post("/api/chat/start", response_model=StartSessionResponse)
async def start_chat_session(request: Request) -> StartSessionResponse:
    """Start a new configuration chat session"""
    return await endpoints.start_chat_session(request)


@app.post("/api/chat/{session_id}/message", response_model=SendMessageResponse)
async def send_message(session_id: str, message: ChatMessage) -> SendMessageResponse:
    """Send a message to the chat session"""
    return await endpoints.send_message(session_id, message)


@app.get("/api/chat/{session_id}/config", response_model=ConfigResponse)
async def get_current_config(session_id: str) -> ConfigResponse:
    """Get current configuration for a session"""
    return await endpoints.get_current_config(session_id)


@app.post("/api/chat/{session_id}/config", response_model=ConfigResponse)
async def update_config(session_id: str, config_updates: Dict[str, Any]) -> ConfigResponse:
    """Update configuration for a session"""
    return await endpoints.update_config(session_id, config_updates)


# Task Management endpoints
@app.post("/api/tasks/create", response_model=CreateTaskResponse)
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks) -> CreateTaskResponse:
    """Create a new translation task"""
    return await endpoints.create_task(task_request, background_tasks)


@app.post("/api/sessions/{session_id}/launch", response_model=CreateTaskResponse)
async def launch_task_from_session(session_id: str, request: Request) -> CreateTaskResponse:
    """Launch a translation task from a completed configuration session"""
    return await endpoints.launch_task_from_session(session_id, request)


@app.get("/api/tasks/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str, session_id: str) -> TaskStatus:
    """Get status of a translation task"""
    return await endpoints.get_task_status(task_id, session_id)


@app.get("/api/sessions/{session_id}/tasks", response_model=List[TaskInfo])
async def list_tasks(session_id: str) -> List[TaskInfo]:
    """List translation tasks for a specific session"""
    return await endpoints.list_tasks(session_id)


@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str, session_id: str) -> Dict[str, str]:
    """Cancel a running task"""
    return await endpoints.cancel_task(task_id, session_id)


@app.get("/api/tasks/{task_id}/conversation", response_model=AgentConversationResponse)
async def get_task_conversation(task_id: str, session_id: str) -> AgentConversationResponse:
    """Get agent conversation history for a task"""
    return await endpoints.get_task_conversation(task_id, session_id)


# Task Result endpoints
@app.get("/api/tasks/{task_id}/results", response_model=TaskResultResponse)
async def get_task_results(task_id: str, session_id: str) -> TaskResultResponse:
    """Get information about task result files"""
    return await endpoints.get_task_results(task_id, session_id)


@app.get("/api/tasks/{task_id}/download", response_class=FileResponse)
async def download_primary_result(task_id: str, session_id: str) -> FileResponse:
    """Download the primary result file (usually subtitle) for a task"""
    return await endpoints.download_primary_result(task_id, session_id)


@app.get("/api/tasks/{task_id}/download/{filename}", response_class=FileResponse)
async def download_task_file(task_id: str, filename: str, session_id: str) -> FileResponse:
    """Download a specific result file for a task"""
    return await endpoints.download_task_file(task_id, filename, session_id)


# File Upload endpoint
@app.post("/api/upload/{session_id}", response_model=UploadFileResponse)
async def upload_file(session_id: str, file: UploadFile = File(...), request: Request = None) -> UploadFileResponse:
    """Upload a file for translation and associate it with a session"""
    return await endpoints.upload_file(session_id, file, request)


# YouTube URL submission endpoint
@app.post("/api/youtube/{session_id}", response_model=YouTubeUrlResponse)
async def submit_youtube_url(session_id: str, request: YouTubeUrlRequest) -> YouTubeUrlResponse:
    """Submit a YouTube URL for translation and associate it with a session"""
    return await endpoints.submit_youtube_url(session_id, request)


# Application entry point
if __name__ == "__main__":
    import uvicorn
    try:
        # Configure uvicorn to allow larger file uploads (e.g., 1GB limit)
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            limit_max_requests=1000,
            limit_concurrency=1000,
            # Set request body size limit to 512MB (reduced from 2GB for better memory management)
            # This allows file uploads while leaving more memory for processing
            h11_max_incomplete_event_size= 512 * 1024 * 1024
        )
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        # Ensure cleanup even if not run through uvicorn's lifecycle
        endpoints.cleanup_resources()
        if hasattr(executor, 'shutdown'):
            executor.shutdown(wait=True)
