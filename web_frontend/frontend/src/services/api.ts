import {
  ChatMessage,
  StartSessionResponse,
  SendMessageResponse,
  ConfigResponse,
  TaskRequest,
  CreateTaskResponse,
  TaskStatus,
  TaskInfo,
  UploadFileResponse,
  YouTubeUrlRequest,
  YouTubeUrlResponse,
  SessionConfig,
  TaskResultResponse,
  AgentConversationResponse,
  ApiError as ApiErrorInterface
} from '../types/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || `${window.location.protocol}//${window.location.hostname}:8000`;

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData: ApiErrorInterface = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`
        }));
        throw new ApiError(response.status, errorData.detail);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      throw new Error(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Chat & Configuration endpoints
  async startChatSession(): Promise<StartSessionResponse> {
    return this.request<StartSessionResponse>('/api/chat/start', {
      method: 'POST',
    });
  }

  async sendMessage(sessionId: string, message: ChatMessage): Promise<SendMessageResponse> {
    return this.request<SendMessageResponse>(`/api/chat/${sessionId}/message`, {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  async getCurrentConfig(sessionId: string): Promise<ConfigResponse> {
    return this.request<ConfigResponse>(`/api/chat/${sessionId}/config`);
  }

  async updateConfig(sessionId: string, configUpdates: Partial<SessionConfig>): Promise<ConfigResponse> {
    return this.request<ConfigResponse>(`/api/chat/${sessionId}/config`, {
      method: 'POST',
      body: JSON.stringify(configUpdates),
    });
  }

  // Task Management endpoints
  async createTask(taskRequest: TaskRequest): Promise<CreateTaskResponse> {
    return this.request<CreateTaskResponse>('/api/tasks/create', {
      method: 'POST',
      body: JSON.stringify(taskRequest),
    });
  }

  async getTaskStatus(taskId: string, sessionId: string): Promise<TaskStatus> {
    return this.request<TaskStatus>(`/api/tasks/${taskId}/status?session_id=${sessionId}`);
  }

  async listTasks(sessionId: string): Promise<TaskInfo[]> {
    return this.request<TaskInfo[]>(`/api/sessions/${sessionId}/tasks`);
  }

  // Task launch from session
  async launchTaskFromSession(sessionId: string): Promise<CreateTaskResponse> {
    return this.request<CreateTaskResponse>(`/api/sessions/${sessionId}/launch`, {
      method: 'POST',
    });
  }

  // File Upload
  async uploadFile(sessionId: string, file: File): Promise<UploadFileResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<UploadFileResponse>(`/api/upload/${sessionId}`, {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it with boundary
      body: formData,
    });
  }

  // YouTube URL Submission
  async submitYouTubeUrl(sessionId: string, youtubeUrl: string): Promise<YouTubeUrlResponse> {
    const request: YouTubeUrlRequest = {
      youtube_url: youtubeUrl,
    };

    return this.request<YouTubeUrlResponse>(`/api/youtube/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Task Results and Downloads
  async getTaskResults(taskId: string, sessionId: string): Promise<TaskResultResponse> {
    return this.request<TaskResultResponse>(`/api/tasks/${taskId}/results?session_id=${sessionId}`);
  }

  // Agent Conversation
  async getTaskConversation(taskId: string, sessionId: string): Promise<AgentConversationResponse> {
    return this.request<AgentConversationResponse>(`/api/tasks/${taskId}/conversation?session_id=${sessionId}`);
  }

  async downloadPrimaryResult(taskId: string, sessionId: string): Promise<void> {
    const url = `${API_BASE_URL}/api/tasks/${taskId}/download?session_id=${sessionId}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorData: ApiErrorInterface = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`
      }));
      throw new ApiError(response.status, errorData.detail);
    }

    // Create download link
    const blob = await response.blob();
    const contentDisposition = response.headers.get('content-disposition');
    const filename = contentDisposition 
      ? contentDisposition.split('filename=')[1]?.replace(/"/g, '') 
      : `task_${taskId}_result.srt`;

    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  }

  async downloadTaskFile(taskId: string, filename: string, sessionId: string): Promise<void> {
    const url = `${API_BASE_URL}/api/tasks/${taskId}/download/${encodeURIComponent(filename)}?session_id=${sessionId}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorData: ApiErrorInterface = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`
      }));
      throw new ApiError(response.status, errorData.detail);
    }

    // Create download link
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  }

  // Health check
  async healthCheck(): Promise<{ message: string; version: string }> {
    return this.request<{ message: string; version: string }>('/');
  }
}

export const apiService = new ApiService();
export { ApiError };