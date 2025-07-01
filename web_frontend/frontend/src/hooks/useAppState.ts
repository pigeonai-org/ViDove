import { useState, useCallback } from 'react';
import { ChatMessage, SessionConfig, TaskInfo, UploadFileResponse, YouTubeUrlResponse } from '../types/api';

export interface AppState {
  sessionId: string | null;
  messages: ChatMessage[];
  currentMessage: string;
  currentConfig: SessionConfig;
  isLoading: boolean;
  tasks: TaskInfo[];
  uploadedFile: UploadFileResponse | null;
  youtubeUrlSubmitted: YouTubeUrlResponse | null;
  showTaskManager: boolean;
  error: string | null;
}

export const useAppState = () => {
  const [state, setState] = useState<AppState>({
    sessionId: null,
    messages: [],
    currentMessage: '',
    currentConfig: {} as SessionConfig,
    isLoading: false,
    tasks: [],
    uploadedFile: null,
    youtubeUrlSubmitted: null,
    showTaskManager: false,
    error: null,
  });

  const updateState = useCallback((updates: Partial<AppState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  return { state, updateState };
};
