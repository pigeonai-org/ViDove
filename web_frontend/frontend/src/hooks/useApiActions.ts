import { useCallback } from 'react';
import { apiService, ApiError } from '../services/api';
import { ChatMessage, UploadFileResponse } from '../types/api';
import { AppState } from './useAppState';

export const useApiActions = (
  state: AppState, 
  updateState: (updates: Partial<AppState>) => void
) => {
  const handleError = useCallback((error: unknown) => {
    const errorMessage = error instanceof ApiError 
      ? error.message 
      : error instanceof Error 
        ? error.message 
        : 'An unknown error occurred';
    
    console.error('Error:', error);
    updateState({ error: errorMessage, isLoading: false });
  }, [updateState]);

  const startNewSession = useCallback(async () => {
    try {
      updateState({ isLoading: true, error: null });
      
      const data = await apiService.startChatSession();
      
      updateState({
        sessionId: data.session_id,
        currentConfig: { ...data.current_config }, // Create new object reference
        messages: [
          {
            role: 'assistant',
            content: data.message,
            timestamp: new Date().toISOString(),
          },
        ],
        isLoading: false,
      });
    } catch (error) {
      handleError(error);
    }
  }, [updateState, handleError]);

  const sendMessage = useCallback(async () => {
    if (!state.currentMessage.trim() || !state.sessionId || state.isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: state.currentMessage.trim(),
      timestamp: new Date().toISOString(),
    };

    updateState({
      messages: [...state.messages, userMessage],
      currentMessage: '',
      isLoading: true,
      error: null,
    });

    try {
      const data = await apiService.sendMessage(state.sessionId, userMessage);
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.message,
        timestamp: new Date().toISOString(),
      };

      // Check if a YouTube URL was detected in the config updates
      const configUpdates = data.config_updates || {};
      if (configUpdates.youtube_url && configUpdates.input_type === 'youtube') {
        try {
          // Submit the YouTube URL to the backend
          const youtubeResponse = await apiService.submitYouTubeUrl(state.sessionId, configUpdates.youtube_url);
          
          // Update the state to show YouTube URL was submitted
        updateState({
          messages: [...state.messages, userMessage, assistantMessage],
          currentConfig: { ...data.current_config }, // Create new object reference
          youtubeUrlSubmitted: youtubeResponse,
          uploadedFile: null, // Clear uploaded file when YouTube URL is set
          isLoading: false,
        });
        } catch (youtubeError) {
          console.error('Failed to submit YouTube URL:', youtubeError);
          // Continue with normal flow even if YouTube URL submission fails
          updateState({
            messages: [...state.messages, userMessage, assistantMessage],
            currentConfig: { ...data.current_config }, // Create new object reference
            isLoading: false,
          });
        }
      } else {
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: data.message,
          timestamp: new Date().toISOString(),
        };
        
        updateState({
          messages: [...state.messages, userMessage, assistantMessage],
          currentConfig: { ...data.current_config }, // Create new object reference
          isLoading: false,
        });
      }

    } catch (error) {
      handleError(error);
      updateState({
        messages: [...state.messages, userMessage, {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date().toISOString(),
        }],
      });
    }
  }, [state.currentMessage, state.sessionId, state.isLoading, state.messages, updateState, handleError]);

  const uploadFile = useCallback(async (file: File): Promise<UploadFileResponse | null> => {
    if (!state.sessionId) {
      updateState({ error: 'No active session. Please start a new session first.' });
      return null;
    }

    try {
      updateState({ error: null, isLoading: true });
      const data = await apiService.uploadFile(state.sessionId, file);
      updateState({ 
        uploadedFile: data,
        youtubeUrlSubmitted: null, // Clear YouTube URL when file is uploaded
        isLoading: false 
      });
      
      // Send a message to the assistant about the file upload
      const uploadMessage: ChatMessage = {
        role: 'user',
        content: `I've uploaded a file: ${data.filename}. Please help me configure the translation settings for this file.`,
        timestamp: new Date().toISOString(),
      };
      
      // Update messages and send to assistant
      const updatedMessages = [...state.messages, uploadMessage];
      updateState({ messages: updatedMessages });
      
      const assistantResponse = await apiService.sendMessage(state.sessionId, uploadMessage);
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: assistantResponse.message,
        timestamp: new Date().toISOString(),
      };

      updateState({
        messages: [...updatedMessages, assistantMessage],
        currentConfig: { ...assistantResponse.current_config }, // Create new object reference
      });
      
      return data;
    } catch (error) {
      handleError(error);
      return null;
    }
  }, [state.sessionId, state.messages, updateState, handleError]);

  const loadTasks = useCallback(async () => {
    if (!state.sessionId) {
      console.warn('Cannot load tasks: No active session');
      return;
    }
    
    try {
      const data = await apiService.listTasks(state.sessionId);
      updateState({ tasks: data });
    } catch (error) {
      console.error('Error loading tasks:', error);
    }
  }, [state.sessionId, updateState]);

  const launchTask = useCallback(async () => {
    if (!state.sessionId) {
      updateState({ error: 'No active session.' });
      return;
    }

    try {
      updateState({ isLoading: true, error: null });
      
      const data = await apiService.launchTaskFromSession(state.sessionId);
      
      updateState({ 
        isLoading: false,
        showTaskManager: true 
      });
      
      // Add a success message to the chat
      const taskMessage: ChatMessage = {
        role: 'assistant',
        content: `🚀 Great! I've launched your translation task (ID: ${data.task_id.substring(0, 8)}...). You can monitor its progress in the Task Manager.`,
        timestamp: new Date().toISOString(),
      };
      
      updateState({
        messages: [...state.messages, taskMessage],
      });
      
      loadTasks();

    } catch (error) {
      handleError(error);
    }
  }, [state.sessionId, state.messages, updateState, handleError, loadTasks]);

  const updateConfig = useCallback(async (key: string, value: any) => {
    if (!state.sessionId) return;

    try {
      // Send the key as-is (dotted notation) to the API
      const configUpdates: Record<string, any> = { [key]: value };
      
      const data = await apiService.updateConfig(state.sessionId, configUpdates);
      
      // Merge the returned config with the existing config to ensure all fields are preserved
      const mergedConfig = {
        ...state.currentConfig,
        ...data.config
      };
      
      updateState({
        currentConfig: { ...mergedConfig } // Create new object reference
      });
    } catch (error) {
      handleError(error);
      throw error; // Re-throw so the component can handle UI rollback
    }
  }, [state.sessionId, state.currentConfig, updateState, handleError]);

  return {
    startNewSession,
    sendMessage,
    uploadFile,
    loadTasks,
    launchTask,
    updateConfig,
    handleError
  };
};
