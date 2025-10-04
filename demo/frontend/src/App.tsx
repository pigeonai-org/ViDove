import React, { useEffect, useCallback } from 'react';
import './App.css';

// Components
import {
  WelcomeScreen,
  Header,
  ErrorBanner,
  ChatInterface,
  ConfigurationPanel,
  TaskManager
} from './components';

// Hooks
import { useAppState, useApiActions } from './hooks';

const App: React.FC = () => {
  const { state, updateState } = useAppState();
  const {
    startNewSession,
    sendMessage,
    uploadFile,
    loadTasks,
    launchTask,
    updateConfig
  } = useApiActions(state, updateState);

  // Event handlers
  const handleKeyPress = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadFile(file);
    }
  }, [uploadFile]);

  const handleFileDropped = useCallback((file: File) => {
    uploadFile(file);
  }, [uploadFile]);

  const handleToggleTaskManager = useCallback(() => {
    updateState({ showTaskManager: !state.showTaskManager });
  }, [state.showTaskManager, updateState]);

  const handleDismissError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  const handleMessageChange = useCallback((message: string) => {
    updateState({ currentMessage: message });
  }, [updateState]);

  // Auto-load tasks when task manager is visible
  useEffect(() => {
    if (state.showTaskManager) {
      loadTasks();
      const interval = setInterval(loadTasks, 5000);
      return () => clearInterval(interval);
    }
  }, [state.showTaskManager, loadTasks]);

  // Show welcome screen if no session
  if (!state.sessionId) {
    return (
      <WelcomeScreen
        isLoading={state.isLoading}
        error={state.error}
        onStartSession={startNewSession}
      />
    );
  }

  return (
    <div className="app">
      <Header
        showTaskManager={state.showTaskManager}
        onToggleTaskManager={handleToggleTaskManager}
        onNewSession={startNewSession}
      />

      {state.error && (
        <ErrorBanner
          error={state.error}
          onDismiss={handleDismissError}
        />
      )}

      <div className="main-content">
        <ChatInterface
          messages={state.messages}
          currentMessage={state.currentMessage}
          isLoading={state.isLoading}
          uploadedFile={state.uploadedFile}
          youtubeUrlSubmitted={state.youtubeUrlSubmitted}
          onMessageChange={handleMessageChange}
          onSendMessage={sendMessage}
          onKeyPress={handleKeyPress}
          onFileUpload={handleFileUpload}
          onFileDropped={handleFileDropped}
        />

        <ConfigurationPanel
          currentConfig={state.currentConfig}
          uploadedFile={state.uploadedFile}
          youtubeUrlSubmitted={state.youtubeUrlSubmitted}
          isLoading={state.isLoading}
          onLaunchTask={launchTask}
          onConfigUpdate={updateConfig}
          sessionId={state.sessionId}
        />
      </div>

      <TaskManager
        tasks={state.tasks}
        isVisible={state.showTaskManager}
        sessionId={state.sessionId}
      />
    </div>
  );
};

export default App;