import React from 'react';

interface WelcomeScreenProps {
  isLoading: boolean;
  error: string | null;
  onStartSession: () => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  isLoading,
  error,
  onStartSession
}) => {
  return (
    <div className="app">
      <div className="welcome-container">
        <div className="welcome-content">
          <h1>🐦 ViDove Translation Assistant</h1>
          <p>I'll help you configure and run video translation tasks through an interactive conversation.</p>
          {error && (
            <div className="error-message">
              Error: {error}
            </div>
          )}
          <button 
            onClick={onStartSession} 
            disabled={isLoading} 
            className="start-button"
          >
            {isLoading ? 'Starting...' : 'Start Configuration Chat'}
          </button>
        </div>
      </div>
    </div>
  );
};
