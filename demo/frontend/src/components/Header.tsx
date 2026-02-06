import React from 'react';

interface HeaderProps {
  showTaskManager: boolean;
  onToggleTaskManager: () => void;
  onNewSession: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  showTaskManager,
  onToggleTaskManager,
  onNewSession
}) => {
  return (
    <div className="header">
      <h1>🐦 ViDove Translation Assistant</h1>
      <div className="header-buttons">
        <button 
          onClick={onToggleTaskManager}
          className="header-button"
        >
          {showTaskManager ? 'Hide Tasks' : 'Show Tasks'}
        </button>
        <button onClick={onNewSession} className="header-button">
          New Session
        </button>
      </div>
    </div>
  );
};
