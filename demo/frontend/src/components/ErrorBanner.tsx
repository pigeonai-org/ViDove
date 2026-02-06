import React from 'react';

interface ErrorBannerProps {
  error: string;
  onDismiss: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({ error, onDismiss }) => {
  return (
    <div className="error-banner">
      <span>Error: {error}</span>
      <button onClick={onDismiss}>×</button>
    </div>
  );
};
