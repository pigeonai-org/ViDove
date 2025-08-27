import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChatMessage } from '../types/api';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  currentMessage: string;
  isLoading: boolean;
  uploadedFile: { filename: string } | null;
  youtubeUrlSubmitted: { youtube_url: string } | null;
  onMessageChange: (message: string) => void;
  onSendMessage: () => void;
  onKeyPress: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onFileDropped: (file: File) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  currentMessage,
  isLoading,
  uploadedFile,
  youtubeUrlSubmitted,
  onMessageChange,
  onSendMessage,
  onKeyPress,
  onFileUpload,
  onFileDropped
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      // Check if it's a valid file type
      const validExtensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg', '.srt'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      
      if (validExtensions.includes(fileExtension)) {
        onFileDropped(file);
      } else {
        alert('Please upload a valid video, audio, or SRT file.');
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="chat-container">
      <div 
        className={`chat-messages ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragOver && (
          <div className="drag-overlay">
            <div className="drag-message">
              📁 Drop your file here to upload
            </div>
          </div>
        )}
        
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-content">
              <div className="message-text">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <div className="input-row">
          <button 
            onClick={handleUploadClick}
            className="upload-icon-button"
            title="Upload file"
            disabled={isLoading}
          >
            📎
          </button>
          
          <input
            ref={fileInputRef}
            type="file"
            onChange={onFileUpload}
            accept="video/*,audio/*,.srt"
            style={{ display: 'none' }}
          />
          
          <textarea
            value={currentMessage}
            onChange={(e) => onMessageChange(e.target.value)}
            onKeyPress={onKeyPress}
            placeholder="Ask me about translation settings, upload a file, or paste a YouTube URL..."
            disabled={isLoading}
            rows={2}
          />
          <div className="input-buttons">
            <button 
              onClick={onSendMessage} 
              disabled={isLoading || !currentMessage.trim()}
              className="send-button"
            >
              Send
            </button>
          </div>
        </div>
        
        {(uploadedFile || youtubeUrlSubmitted) && (
          <div className="upload-status">
            {uploadedFile && (
              <span>✅ {uploadedFile.filename} uploaded successfully</span>
            )}
            {youtubeUrlSubmitted && (
              <span>✅ YouTube URL submitted successfully</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
