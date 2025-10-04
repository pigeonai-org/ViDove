import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AgentConversationMessage } from '../types/api';
import { apiService } from '../services/api';

interface AgentConversationProps {
  taskId: string;
  sessionId: string;
  isVisible: boolean;
  taskStatus: string;
}

export const AgentConversation: React.FC<AgentConversationProps> = ({ 
  taskId, 
  sessionId, 
  isVisible, 
  taskStatus 
}) => {
  const [conversation, setConversation] = useState<AgentConversationMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchConversation = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await apiService.getTaskConversation(taskId, sessionId);
      setConversation(response.conversation);
      setIsLive(response.is_live);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch conversation');
      console.error('Failed to fetch conversation:', err);
    } finally {
      setIsLoading(false);
    }
  }, [taskId, sessionId]);

  // Initial fetch
  useEffect(() => {
    if (isVisible && taskId && sessionId) {
      fetchConversation();
    }
  }, [isVisible, taskId, sessionId]);

  // Set up polling for running tasks
  useEffect(() => {
    if (!isVisible || taskStatus === 'COMPLETED' || taskStatus === 'FAILED') {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    if (taskStatus === 'RUNNING') {
      pollIntervalRef.current = setInterval(() => {
        fetchConversation();
      }, 3000); // Poll every 3 seconds for running tasks
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [isVisible, taskStatus]);

  if (!isVisible) {
    return null;
  }

  const getRoleColor = (role: string): string => {
    const roleColors: { [key: string]: string } = {
      'pipeline_coordinator': '#8B5CF6',
      'vision_agent': '#10B981',
      'audio_agent': '#F59E0B',
      'translator': '#3B82F6',
      'proofreader': '#EF4444',
      'editor': '#EC4899',
    };
    return roleColors[role] || '#6B7280';
  };

  const formatRole = (role: string): string => {
    return role.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div style={styles.agentConversation}>
      <div style={styles.conversationHeader}>
        <h4 style={styles.headerTitle}>Agent Conversation</h4>
        {isLive && taskStatus === 'RUNNING' && (
          <span style={styles.liveIndicator}>
            <div style={styles.liveDot}></div>
            Live
          </span>
        )}
      </div>

      {isLoading && conversation.length === 0 && (
        <div style={styles.loading}>Loading conversation...</div>
      )}

      {error && (
        <div style={styles.error}>
          {error}
          <button onClick={fetchConversation} style={styles.retryButton}>
            Retry
          </button>
        </div>
      )}

      {conversation.length === 0 && !isLoading && !error && (
        <div style={styles.noConversation}>
          No agent conversation available yet.
        </div>
      )}

      {conversation.length > 0 && (
        <div style={styles.conversationMessages}>
          {conversation.map((message, index) => (
            <div key={index} style={styles.conversationMessage}>
              <div 
                style={{
                  ...styles.messageRole,
                  color: getRoleColor(message.role)
                }}
              >
                {formatRole(message.role)}
              </div>
              <div style={styles.messageContent}>
                {message.message}
              </div>
              {message.timestamp && (
                <div style={styles.messageTimestamp}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const styles = {
  agentConversation: {
    marginTop: '16px',
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    background: '#f9fafb'
  },
  conversationHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    borderBottom: '1px solid #e5e7eb',
    background: 'white',
    borderRadius: '8px 8px 0 0'
  },
  headerTitle: {
    margin: 0,
    color: '#374151',
    fontSize: '14px',
    fontWeight: '600'
  },
  liveIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: '#10B981',
    fontSize: '12px',
    fontWeight: '500'
  },
  liveDot: {
    width: '8px',
    height: '8px',
    background: '#10B981',
    borderRadius: '50%',
    animation: 'pulse 2s infinite'
  },
  conversationMessages: {
    maxHeight: '300px',
    overflowY: 'auto' as const,
    padding: '12px 16px'
  },
  conversationMessage: {
    marginBottom: '12px',
    padding: '8px 12px',
    background: 'white',
    borderRadius: '6px',
    borderLeft: '3px solid #e5e7eb'
  },
  messageRole: {
    fontSize: '12px',
    fontWeight: '600',
    marginBottom: '4px'
  },
  messageContent: {
    fontSize: '13px',
    color: '#374151',
    lineHeight: '1.4'
  },
  messageTimestamp: {
    fontSize: '11px',
    color: '#6b7280',
    marginTop: '4px'
  },
  loading: {
    padding: '16px',
    textAlign: 'center' as const,
    color: '#6b7280',
    fontSize: '13px'
  },
  error: {
    padding: '16px',
    textAlign: 'center' as const,
    color: '#ef4444',
    fontSize: '13px'
  },
  noConversation: {
    padding: '16px',
    textAlign: 'center' as const,
    color: '#6b7280',
    fontSize: '13px'
  },
  retryButton: {
    marginLeft: '8px',
    padding: '4px 8px',
    background: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '11px',
    cursor: 'pointer'
  }
};
