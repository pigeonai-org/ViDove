import React, { useState, useEffect, useCallback, useRef } from 'react';
import { TaskInfo, TaskResultResponse, TaskStatus as TaskStatusType } from '../types/api';
import { apiService } from '../services/api';
import { AgentConversation } from './AgentConversation';

type TaskStatus = 'CREATED' | 'RUNNING' | 'COMPLETED' | 'FAILED';

interface TaskManagerProps {
  tasks: TaskInfo[];
  isVisible: boolean;
  sessionId: string | null;  // Add sessionId prop
}

interface TaskDetails extends TaskInfo {
  fullStatus?: TaskStatusType;
  results?: TaskResultResponse;
  isLoadingResults?: boolean;
  resultsFetched?: boolean; // Flag to track if results have been successfully fetched
}

const getStatusColor = (status: string): string => {
  switch (status as TaskStatus) {
    case 'COMPLETED': return '#10B981';
    case 'RUNNING': return '#3B82F6';
    case 'FAILED': return '#EF4444';
    default: return '#6B7280';
  }
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const formatDate = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleString();
};

const getInputTypeDisplay = (inputType: string): string => {
  if (!inputType || inputType === 'Unknown') {
    return 'video'; // Default assumption based on the UI context
  }
  return inputType;
};

export const TaskManager: React.FC<TaskManagerProps> = ({ tasks, isVisible, sessionId }) => {
  const [taskDetails, setTaskDetails] = useState<{ [key: string]: TaskDetails }>({});
  const processedTasksRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    // Update task details while preserving existing data
    setTaskDetails(prevDetails => {
      const newDetails = { ...prevDetails };
      
      tasks.forEach(task => {
        if (newDetails[task.task_id]) {
          // Task exists, update basic info but preserve fetched results and flags
          newDetails[task.task_id] = {
            ...newDetails[task.task_id],
            ...task, // Update basic task info (status, etc.)
            // Preserve existing result-related fields if they exist
            results: newDetails[task.task_id].results,
            fullStatus: newDetails[task.task_id].fullStatus,
            isLoadingResults: newDetails[task.task_id].isLoadingResults,
            resultsFetched: newDetails[task.task_id].resultsFetched
          };
        } else {
          // New task, initialize it
          newDetails[task.task_id] = { ...task };
        }
      });
      
      // Remove tasks that no longer exist
      const currentTaskIds = new Set(tasks.map(t => t.task_id));
      Object.keys(newDetails).forEach(taskId => {
        if (!currentTaskIds.has(taskId)) {
          delete newDetails[taskId];
          processedTasksRef.current.delete(taskId);
        }
      });
      
      return newDetails;
    });
  }, [tasks]);

  const fetchTaskResults = useCallback(async (taskId: string) => {
    // Early exit if already processed or no session ID
    if (processedTasksRef.current.has(taskId) || !sessionId) {
      return;
    }
    
    // Mark as processed immediately
    processedTasksRef.current.add(taskId);
    
    setTaskDetails(prev => ({
      ...prev,
      [taskId]: { 
        ...prev[taskId], 
        isLoadingResults: true 
      }
    }));

    try {
      const [statusResponse, resultsResponse] = await Promise.all([
        apiService.getTaskStatus(taskId, sessionId),
        apiService.getTaskResults(taskId, sessionId)
      ]);

      setTaskDetails(prev => ({
        ...prev,
        [taskId]: {
          ...prev[taskId],
          fullStatus: statusResponse,
          results: resultsResponse,
          isLoadingResults: false,
          resultsFetched: true
        }
      }));
    } catch (error) {
      console.error('Failed to fetch task results:', error);
      setTaskDetails(prev => ({
        ...prev,
        [taskId]: { 
          ...prev[taskId], 
          isLoadingResults: false 
        }
      }));
      // Remove from processed on error so it can be retried
      processedTasksRef.current.delete(taskId);
    }
  }, [sessionId]);

  // Auto-fetch results for completed tasks
  useEffect(() => {
    tasks.forEach(task => {
      if (task.status === 'COMPLETED' && !processedTasksRef.current.has(task.task_id)) {
        // Delay to ensure files are written
        setTimeout(() => {
          fetchTaskResults(task.task_id);
        }, 1000);
      }
    });
  }, [tasks, fetchTaskResults]);

  const handleDownload = async (taskId: string, filename?: string) => {
    if (!sessionId) {
      alert('No active session. Please refresh the page.');
      return;
    }
    
    try {
      if (filename) {
        await apiService.downloadTaskFile(taskId, filename, sessionId);
      } else {
        await apiService.downloadPrimaryResult(taskId, sessionId);
      }
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  };

  if (!isVisible) return null;

  return (
    <div className="task-manager">
      <h3>Task Manager</h3>
      <div className="tasks-list">
        {tasks.length === 0 ? (
          <p>No tasks created yet.</p>
        ) : (
          tasks.map((task) => {
            const details = taskDetails[task.task_id] || task;
            const isCompleted = task.status === 'COMPLETED';
            const hasResults = details.results?.has_results;

            return (
              <div key={task.task_id} className="task-item">
                <div className="task-header">
                  <div className="task-info">
                    <div className="task-id">Task: {task.task_id.substring(0, 8)}...</div>
                    <div className="task-type">Type: {getInputTypeDisplay(task.input_type)}</div>
                    <div className="task-created">
                      Created: {new Date(task.created_at).toLocaleString()}
                    </div>
                    {details.fullStatus?.error && (
                      <div className="task-error">Error: {details.fullStatus.error}</div>
                    )}
                  </div>
                  
                  <div 
                    className="task-status"
                    style={{ backgroundColor: getStatusColor(task.status) }}
                  >
                    {task.status}
                  </div>
                </div>

                <div className="task-content">
                  {/* Left column - Agent Conversation */}
                  <div className="task-conversation">
                    {(task.status === 'RUNNING' || task.status === 'COMPLETED') && sessionId && (
                      <AgentConversation
                        taskId={task.task_id}
                        sessionId={sessionId}
                        isVisible={true}
                        taskStatus={task.status}
                      />
                    )}
                  </div>

                  {/* Right column - Downloads and Results */}
                  <div className="task-actions">
                    {isCompleted && (
                      <div className="task-downloads">
                        {details.isLoadingResults && (
                          <div className="loading">Fetching results automatically...</div>
                        )}

                        {hasResults && (
                          <div className="download-section">
                            <div className="download-header">Available Downloads:</div>
                            {details.results!.files.map((file, index) => (
                              <div key={index} className="file-item">
                                <div className="file-info">
                                  <span className="file-name">{file.filename}</span>
                                  <span className="file-details">
                                    {file.file_type} • {formatFileSize(file.size_bytes)}
                                    {file.created_at && ` • ${formatDate(file.created_at)}`}
                                  </span>
                                </div>
                                <button
                                  onClick={() => handleDownload(task.task_id, file.filename)}
                                  className="btn-download"
                                >
                                  Download
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        {details.results && !hasResults && (
                          <div className="no-results">No result files found</div>
                        )}

                        {!details.results && !details.isLoadingResults && !details.resultsFetched && (
                          <div className="loading">Preparing to fetch results...</div>
                        )}

                        {details.resultsFetched && !details.results?.has_results && (
                          <div className="no-results">Task completed but no result files were generated</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
