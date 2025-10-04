import React, { useState, useEffect, useRef } from 'react';
import { SessionConfig, UploadFileResponse, YouTubeUrlResponse } from '../types/api';

type ConfigFieldType = 'select' | 'boolean' | 'number' | 'readonly' | 'text-array';

interface ConfigFieldOption {
  value: string | number;
  label: string;
}

interface ConfigField {
  key: string;
  label: string;
  type: ConfigFieldType;
  description: string;
  options?: ConfigFieldOption[];
  min?: number;
  max?: number;
  step?: number;
  readonly?: boolean;
  category: string;
}

// Configuration schema based on backend CONFIGURATION_SCHEMA
const CONFIG_FIELDS: ConfigField[] = [
  // Language settings
  {
    key: 'source_lang',
    label: 'Source Language',
    type: 'select',
    description: 'Source language of the video content',
    category: 'Language',
    options: [
      { value: 'EN', label: 'English' },
      { value: 'ZH', label: 'Chinese' },
      { value: 'ES', label: 'Spanish' },
      { value: 'FR', label: 'French' },
      { value: 'DE', label: 'German' },
      { value: 'RU', label: 'Russian' },
      { value: 'JA', label: 'Japanese' },
      { value: 'AR', label: 'Arabic' },
      { value: 'KR', label: 'Korean' }
    ]
  },
  {
    key: 'target_lang',
    label: 'Target Language',
    type: 'select',
    description: 'Target language for translation',
    category: 'Language',
    options: [
      { value: 'EN', label: 'English' },
      { value: 'ZH', label: 'Chinese' },
      { value: 'ES', label: 'Spanish' },
      { value: 'FR', label: 'French' },
      { value: 'DE', label: 'German' },
      { value: 'RU', label: 'Russian' },
      { value: 'JA', label: 'Japanese' },
      { value: 'AR', label: 'Arabic' },
      { value: 'KR', label: 'Korean' }
    ]
  },
  {
    key: 'domain',
    label: 'Domain',
    type: 'select',
    description: 'Content domain for specialized translation',
    category: 'Language',
    options: [
      { value: 'General', label: 'General' },
      { value: 'SC2', label: 'StarCraft 2' },
      { value: 'CS:GO', label: 'Counter-Strike: Global Offensive' }
    ]
  },
  
  // Translation settings
  {
    key: 'translation.model',
    label: 'Translation Model',
    type: 'select',
    description: 'LLM model for translation',
    category: 'Translation',
    options: [
      { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
      { value: 'gpt-4o', label: 'GPT-4o' },
      { value: 'gpt-5', label: 'GPT-5' },
      { value: 'gpt-5-mini', label: 'GPT-5 Mini' }
    ]
  },
  {
    key: 'translation.chunk_size',
    label: 'Chunk Size',
    type: 'number',
    description: 'Text chunk size for translation (characters)',
    category: 'Translation',
    min: 100,
    max: 5000,
    step: 100
  },
  
  // Video settings
  {
    key: 'video_download.resolution',
    label: 'Video Resolution',
    type: 'select',
    description: 'Video resolution for download',
    category: 'Video',
    options: [
      { value: 360, label: '360p' },
      { value: 480, label: '480p' },
      { value: 720, label: '720p' },
      { value: 'best', label: 'Best Available' }
    ]
  },
  
  // Audio settings (audio is always enabled with Whisper agent)
  {
    key: 'audio.audio_agent',
    label: 'Audio Agent',
    type: 'readonly',
    description: 'Audio processing agent (WhisperAudioAgent - system managed)',
    category: 'Audio',
    readonly: true
  },
  {
    key: 'audio.model_path',
    label: 'Audio Model Path',
    type: 'readonly',
    description: 'Path to audio model (system managed)',
    category: 'Audio',
    readonly: true
  },
  {
    key: 'audio.VAD_model',
    label: 'VAD Model',
    type: 'select',
    description: 'Voice Activity Detection model',
    category: 'Audio',
    options: [
      { value: 'API', label: 'Pyannote Speaker Diarization 3.1 API' },
    ]
  },
  {
    key: 'audio.src_lang',
    label: 'Audio Source Language',
    type: 'select',
    description: 'Source language for audio processing',
    category: 'Audio',
    options: [
      { value: 'en', label: 'English' },
      { value: 'zh', label: 'Chinese' },
      { value: 'es', label: 'Spanish' },
      { value: 'fr', label: 'French' },
      { value: 'de', label: 'German' },
      { value: 'ru', label: 'Russian' },
      { value: 'ja', label: 'Japanese' },
      { value: 'ar', label: 'Arabic' },
      { value: 'ko', label: 'Korean' }
    ]
  },
  {
    key: 'audio.tgt_lang',
    label: 'Audio Target Language',
    type: 'select',
    description: 'Target language for audio processing',
    category: 'Audio',
    options: [
      { value: 'en', label: 'English' },
      { value: 'zh', label: 'Chinese' },
      { value: 'es', label: 'Spanish' },
      { value: 'fr', label: 'French' },
      { value: 'de', label: 'German' },
      { value: 'ru', label: 'Russian' },
      { value: 'ja', label: 'Japanese' },
      { value: 'ar', label: 'Arabic' },
      { value: 'ko', label: 'Korean' }
    ]
  },
  
  // Vision settings
  {
    key: 'vision.enable_vision',
    label: 'Enable Vision',
    type: 'boolean',
    description: 'Enable vision processing',
    category: 'Vision'
  },
  {
    key: 'vision.vision_model',
    label: 'Vision Model',
    type: 'select',
    description: 'Vision model for visual content analysis',
    category: 'Vision',
    options: [
      { value: 'CLIP', label: 'CLIP' },
      { value: 'gpt-4o', label: 'GPT-4o Vision' },
      { value: 'gpt-4o-mini', label: 'GPT-4o Mini Vision' }
    ]
  },
  {
    key: 'vision.model_path',
    label: 'Vision Model Path',
    type: 'readonly',
    description: 'Path to vision model (system managed)',
    category: 'Vision',
    readonly: true
  },
  {
    key: 'vision.frame_cache_dir',
    label: 'Frame Cache Directory',
    type: 'readonly',
    description: 'Directory for caching frames (system managed)',
    category: 'Vision',
    readonly: true
  },
  {
    key: 'vision.frame_per_seg',
    label: 'Frames per Segment',
    type: 'number',
    description: 'Frames extracted per segment for vision analysis',
    category: 'Vision',
    min: 1,
    max: 10,
    step: 1
  },
  
  // Pre-processing
  {
    key: 'pre_process.sentence_form',
    label: 'Sentence Form',
    type: 'boolean',
    description: 'Normalize sentence structure before translation',
    category: 'Pre-processing'
  },
  {
    key: 'pre_process.spell_check',
    label: 'Spell Check',
    type: 'boolean',
    description: 'Check and correct spelling errors',
    category: 'Pre-processing'
  },
  {
    key: 'pre_process.term_correct',
    label: 'Term Correction',
    type: 'boolean',
    description: 'Apply domain-specific terminology corrections',
    category: 'Pre-processing'
  },
  
  // Post-processing
  {
    key: 'post_process.enable_post_process',
    label: 'Enable Post-processing',
    type: 'boolean',
    description: 'Enable post-processing module',
    category: 'Post-processing'
  },
  {
    key: 'post_process.check_len_and_split',
    label: 'Length Check & Split',
    type: 'boolean',
    description: 'Check subtitle length and split if necessary',
    category: 'Post-processing'
  },
  {
    key: 'post_process.remove_trans_punctuation',
    label: 'Remove Extra Punctuation',
    type: 'boolean',
    description: 'Remove translation artifacts and extra punctuation',
    category: 'Post-processing'
  },
  
  // Proofreader settings
  {
    key: 'proofreader.enable_proofreading',
    label: 'Enable Proofreading',
    type: 'boolean',
    description: 'Enable proofreading of translations',
    category: 'Proofreader'
  },
  {
    key: 'proofreader.window_size',
    label: 'Window Size',
    type: 'number',
    description: 'Number of sentences per proofreading chunk',
    category: 'Proofreader',
    min: 1,
    max: 20,
    step: 1
  },
  {
    key: 'proofreader.short_term_memory_len',
    label: 'Short Term Memory Length',
    type: 'number',
    description: 'Max sentences stored in short term memory',
    category: 'Proofreader',
    min: 1,
    max: 20,
    step: 1
  },
  {
    key: 'proofreader.enable_short_term_memory',
    label: 'Enable Short Term Memory',
    type: 'boolean',
    description: 'Use short term memory for proofreading',
    category: 'Proofreader'
  },
  {
    key: 'proofreader.verbose',
    label: 'Verbose Proofreading',
    type: 'boolean',
    description: 'Whether to print the proofreading process',
    category: 'Proofreader'
  },
  
  // Editor settings
  {
    key: 'editor.enable_editor',
    label: 'Enable Editor',
    type: 'boolean',
    description: 'Enable editor module for translation improvement',
    category: 'Editor'
  },
  {
    key: 'editor.user_instruction',
    label: 'User Instruction',
    type: 'select',
    description: 'Additional instructions for the editor',
    category: 'Editor',
    options: [
      { value: 'none', label: 'None' },
      { value: 'formal', label: 'Formal' },
      { value: 'casual', label: 'Casual' },
      { value: 'technical', label: 'Technical' }
    ]
  },
  {
    key: 'instructions',
    label: 'Custom Instructions',
    type: 'text-array',
    description: 'List of custom instructions for translation habits, jargon, and style preferences',
    category: 'Editor'
  },
  {
    key: 'editor.editor_context_window',
    label: 'Context Window',
    type: 'number',
    description: 'Sentences to provide as context',
    category: 'Editor',
    min: 1,
    max: 20,
    step: 1
  },
  {
    key: 'editor.history_length',
    label: 'History Length',
    type: 'number',
    description: 'Number of sentences to provide as history for the editor',
    category: 'Editor',
    min: 1,
    max: 20,
    step: 1
  },
  
  // Memory settings
  {
    key: 'MEMEORY.enable_local_knowledge',
    label: 'Local Knowledge',
    type: 'boolean',
    description: 'Enable local knowledge base',
    category: 'Memory'
  },
  {
    key: 'MEMEORY.enable_web_search',
    label: 'Web Search',
    type: 'boolean',
    description: 'Enable web search for additional context',
    category: 'Memory'
  },
  {
    key: 'MEMEORY.enable_vision_knowledge',
    label: 'Vision Knowledge',
    type: 'boolean',
    description: 'Enable vision-based knowledge extraction',
    category: 'Memory'
  },
  {
    key: 'MEMEORY.local_knowledge_path',
    label: 'Local Knowledge Path',
    type: 'readonly',
    description: 'Path to local knowledge base (system managed)',
    category: 'Memory',
    readonly: true
  },
  
  // Output settings
  {
    key: 'output_type.video',
    label: 'Generate Video',
    type: 'boolean',
    description: 'Generate video with embedded subtitles',
    category: 'Output'
  },
  {
    key: 'output_type.bilingual',
    label: 'Bilingual Subtitles',
    type: 'boolean',
    description: 'Create bilingual subtitles',
    category: 'Output'
  },
  {
    key: 'output_type.subtitle',
    label: 'Subtitle Format',
    type: 'select',
    description: 'Subtitle file format',
    category: 'Output',
    options: [
      { value: 'srt', label: 'SRT' },
      { value: 'ass', label: 'ASS' }
    ]
  },
  
  // System settings (readonly)
  {
    key: 'uploaded_file_path',
    label: 'Upload Path',
    type: 'readonly',
    description: 'File upload path (system managed)',
    category: 'System',
    readonly: true
  },
  {
    key: 'input_type',
    label: 'Input Type',
    type: 'readonly',
    description: 'Type of input (system detected)',
    category: 'System',
    readonly: true
  }
];

const getFileType = (filename: string): string => {
  const ext = filename.toLowerCase().split('.').pop();
  switch (ext) {
    case 'mp4':
    case 'avi':
    case 'mov':
    case 'mkv':
    case 'webm':
    case 'flv':
    case 'wmv':
      return 'video';
    case 'mp3':
    case 'wav':
    case 'flac':
    case 'aac':
    case 'm4a':
    case 'ogg':
      return 'audio';
    case 'srt':
    case 'ass':
    case 'vtt':
      return 'srt';
    default:
      return 'unknown';
  }
};

// Helper function to get value from config using dotted key
const getConfigValue = (config: SessionConfig, key: string): any => {
  // First try the key as-is (for dotted notation fields)
  let value = (config as any)[key];
  
  // If not found and key contains dots, try with underscores
  if (value === undefined && key.includes('.')) {
    const underscoreKey = key.replace(/\./g, '_');
    value = (config as any)[underscoreKey];
  }
  
  // If still not found and key doesn't contain dots, try with dots
  if (value === undefined && !key.includes('.')) {
    // This would be for keys like 'source_lang' that don't have dots
    value = (config as any)[key];
  }
  
  // Only log for translation model and instructions to debug specific issues
  if (key === 'translation.model') {
  }
  if (key === 'instructions') {
  }
  return value;
};

interface ConfigurationPanelProps {
  currentConfig: SessionConfig;
  uploadedFile: UploadFileResponse | null;
  youtubeUrlSubmitted: YouTubeUrlResponse | null;
  isLoading: boolean;
  onLaunchTask: () => void;
  onConfigUpdate?: (key: string, value: any) => void;
  sessionId?: string;
}

export const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({
  currentConfig,
  uploadedFile,
  youtubeUrlSubmitted,
  isLoading,
  onLaunchTask,
  onConfigUpdate,
  sessionId
}) => {
  const [activeCategory, setActiveCategory] = useState<string>('Language');
  const [localConfig, setLocalConfig] = useState<SessionConfig>(currentConfig);
  const prevInstructionsRef = useRef<string[] | null>(null);

  // Sync local config with prop changes
  useEffect(() => {
    const prevInstructions = prevInstructionsRef.current;
    const newInstructions = currentConfig.instructions;
    
    
    setLocalConfig(currentConfig);
    prevInstructionsRef.current = newInstructions || null;
  }, [currentConfig]);

  // Debug: Log only translation model changes
  useEffect(() => {
    const translationModel = (localConfig as any)['translation_model'] || localConfig['translation.model'];
    if (translationModel) {
      // Translation model changed - useful for debugging
    }
  }, [localConfig]);

  const hasInput = uploadedFile || youtubeUrlSubmitted;

  // Get unique categories
  const categories = Array.from(new Set(CONFIG_FIELDS.map(field => field.category)));

  const handleConfigChange = async (key: string, value: any) => {
    
    // Store the current config before making changes for potential rollback
    const originalConfig = { ...localConfig };
    
    // Update local config immediately for UI responsiveness
    const updatedConfig = { ...localConfig };
    
    // Set the value using the same key format that the backend expects/returns
    (updatedConfig as any)[key] = value;
    
    setLocalConfig(updatedConfig);

    // Call parent callback if provided
    if (onConfigUpdate && sessionId) {
      try {
        await onConfigUpdate(key, value);
      } catch (error) {
        console.error('ConfigurationPanel: Failed to update config:', error);
        // Revert local change on error
        setLocalConfig(originalConfig);
      }
    }
  };

  const renderConfigField = (field: ConfigField) => {
    const value = getConfigValue(localConfig, field.key);
    const isReadonly = field.readonly || field.type === 'readonly';
    
    // Only log for translation model to debug the specific issue
    if (field.key === 'translation.model') {
    }

    if (isReadonly) {
      return (
        <div key={field.key} className="config-field readonly">
          <label className="config-label">
            {field.label}
            <span className="readonly-badge">Read-only</span>
          </label>
          <div className="config-value readonly-value">
            {value?.toString() || 'Not set'}
          </div>
          <div className="config-description">{field.description}</div>
        </div>
      );
    }

    switch (field.type) {
      case 'select':
        return (
          <div key={field.key} className="config-field">
            <label className="config-label">{field.label}</label>
            <select
              className="config-input config-select"
              value={value?.toString() || ''}
              onChange={(e) => {
                const newValue = e.target.value;
                // Convert back to number if needed
                const convertedValue = field.options?.find(opt => opt.value.toString() === newValue)?.value || newValue;
                handleConfigChange(field.key, convertedValue);
              }}
            >
              {field.options?.map(option => (
                <option key={option.value.toString()} value={option.value.toString()}>
                  {option.label}
                </option>
              ))}
            </select>
            <div className="config-description">{field.description}</div>
          </div>
        );

      case 'boolean':
        return (
          <div key={field.key} className="config-field">
            <label className="config-label checkbox-label">
              <input
                type="checkbox"
                className="config-checkbox"
                checked={Boolean(value)}
                onChange={(e) => handleConfigChange(field.key, e.target.checked)}
              />
              {field.label}
            </label>
            <div className="config-description">{field.description}</div>
          </div>
        );

      case 'number':
        return (
          <div key={field.key} className="config-field">
            <label className="config-label">{field.label}</label>
            <input
              type="number"
              className="config-input config-number"
              value={value?.toString() || ''}
              min={field.min}
              max={field.max}
              step={field.step}
              onChange={(e) => {
                const newValue = parseFloat(e.target.value);
                if (!isNaN(newValue)) {
                  handleConfigChange(field.key, newValue);
                }
              }}
            />
            <div className="config-description">
              {field.description}
              {field.min !== undefined && field.max !== undefined && (
                <span className="range-info"> (Range: {field.min}-{field.max})</span>
              )}
            </div>
          </div>
        );

      case 'text-array':
        const arrayValue = Array.isArray(value) ? value : [];
        return (
          <div key={field.key} className="config-field">
            <label className="config-label">{field.label}</label>
            <div className="text-array-container">
              {arrayValue.map((item: string, index: number) => (
                <div key={index} className="text-array-item">
                  <input
                    type="text"
                    className="config-input text-array-input"
                    value={item}
                    placeholder={`Instruction ${index + 1}`}
                    onChange={(e) => {
                      const newArray = [...arrayValue];
                      newArray[index] = e.target.value;
                      handleConfigChange(field.key, newArray);
                    }}
                  />
                  <button
                    type="button"
                    className="remove-item-btn"
                    onClick={() => {
                      const newArray = arrayValue.filter((_: string, i: number) => i !== index);
                      handleConfigChange(field.key, newArray.length > 0 ? newArray : null);
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
              <button
                type="button"
                className="add-item-btn"
                onClick={() => {
                  const newArray = [...arrayValue, ''];
                  handleConfigChange(field.key, newArray);
                }}
              >
                + Add Instruction
              </button>
            </div>
            <div className="config-description">{field.description}</div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="config-panel">
      <h3>Configuration Settings</h3>
      
      <div className="config-content">
        {/* Category tabs */}
        <div className="config-categories">
          {categories.map(category => (
            <button
              key={category}
              className={`category-tab ${activeCategory === category ? 'active' : ''}`}
              onClick={() => setActiveCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>

        {/* Configuration fields for active category */}
        <div className="config-fields">
          {CONFIG_FIELDS
            .filter(field => field.category === activeCategory)
            .map(renderConfigField)}
        </div>
      </div>

      {hasInput && (
        <div className="task-creation">
          <h4>🚀 Ready to Launch!</h4>
          <p>Your configuration looks good. Ready to start the translation task?</p>
          
          <div className="upload-ready">
            {uploadedFile && (
              <div className="uploaded-file-info">
                <span className="file-icon">📁</span>
                <div className="file-details">
                  <div className="file-name">{uploadedFile.filename}</div>
                  <div className="file-type">
                    Type: {localConfig.input_type || getFileType(uploadedFile.filename)}
                  </div>
                </div>
                <span className="ready-status">✅ Ready</span>
              </div>
            )}
            
            {youtubeUrlSubmitted && (
              <div className="uploaded-file-info">
                <span className="file-icon">🎬</span>
                <div className="file-details">
                  <div className="file-name">YouTube Video</div>
                  <div className="file-type">
                    URL: {youtubeUrlSubmitted.youtube_url}
                  </div>
                </div>
                <span className="ready-status">✅ Ready</span>
              </div>
            )}
          </div>

          <button 
            onClick={onLaunchTask} 
            disabled={isLoading}
            className="create-task-button launch-button"
          >
            {isLoading ? 'Launching Task...' : '🚀 Launch Translation Task'}
          </button>
        </div>
      )}

      {!hasInput && (
        <div className="task-creation">
          <h4>📎 Input Required</h4>
          <p>Please upload a file or provide a YouTube URL to continue with the translation task.</p>
        </div>
      )}
    </div>
  );
};
