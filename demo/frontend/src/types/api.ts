export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ConfigurationValue {
  type: 'select' | 'boolean' | 'number';
  options?: (string | number)[];
  range?: [number, number];
  default: string | number | boolean;
  description: string;
}

export interface ConfigurationSchema {
  [key: string]: ConfigurationValue;
}

export interface SessionConfig {
  source_lang: "EN" | "ZH" | "ES" | "FR" | "DE" | "RU" | "JA" | "AR" | "KR";
  target_lang: "EN" | "ZH" | "ES" | "FR" | "DE" | "RU" | "JA" | "AR" | "KR";
  domain: "General" | "SC2" | "CS:GO";
  num_workers: number;
  'video_download.resolution': 360 | 480 | 720 | "best";
  'translation.model': "gpt-4" | "gpt-4o-mini" | "gpt-4o" | "gpt-5" | "gpt-5-mini";
  'translation.chunk_size': number;
  'translation.use_history': boolean;
  'translation.max_retries': number;
  'audio.enable_audio': boolean;
  'audio.audio_agent': "GeminiAudioAgent" | "WhisperAudioAgent" | "QwenAudioAgent" | "GPT4oAudioAgent";
  'audio.model_path': string | null;
  'audio.VAD_model': "pyannote/speaker-diarization-3.1" | "API";
  'audio.src_lang': string;
  'audio.tgt_lang': string;
  'vision.enable_vision': boolean;
  'vision.vision_model': "CLIP" | "gpt-4o" | "gpt-4o-mini";
  'vision.model_path': string;
  'vision.frame_cache_dir': string;
  'vision.frame_per_seg': number;
  'pre_process.sentence_form': boolean;
  'pre_process.spell_check': boolean;
  'pre_process.term_correct': boolean;
  'post_process.enable_post_process': boolean;
  'post_process.check_len_and_split': boolean;
  'post_process.remove_trans_punctuation': boolean;
  'proofreader.enable_proofreading': boolean;
  'proofreader.window_size': number;
  'proofreader.short_term_memory_len': number;
  'proofreader.enable_short_term_memory': boolean;
  'proofreader.verbose': boolean;
  'editor.enable_editor': boolean;
  'editor.user_instruction': "none" | "formal" | "casual" | "technical";
  'editor.editor_context_window': number;
  'editor.history_length': number;
  'MEMEORY.enable_local_knowledge': boolean;
  'MEMEORY.enable_web_search': boolean;
  'MEMEORY.enable_vision_knowledge': boolean;
  'MEMEORY.local_knowledge_path': string;
  'output_type.video': boolean;
  'output_type.bilingual': boolean;
  'output_type.subtitle': "srt" | "ass";
  // Custom user instructions for translation habits/jargon
  instructions?: string[];
  // File upload tracking
  uploaded_file_path?: string;
  uploaded_file_name?: string;
  youtube_url?: string;
  input_type?: 'youtube' | 'video' | 'audio' | 'srt';
}

export interface ConfigurationSession {
  session_id: string;
  messages: ChatMessage[];
  current_config: SessionConfig;
  is_complete: boolean;
}

export interface StartSessionResponse {
  session_id: string;
  message: string;
  current_config: SessionConfig;
}

export interface SendMessageResponse {
  message: string;
  config_updates: Partial<SessionConfig>;
  current_config: SessionConfig;
  is_complete: boolean;
  auto_created_task_id?: string;
}

export interface TaskRequest {
  session_id: string;
  input_type: 'youtube' | 'video' | 'audio' | 'srt';
  input_data: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'CREATED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  progress?: number;
  result_path?: string;
  error?: string;
}

export interface TaskInfo {
  task_id: string;
  status: string;
  created_at: string;
  input_type: string;
}

export interface CreateTaskResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface UploadFileResponse {
  filename: string;
  file_path: string;
  size: number;
}

export interface YouTubeUrlRequest {
  youtube_url: string;
}

export interface YouTubeUrlResponse {
  youtube_url: string;
  message: string;
}

export interface ConfigResponse {
  config: SessionConfig;
  is_complete?: boolean;
}

export interface FileDownloadInfo {
  filename: string;
  file_type: 'subtitle' | 'video' | 'log' | 'unknown';
  size_bytes: number;
  created_at: number;
}

export interface TaskResultResponse {
  task_id: string;
  status: string;
  has_results: boolean;
  files: FileDownloadInfo[];
  error?: string;
}

export interface ApiError {
  detail: string;
}

export interface AgentConversationMessage {
  role: string;
  message: string;
  timestamp?: string;
}

export interface AgentConversationResponse {
  task_id: string;
  conversation: AgentConversationMessage[];
  is_live: boolean;
}