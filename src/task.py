import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from os import getenv
from pathlib import Path
from time import gmtime, strftime, time, sleep

# pytube deprecated
# from pytube import YouTube
import yt_dlp
from openai import AzureOpenAI, OpenAI
import os

from src.SRT.srt import SrtScript
from src.SRT.srt2ass import srt2ass
from src.memory.basic_rag import BasicRAG
from src.memory.direct_search_RAG import TavilySearchRAG
from src.translators.translator import Translator
from src.vision.gpt_vision_agent import GptVisionAgent, CLIPVisionAgent
from src.audio.audio_agent import (
    GeminiAudioAgent,
    WhisperAudioAgent,
    GPT4oAudioAgent,
    Qwen3ASRAudioAgent,
)
from src.editorial.editor import EditorAgent


class TaskStatus(str, Enum):
    """
    An enumeration class representing the different statuses a task can have in the translation pipeline.
    TODO: add translation progress indicator (%).
    """

    CREATED = "CREATED"
    INITIALIZING_ASR = "INITIALIZING_ASR"
    PRE_PROCESSING = "PRE_PROCESSING"
    TRANSLATING = "TRANSLATING"
    POST_PROCESSING = "POST_PROCESSING"
    OUTPUT_MODULE = "OUTPUT_MODULE"


class Task:
    """
    A class representing a task in the translation pipeline. It includes methods for handling different stages of the task.
    If one want to add a new entry type (e.g. add support for different video formats),
    one should extend this class and override the `run` method.
    """

    @property
    def status(self):
        with self.__status_lock:
            return self.__status

    @status.setter
    def status(self, new_status):
        """
        Sets the new status of the task, ensuring thread safety with a lock.
        """
        with self.__status_lock:
            self.__status = new_status

    def __init__(self, task_id, task_local_dir, task_cfg):
        """
        Constructor for initializing a task with its ID, local directory, and configuration settings.
        """
        self.__status_lock = threading.Lock()
        self.__status = TaskStatus.CREATED
        self.gpu_status = 0

        self.task_id = task_id

        self.base_dir = os.getcwd()
        self.task_local_dir = task_local_dir
        self.vision_setting = task_cfg["vision"]
        self.audio_setting = task_cfg["audio"]
        self.memory_setting = task_cfg["MEMEORY"]
        self.translation_setting = task_cfg["translation"]
        self.translation_model = self.translation_setting["model"]

        self.output_type = task_cfg["output_type"]
        self.target_lang = task_cfg["target_lang"]
        self.source_lang = task_cfg["source_lang"]
        self.domain = task_cfg["domain"]
        self.instructions = task_cfg.get("instructions", [])
        # Global workers control for VAD, proofreading, and editing
        self.num_workers = task_cfg.get("num_workers", 4)
        self.pre_setting = task_cfg["pre_process"]
        self.post_setting = task_cfg["post_process"]
        self.chunk_size = task_cfg["translation"]["chunk_size"]
        self.api_source = task_cfg["api_source"]

        self.proofreader_setting = task_cfg["proofreader"]
        self.editor_setting = task_cfg["editor"]

        self.audio_path = None
        self.SRT_Script = None
        self.result = None
        self.s_t = None
        self.t_e = None
        self.t_s = time()
        self.local_knowledge = None
        self.web_search = None
        self.vision_knowledge = None

        # logging setting
        self.task_logger = logging.getLogger(f"task_{task_id}")
        logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        self.task_logger.setLevel(logging.INFO)
        self.log_dir = "{}/{}_{}.log".format(
            task_local_dir, f"task_{task_id}", datetime.now().strftime("%m%d%Y_%H%M%S")
        )
        task_file_handler = logging.FileHandler(self.log_dir, "w", encoding="utf-8")
        task_file_handler.setFormatter(logging.Formatter(logfmt))
        self.task_logger.addHandler(task_file_handler)

        # log agent conversation history
        self.agent_history_logger = logging.getLogger(f"agent_history_{task_id}")
        self.agent_history_logger.setLevel(logging.INFO)
        agent_history_file_handler = logging.FileHandler(
            f"{self.task_local_dir}/agent_history.jsonl", "w", encoding="utf-8"
        )
        self.agent_history_logger.addHandler(agent_history_file_handler)
        # usage log path for per-request token usage events
        self.usage_log_path = f"{self.task_local_dir}/usage.jsonl"

        print(f"Task ID: {self.task_id}")
        self.task_logger.info(f"Task ID: {self.task_id}")

        if self.api_source == "openai":
            self.task_logger.info("Using OpenAI API")
            if "OPENAI_API_KEY" in task_cfg:
                self.task_logger.info("Using OPENAI_API_KEY from gradio interface.")
                self.oai_api_key = task_cfg["OPENAI_API_KEY"]
            else:
                self.task_logger.info("Using OPENAI_API_KEY from environment variable.")
                self.oai_api_key = getenv("OPENAI_API_KEY")
        elif self.api_source == "azure":
            self.task_logger.info("Using Azure OpenAI API")
            if "AZURE_OPENAI_API_KEY" in task_cfg:
                self.task_logger.info(
                    "Using AZURE_OPENAI_API_KEY from gradio interface."
                )
                self.oai_api_key = task_cfg["AZURE_OPENAI_API_KEY"]
            else:
                self.task_logger.info(
                    "Using AZURE_OPENAI_API_KEY from environment variable."
                )
                self.oai_api_key = getenv("AZURE_OPENAI_API_KEY")
        self.task_logger.info(
            f"{self.source_lang} -> {self.target_lang} task in {self.domain}"
        )
        self.task_logger.info(f"Translation Model: {self.translation_model}")
        self.task_logger.info(f"Chunk Size: {self.chunk_size}")
        self.task_logger.info(f"subtitle_type: {self.output_type['subtitle']}")
        self.task_logger.info(f"video_ouput: {self.output_type['video']}")
        self.task_logger.info(f"bilingual_ouput: {self.output_type['bilingual']}")
        self.task_logger.info("Pre-process setting:")
        for key in self.pre_setting:
            self.task_logger.info(f"{key}: {self.pre_setting[key]}")
        self.task_logger.info("Post-process setting:")
        for key in self.post_setting:
            self.task_logger.info(f"{key}: {self.post_setting[key]}")

        # init openai client
        if self.api_source == "openai":
            self.client = OpenAI(api_key=self.oai_api_key)
        elif self.api_source == "azure":
            self.client = AzureOpenAI(
                api_key=self.oai_api_key,
                azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-05-01-preview",
            )

        # init memory module
        if self.memory_setting["enable_local_knowledge"] and self.domain != "General":
            self.local_knowledge = BasicRAG(self.task_logger, self.domain)
            data_dir = f"{self.memory_setting['local_knowledge_path']}/{self.domain}"
            self.local_knowledge.load_knowledge_base(
                data_dir=data_dir, num_retrievals=10
            )

        if self.memory_setting["enable_web_search"]:
            # TODO: init web search
            self.web_search = TavilySearchRAG(self.task_logger, self.domain)

        if self.memory_setting["enable_vision_knowledge"]:
            self.vision_knowledge = BasicRAG(self.task_logger, "vision")
            self.vision_knowledge.load_knowledge_base(data_dir=None)

        # initialize translator
        self.translator = Translator(
            self.translation_model,
            self.source_lang,
            self.target_lang,
            self.domain,
            self.task_id,
            self.client,
            self.local_knowledge,
            self.web_search,
            self.vision_knowledge,
            self.chunk_size,
            usage_log_path=self.usage_log_path,
        )

        # initialize vision agent
        self.vision_agent = None
        if self.vision_setting["enable_vision"]:
            if self.vision_setting["vision_model"] == "CLIP":
                self.vision_agent = CLIPVisionAgent(
                    model_name=self.vision_setting["vision_model"],
                    model_path=self.vision_setting["model_path"]
                    if self.vision_setting["model_path"]
                    else None,
                    frame_per_seg=self.vision_setting["frame_per_seg"],
                    cache_dir=self.vision_setting["frame_cache_dir"],
                )
            elif self.vision_setting["vision_model"] in ("gpt-4o", "gpt-4o-mini"):
                self.vision_agent = GptVisionAgent(
                    model_name=self.vision_setting["vision_model"],
                    model_path=None,
                    frame_per_seg=self.vision_setting["frame_per_seg"],
                    cache_dir=self.vision_setting["frame_cache_dir"],
                )
                # Set agent history logger for vision agent
                if hasattr(self.vision_agent, "set_agent_history_logger"):
                    self.vision_agent.set_agent_history_logger(
                        self.agent_history_logger
                    )
                # Wire usage tracking context
                if hasattr(self.vision_agent, "set_task_id"):
                    self.vision_agent.set_task_id(self.task_id)
                if hasattr(self.vision_agent, "set_usage_log_path"):
                    self.vision_agent.set_usage_log_path(self.usage_log_path)
            else:
                raise ValueError(
                    f"Unsupported vision model: {self.vision_setting['vision_model']}"
                )

        self.audio_agent = None
        if self.audio_setting["enable_audio"]:
            agent_choice = self.audio_setting.get("audio_agent")
            audio_config = self.audio_setting.copy()
            audio_config["task_id"] = self.task_id
            audio_config["usage_log_path"] = self.usage_log_path
            if agent_choice == "GeminiAudioAgent":
                # Add task_id to audio_config for logger
                self.audio_agent = GeminiAudioAgent(audio_config=audio_config)
                self.task_logger.info(
                    f"Using GeminiAudioAgent with model: {self.audio_setting['audio_agent']}"
                )
                self.audio_agent.set_agent_history_logger(self.agent_history_logger)
            elif agent_choice == "WhisperAudioAgent":
                # Whisper audio agent that delegates to Whisper ASR and uses VAD when configured
                self.audio_agent = WhisperAudioAgent(
                    model_name="whisper-api", audio_config=audio_config
                )
                self.task_logger.info(
                    f"Using WhisperAudioAgent with model: {self.audio_setting['audio_agent']}"
                )
                self.audio_agent.set_agent_history_logger(self.agent_history_logger)
            elif agent_choice == "GPT4oAudioAgent":
                self.audio_agent = GPT4oAudioAgent(
                    model_name="gpt-4o", audio_config=audio_config
                )
                self.task_logger.info(
                    f"Using GPT4oAudioAgent with model: {self.audio_setting['audio_agent']}"
                )
                self.audio_agent.set_agent_history_logger(self.agent_history_logger)
            elif agent_choice == "Qwen3ASRAudioAgent":
                model_name = self.audio_setting.get("model_name", "qwen3-asr-flash")
                self.audio_agent = Qwen3ASRAudioAgent(
                    model_name=model_name, audio_config=audio_config
                )
                self.task_logger.info(
                    f"Using Qwen3ASRAudioAgent with model: {model_name}"
                )
                self.audio_agent.set_agent_history_logger(self.agent_history_logger)
            else:
                raise ValueError(f"Unsupported audio model: {agent_choice}")

        self.proofreader = None
        if self.proofreader_setting["enable_proofreading"]:
            from src.editorial.proofreader import ProofreaderAgent

            self.proofreader = ProofreaderAgent(
                client=self.client,
                srt=None,  # Will be set later
                local_knowledge=self.local_knowledge,
                web_search=self.web_search,
                logger=self.task_logger,
                batch_size=self.proofreader_setting["window_size"],
                stm_len=self.proofreader_setting["short_term_memory_len"],
                use_short_term_memory=self.proofreader_setting[
                    "enable_short_term_memory"
                ],
                num_workers=self.num_workers,
                usage_log_path=self.usage_log_path,
                task_id=self.task_id,
            )
            # Set agent history logger for proofreader
            if hasattr(self.proofreader, "set_agent_history_logger"):
                self.proofreader.set_agent_history_logger(self.agent_history_logger)
            self.task_logger.info("Proofreader initialized.")

        self.agent_history_logger.info(
            '{"role": "pipeline_coordinator", "message": "All modules initialized successfully. Task ready for execution."}'
        )

    @staticmethod
    def fromYoutubeLink(youtube_url, task_id, task_dir, task_cfg):
        """
        Creates a YoutubeTask instance from a YouTube URL.
        """
        return YoutubeTask(task_id, task_dir, task_cfg, youtube_url)

    @staticmethod
    def fromAudioFile(audio_path, task_id, task_dir, task_cfg):
        """
        Creates an AudioTask instance from an audio file path.
        """
        return AudioTask(task_id, task_dir, task_cfg, audio_path)

    @staticmethod
    def fromVideoFile(video_path, task_id, task_dir, task_cfg):
        """
        Creates a VideoTask instance from a video file path.
        """
        return VideoTask(task_id, task_dir, task_cfg, video_path)

    @staticmethod
    def fromSRTFile(srt_path, task_id, task_dir, task_cfg):
        """
        Creates a SRTTask instance from a srt file path.
        """
        return SRTTask(task_id, task_dir, task_cfg, srt_path)

    # Module 0: VAD: audio --> speaker segments
    def get_speaker_segments(self):
        """
        Handles the VAD module to convert audio to speaker segments.
        """
        self.SRT_Script = self.audio_agent.segment_audio(
            self.audio_path, f"{self.task_local_dir}/.cache/audio"
        )
        self.task_logger.info(
            f"Speaker segments created with {self.audio_agent.model_name} for audio: {self.audio_path}"
        )
        if self.video_path is not None and self.vision_agent is not None:
            self.audio_agent.clip_video_and_save(
                self.video_path, f"{self.task_local_dir}/.cache/video"
            )

    def get_visual_cues(self):
        """
        Handles the vision agent to convert video to visual cues.
        """
        if self.vision_agent is None:
            self.task_logger.info(
                "No vision agent found, skipping visual cues extraction"
            )
            return
        cache_dir = f"{self.task_local_dir}/.cache/video"
        if not os.path.isdir(cache_dir):
            self.task_logger.info(
                "No video segments found; skipping visual cues extraction"
            )
            return
        files = sorted(os.listdir(cache_dir))
        if not files:
            self.task_logger.info(
                "Video segments directory is empty; skipping visual cues extraction"
            )
            return
        self.task_logger.info(
            f"Extracting visual cues from video using {self.vision_agent.model_name}"
        )

        # Prepare jobs
        jobs = [
            (idx, f"{cache_dir}/{segment_file}")
            for idx, segment_file in enumerate(files)
        ]
        results = {}

        # Run in parallel across segments, controlled by global num_workers
        max_workers = (
            self.num_workers
            if isinstance(self.num_workers, int) and self.num_workers > 0
            else 1
        )

        def analyze_one(idx, path):
            # simple retry for robustness
            import random

            for attempt in range(3):
                try:
                    return self.vision_agent.analyze_video(path)
                except Exception as e:
                    backoff = min(10, 1 + attempt * 2) + random.random()
                    self.task_logger.warning(
                        f"Vision analysis failed for segment {idx} (attempt {attempt + 1}/3): {e}. Retrying in {backoff:.1f}s"
                    )
                    sleep(backoff)
            self.task_logger.error(
                f"Vision analysis failed for segment {idx} after retries; leaving empty."
            )
            return ""

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(analyze_one, idx, path): idx for idx, path in jobs
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        summary = future.result()
                    except Exception as e:
                        self.task_logger.error(f"Worker crashed for segment {idx}: {e}")
                        summary = ""
                    results[idx] = summary
        else:
            for idx, path in jobs:
                results[idx] = analyze_one(idx, path)

        # Apply results in order; update memory sequentially to avoid concurrency issues
        for idx in range(len(files)):
            visual_cues = results.get(idx, "")
            if self.vision_knowledge and visual_cues:
                self.vision_knowledge.add_to_index(
                    visual_cues, chunk_size=100, chunk_overlap=5
                )
            if idx < len(self.SRT_Script.segments):
                self.SRT_Script.segments[idx].visual_cues = visual_cues
            # print(self.vision_knowledge.retrieve_relevant_nodes("Protoss"))

    # Module 1 ASR: audio --> SRT_script
    def transcribe(self):
        """
        Perform ASR on each segment's audio, splitting into smaller segments if needed,
        but maintaining mapping to original segments.
        """
        self.temp_segments_info = []

        # Prepare batch items
        items = []
        idx_to_segment = {}
        for idx, segment in enumerate(self.SRT_Script.segments):
            if getattr(segment, "audio_path", None):
                items.append(
                    {
                        "idx": idx,
                        "audio_path": segment.audio_path,
                        "visual_cues": getattr(segment, "visual_cues", None),
                    }
                )
                idx_to_segment[idx] = segment
            else:
                self.task_logger.info("No audio file found for this segment.")

        if not items:
            self.task_logger.warning("No segments with audio to transcribe.")
            return

        # Use global workers by default; fall back to legacy audio.threads if set
        max_workers = self.num_workers
        if isinstance(self.audio_setting, dict):
            max_workers = self.audio_setting.get("threads", max_workers)

        # print("Number of audio segments to transcribe:", len(items))
        self.task_logger.info(f"Number of audio segments to transcribe: {len(items)}")

        # Execute concurrently if supported
        if hasattr(self.audio_agent, "transcribe_batch"):
            batch_results = (
                self.audio_agent.transcribe_batch(items, max_workers=max_workers) or {}
            )
        else:
            batch_results = {}
            for it in items:
                batch_results[it["idx"]] = (
                    self.audio_agent.transcribe(it["audio_path"], it.get("visual_cues"))
                    or []
                )

        # Normalize and offset per the original logic
        for idx, segment in idx_to_segment.items():
            temp_segment = batch_results.get(idx)
            if not temp_segment:
                self.task_logger.warning(
                    f"No transcription found for segment {idx}, skipping."
                )
                continue

            for idx_, seg in enumerate(temp_segment):
                if segment.timestr_to_seconds(
                    seg["start"]
                ) >= segment.timestr_to_seconds(seg["end"]):
                    if idx_ > 0:
                        self.task_logger.warning(
                            f"Segment {idx_} start time is >= end time, adjusting."
                        )
                        seg["start"] = temp_segment[idx_ - 1]["end"]
                    else:
                        self.task_logger.warning(
                            f"Segment {idx_} start time is >= end time, setting start=0."
                        )
                        seg["start"] = segment.format_time(0)

                if idx_ < len(temp_segment) - 1:
                    if segment.timestr_to_seconds(
                        seg["end"]
                    ) > segment.timestr_to_seconds(temp_segment[idx_ + 1]["start"]):
                        self.task_logger.warning(
                            f"Segment {idx_} end time > next start time, adjusting."
                        )
                        seg["end"] = temp_segment[idx_ + 1]["start"]

            for idx_, seg in enumerate(temp_segment):
                seg["start"] = (
                    segment.timestr_to_seconds(seg["start"]) + segment.start_time
                )
                seg["end"] = segment.timestr_to_seconds(seg["end"]) + segment.start_time

            new_segments = self.SRT_Script.convert_transcribed_segments(temp_segment)

            self.temp_segments_info.append(
                {"orig_idx": idx, "orig_segment": segment, "new_segments": new_segments}
            )

            self.task_logger.info(
                f"Transcribed Length: {len(new_segments)} for segment {idx}"
            )

        self.task_logger.info("Transcription completed, updating SRT script.")

        self.SRT_Script.replace_seg(self.temp_segments_info)

        # Save the transcribed SRT file
        results_dir = f"{self.task_local_dir}/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        transcribed_srt_path = f"{results_dir}/{self.task_id}_transcribed.srt"
        self.SRT_Script.write_srt_file_src(transcribed_srt_path)
        self.task_logger.info(f"Transcribed SRT saved to: {transcribed_srt_path}")

    # Module 2: SRT preprocess: perform preprocess steps
    def preprocess(self):
        """
        Performs preprocessing steps on the SRT script.
        """
        self.status = TaskStatus.PRE_PROCESSING
        self.task_logger.info(
            "--------------------Start Preprocessing SRT class--------------------"
        )
        if self.pre_setting["sentence_form"]:
            self.SRT_Script.form_whole_sentence()
        if self.pre_setting["spell_check"]:
            self.SRT_Script.spell_check_term()
        if self.pre_setting["term_correct"]:
            self.SRT_Script.correct_with_force_term()
        processed_srt_path_src = str(
            Path(self.task_local_dir) / f"{self.task_id}_processed.srt"
        )
        self.SRT_Script.write_srt_file_src(processed_srt_path_src)

        if self.output_type["subtitle"] == "ass":
            self.task_logger.info("write English .srt file to .ass")
            assSub_src = srt2ass(processed_srt_path_src, "default", "No", "Modest")
            self.task_logger.info("ASS subtitle saved as: " + assSub_src)
        self.script_input = self.SRT_Script.get_source_only()
        pass

    def update_translation_progress(self, new_progress):
        """
        (UNUSED)
        Updates the progress (%) of the translation process.
        """
        if self.progress == TaskStatus.TRANSLATING:
            self.progress = TaskStatus.TRANSLATING.value[0], new_progress

    # Module 3: perform srt translation
    def translation(self):
        """
        Handles the translation of the SRT script.
        """
        self.task_logger.info(
            "---------------------Start Translation--------------------"
        )
        self.translator.set_srt(self.SRT_Script)
        # Parallel translation is controlled globally by num_workers: > 1 enables parallel
        max_retries = self.translation_setting.get("max_retries", 2)
        use_history = self.translation_setting.get("use_history", True)
        if isinstance(self.num_workers, int) and self.num_workers > 1:
            self.task_logger.info(
                f"Using parallel translation controlled by num_workers={self.num_workers}; retries={max_retries}, use_history={use_history}"
            )
            self.translator.translate_parallel(
                max_workers=self.num_workers,
                max_retries=max_retries,
                use_history=use_history,
            )
        else:
            self.task_logger.info("Using sequential translation (num_workers <= 1)")
            self.translator.translate()

    # Module 4: perform srt post process steps
    def postprocess(self):
        """
        Performs post-processing steps on the translated SRT script.
        """
        self.status = TaskStatus.POST_PROCESSING

        self.task_logger.info(
            "---------------------Start Post-processing SRT class---------------------"
        )
        if self.post_setting["check_len_and_split"]:
            self.SRT_Script.check_len_and_split()
        if self.post_setting["remove_trans_punctuation"]:
            self.SRT_Script.remove_trans_punctuation()
        self.task_logger.info(
            "---------------------Post-processing SRT class finished---------------------"
        )

    def proofread(self):
        """
        Handles the proofreading of the translated SRT script.
        """
        self.status = TaskStatus.POST_PROCESSING
        self.task_logger.info(
            "---------------------Start Proofreading---------------------"
        )
        if self.proofreader is not None:
            self.proofreader.set_srt(self.SRT_Script)
            self.proofreader.proofread_all()
            self.SRT_Script = self.proofreader.srt
        else:
            self.task_logger.warning(
                "Proofreader is not initialized, skipping proofreading."
            )

    def editor(self):
        """
        Handles the editing of the translated SRT script.
        """
        editor = None
        if self.editor_setting["enable_editor"]:
            # Combine user instructions from config
            user_instructions = []
            if hasattr(self, "instructions") and self.instructions:
                user_instructions.extend(self.instructions)
            if self.editor_setting.get("user_instruction"):
                user_instructions.append(self.editor_setting["user_instruction"])

            combined_instruction = (
                "\n".join(user_instructions) if user_instructions else None
            )

            editor = EditorAgent(
                client=self.client,
                srt=self.SRT_Script,
                memory=self.local_knowledge,
                logger=self.task_logger,
                history_len=self.editor_setting["history_length"],
                user_instruction=combined_instruction,
                num_workers=self.num_workers,
                usage_log_path=self.usage_log_path,
                task_id=self.task_id,
            )
            # Set agent history logger for editor
            if hasattr(editor, "set_agent_history_logger"):
                editor.set_agent_history_logger(self.agent_history_logger)
        if editor is None:
            return
        editor.edit_all()

    def output_render(self):
        self.status = TaskStatus.OUTPUT_MODULE
        video_out = self.output_type["video"]
        subtitle_type = self.output_type["subtitle"]
        is_bilingual = self.output_type["bilingual"]

        results_dir = f"{self.task_local_dir}/results"

        # Always first save pure translation
        subtitle_path_trans = f"{results_dir}/{self.task_id}_{self.target_lang}.srt"
        self.SRT_Script.write_srt_file_translate(subtitle_path_trans)

        # Optionally save bilingual version
        if is_bilingual:
            subtitle_path_bilingual = f"{results_dir}/{self.task_id}_{self.source_lang}_{self.target_lang}.srt"
            self.SRT_Script.write_srt_file_bilingual(subtitle_path_bilingual)

        # Output ass file if needed
        if subtitle_type == "ass":
            ass_path = srt2ass(subtitle_path_trans, "default", "No", "Modest")
            final_res = ass_path
        else:
            final_res = subtitle_path_trans  # Always return pure translation

        # Output video if needed
        if video_out and self.video_path is not None:
            video_output_path = f"{results_dir}/{self.task_id}.mp4"

            # Check if fonts directory exists
            fonts_dir = f"{self.base_dir}/fonts"
            if not os.path.exists(fonts_dir):
                # Create fonts directory if it doesn't exist
                os.makedirs(fonts_dir, exist_ok=True)
                self.task_logger.warning(
                    f"Fonts directory {fonts_dir} does not exist, created it."
                )

            # Use proper ffmpeg subtitles filter syntax with fallback font
            try:
                # Try with custom font first
                subtitle_filter = f"subtitles='{final_res}':force_style='FontName=SourceHanSansCN-Normal,FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Bold=1'"

                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(self.video_path),
                        "-vf",
                        subtitle_filter,
                        "-c:a",
                        "copy",
                        video_output_path,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    self.task_logger.warning(
                        f"FFmpeg with custom font failed: {result.stderr}"
                    )
                    # Fallback to default font
                    subtitle_filter = f"subtitles='{final_res}':force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Bold=1'"

                    result = subprocess.run(
                        [
                            "ffmpeg",
                            "-i",
                            str(self.video_path),
                            "-vf",
                            subtitle_filter,
                            "-c:a",
                            "copy",
                            video_output_path,
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode != 0:
                        self.task_logger.error(
                            f"FFmpeg video generation failed: {result.stderr}"
                        )
                        raise RuntimeError(
                            f"Failed to generate video with subtitles: {result.stderr}"
                        )
                    else:
                        self.task_logger.info("Video generated with default font")
                else:
                    self.task_logger.info("Video generated with custom font")

            except Exception as e:
                self.task_logger.error(f"Error during video generation: {str(e)}")
                raise RuntimeError(f"Video generation failed: {str(e)}")

            final_res = subtitle_path_trans

        self.t_e = time()
        self.task_logger.info(
            "Pipeline finished, time duration:{}".format(
                strftime("%H:%M:%S", gmtime(self.t_e - self.t_s))
            )
        )
        return final_res

    def run_pipeline(self, pre_load_asr_model=None):
        """
        Executes the entire pipeline process for the task.
        """
        self.agent_history_logger.info(
            '{"role": "pipeline_coordinator", "message": "Starting ViDove translation pipeline..."}'
        )
        self.get_speaker_segments()
        self.get_visual_cues()
        self.transcribe()
        # self.preprocess()
        self.translation()

        # self.postprocess()
        self.proofread()
        self.editor()
        self.result = self.output_render()
        self.agent_history_logger.info(
            '{"role": "pipeline_coordinator", "message": "ViDove pipeline execution completed successfully!"}'
        )


class YoutubeTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, youtube_url):
        super().__init__(task_id, task_local_dir, task_cfg)
        self.task_logger.info("Task Creation method: Youtube Link")
        self.youtube_url = youtube_url
        self.video_resolution = task_cfg["video_download"]["resolution"]

        # self.model = model

    def run(self, pre_load_asr_model=None):
        self.task_logger.info(f"Youtube URL: {self.youtube_url}")
        self.task_logger.info(f"Video Resolution: {self.video_resolution}")
        video_download_path = f"{self.task_local_dir}/task_{self.task_id}.mp4"

        if self.video_resolution == "best":
            video_format = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio"
        elif self.video_resolution in [360, 480, 720]:
            video_format = f"bestvideo[height<={self.video_resolution}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={self.video_resolution}]+bestaudio"
        else:
            raise RuntimeError(f"Unsupported video resolution: {self.video_resolution}")

        video_opts = {
            "format": video_format,
            "outtmpl": video_download_path,
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            "prefer_ffmpeg": True,
        }

        # Download video only - we'll extract audio using ffmpeg
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            try:
                ydl.download([self.youtube_url])
            except yt_dlp.utils.DownloadError as e:
                self.task_logger.error(e)
                raise RuntimeError(f"Failed to download video {self.youtube_url}")
            ydl.close()

        # Extract audio from downloaded video using ffmpeg as 16k mono WAV for faster downstream clipping
        self.video_path = self.task_local_dir.joinpath(f"task_{self.task_id}.mp4")
        audio_path = self.task_local_dir.joinpath(f"task_{self.task_id}.wav")

        self.task_logger.info("using ffmpeg to extract audio from downloaded video")
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.video_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.task_logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
            raise RuntimeError(f"Failed to extract audio from video: {result.stderr}")

        self.task_logger.info("audio extraction finished")
        self.audio_path = audio_path

        self.task_logger.info(f" Video File Dir: {self.video_path}")
        self.task_logger.info(f" Audio File Dir: {self.audio_path}")
        self.task_logger.info(" Data Prep Complete. Start pipeline")

        super().run_pipeline(pre_load_asr_model)


class AudioTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, audio_path):
        super().__init__(task_id, task_local_dir, task_cfg)
        # TODO: check audio format
        self.task_logger.info("Task Creation method: Audio File")
        self.audio_path = audio_path
        self.video_path = None

    def run(self, pre_load_asr_model=None):
        self.task_logger.info(f"Video File Dir: {self.video_path}")
        self.task_logger.info(f"Audio File Dir: {self.audio_path}")
        self.task_logger.info("Data Prep Complete. Start pipeline")
        super().run_pipeline(pre_load_asr_model)


class VideoTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, video_path):
        super().__init__(task_id, task_local_dir, task_cfg)
        # TODO: check video format {.mp4}
        self.task_logger.info("Task Creation method: Video File")
        new_video_path = f"{task_local_dir}/task_{self.task_id}.mp4"
        self.task_logger.info(f"Copy video file to: {new_video_path}")
        shutil.copyfile(video_path, new_video_path)
        self.video_path = new_video_path

        # if self.video_path is not None and self.vision_agent is not None:
        #     self.visual_cues = self.vision_agent.analyze_video(self.video_path)
        # else:
        #     self.visual_cues = None

    def run(self, pre_load_asr_model=None):
        self.task_logger.info("using ffmpeg to extract audio (16k mono WAV)")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                self.video_path,
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                self.task_local_dir.joinpath(f"task_{self.task_id}.wav"),
            ]
        )
        self.task_logger.info("audio extraction finished")

        self.audio_path = self.task_local_dir.joinpath(f"task_{self.task_id}.wav")
        self.task_logger.info(f" Video File Dir: {self.video_path}")
        self.task_logger.info(f" Audio File Dir: {self.audio_path}")
        self.task_logger.info("Data Prep Complete. Start pipeline")
        super().run_pipeline(pre_load_asr_model)


class SRTTask(Task):
    def __init__(self, task_id, task_local_dir, task_cfg, srt_path):
        super().__init__(task_id, task_local_dir, task_cfg)
        self.task_logger.info("Task Creation method: SRT File")
        self.audio_path = None
        self.video_path = None
        new_srt_path = f"{task_local_dir}/task_{self.task_id}_{self.source_lang}.srt"
        self.task_logger.info(f"Copy video file to: {new_srt_path}")
        shutil.copyfile(srt_path, new_srt_path)
        self.SRT_Script = SrtScript.parse_from_srt_file(
            self.source_lang,
            self.target_lang,
            self.task_logger,
            self.client,
            domain=self.domain,
            path=srt_path,
        )

    def run(self):
        self.task_logger.info(f"Video File Dir: {self.video_path}")
        self.task_logger.info(f"Audio File Dir: {self.audio_path}")
        self.task_logger.info("Data Prep Complete. Start pipeline")
        super().run_pipeline()
