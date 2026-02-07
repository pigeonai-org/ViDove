import json
import logging
import os
import traceback
import warnings
from collections import deque
from datetime import datetime
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import openai
from llama_index.core import PromptTemplate
from tqdm import tqdm

try:  # Optional dependency for memory telemetry
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from src.memory.basic_rag import BasicRAG
from src.memory.direct_search_RAG import TavilySearchRAG
from src.SRT.srt import split_script
from src.translators.assistant import Assistant
from src.translators.LLM import LLM
from src.translators.MTA import MTA

from .prompts import system_prompt
from .prompts import get_input_prompt

SUPPORT_LANG_MAP = {
    "EN": "English",
    "ZH": "Chinese",
    "ES": "Spanish",
    "FR": "France",
    "DE": "Germany",
    "RU": "Russian",
    "JA": "Japanese",
    "AR": "Arabic",
    "KR": "Korean",
}


class Translator:
    def __init__(
        self, model_name, src_lang, tgt_lang, domain, task_id, client, local_knowledge:BasicRAG=None, web_search:TavilySearchRAG=None, vision_knowledge:BasicRAG=None, chunk_size=1000, usage_log_path: str | None = None,
    ):
        self.task_logger = logging.getLogger(f"task_{task_id}")
        self.task_logger.info("initializing translator")

        self.agent_history_logger = logging.getLogger(f"agent_history_{task_id}")
        self.agent_history_logger.setLevel(logging.INFO)

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.domain = domain
        self.task_id = task_id
        self.system_prompt = self.prompt_selector()
        self.client = client
        self.local_knowledge = local_knowledge
        self.web_search = web_search
        self.vision_knowledge = vision_knowledge
        self.translation_history = []
        self.srt = None
        self.usage_log_path = usage_log_path

        self.summary_interval = self._resolve_summary_interval()
        self._processed_chunks = 0
        self._recent_translations = deque(maxlen=3)

        if self.summary_interval > 0:
            self.task_logger.info(
                f"Live translation summaries enabled every {self.summary_interval} chunks"
            )
        else:
            self.task_logger.info("Live translation summaries disabled")

        if self.model_name == "Assistant":
            self.translator = Assistant(
                self.client, system_prompt=self.system_prompt, domain=domain
            )
        elif self.model_name in ["gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            self.translator = LLM(
                self.client, self.model_name, system_prompt=self.system_prompt, task_id=self.task_id, usage_log_path=self.usage_log_path
            )
        elif self.model_name == "Multiagent":
            self.translator = MTA(
                self.client,
                "gpt-4o",
                self.domain,
                self.src_lang,
                self.tgt_lang,
                SUPPORT_LANG_MAP[self.tgt_lang],
                self.task_logger,
            )
        elif self.model_name == "RAG":
            self.translator = BasicRAG(
                self.task_logger,
                self.domain,
                self.model_name,
            )
        else:
            print(f"Unsupported model name: {self.model_name}")
            raise NotImplementedError

        self.task_logger.info(f"Using {self.model_name} as translation model")

    def set_srt(self, srt):
        self.srt = srt
        self.script_arr, self.range_arr = split_script(
            srt.get_source_only(), self.chunk_size
        )
        self.task_logger.info("SRT file set")
        self.agent_history_logger.info('{"role": "translator", "message": "Got the SRT! Time to flex my translation muscles! 💪"}')

    def prompt_selector(self) -> PromptTemplate:
        try:
            src_lang = SUPPORT_LANG_MAP[self.src_lang]
            tgt_lang = SUPPORT_LANG_MAP[self.tgt_lang]
            assert src_lang != tgt_lang
        except Exception:
            print("Unsupported language, is your abbreviation correct?")
            print(f"supported language map: {SUPPORT_LANG_MAP}")
            self.task_logger.info(
                f"Unsupported language detected: {self.src_lang} to {self.tgt_lang}"
            )

        prompt = PromptTemplate(system_prompt).format(
            domain=self.domain,
            source_language=src_lang,
            target_language=tgt_lang,
        )
        
        self.task_logger.info(f"System Prompt: {prompt}")
        return prompt

    def translate(self, max_retries = 1):
        """
        Translates the given script array into another language using the chatgpt and writes to the SRT file.
, 
        This function takes a script array, a range array, a model name, a video name, and a video link as input. It iterates
        through sentences and range in the script and range arrays. If the translation check fails for five times, the function
        will attempt to resolve merge sentence issues and split the sentence into smaller tokens for a better translation.
        """

        if self.srt is None:
            raise ValueError("SRT file not set")

        if self.system_prompt is None:
            self.system_prompt = "你是一个翻译助理，你的任务是翻译视频，你会被提供一个按行分割的英文段落，你需要在保证句意和行数的情况下输出翻译后的文本。"
            self.task_logger.info(f"translation prompt: {self.system_prompt}")
        
        self.agent_history_logger.info('{"role": "translator", "message": "Starting translation process with knowledge retrieval... "}')
        
        previous_length = 0
        for sentence, range_ in tqdm(zip(self.script_arr, self.range_arr)):
            # update the range based on previous length
            range_ = (range_[0] + previous_length, range_[1] + previous_length)

            print(f"now translating sentences {range_}")
            self.task_logger.info(f"now translating sentences {range_}")
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # add knowledge retrieve before translation
                    # TODO: disable the knowledge if assistant is active
                    # convert nodes to string

                    input_dict = {}
                    if len(self.translation_history) != 0:
                        input_dict["history_str"] = "\n".join(self.translation_history[-5:])
                    if self.local_knowledge is not None:
                        input_dict["context_str"] = self.local_knowledge.retrieve_relevant_nodes(sentence)
                    if self.web_search is not None:
                        input_dict["supporting_documents"] = self.web_search.retrieve_relevant_nodes(sentence)
                    if self.vision_knowledge is not None:
                        input_dict["video_clips_description"] = self.vision_knowledge.retrieve_relevant_nodes(sentence)
                    input_dict["query_str"] = sentence

                    input = get_input_prompt(self.domain, self.src_lang, self.tgt_lang, input_dict)

                    translation = self.translator.send_request(input)
                    self.translation_history.append(translation)
                    break  # Success - exit the loop
                except openai.BadRequestError as e:
                    retry_count += 1
                    # Access the content filter results
                    error_response = e.response.json()
                    filter_results = error_response['error']['innererror']['content_filter_result']
                    
                    # Extract categories where filtered is True
                    filtered_categories = [
                        category for category, details in filter_results.items()
                        if details.get('filtered') is True
                    ]
                    
                    # Optionally, you can also get categories with their severity
                    filtered_with_severity = {
                        category: details['severity']
                        for category, details in filter_results.items()
                        if details.get('filtered') is True
                    }
                    print(f"An error has occurred during translation (attempt {retry_count}/{max_retries}):", e)

                    print(traceback.format_exc())
                    self.task_logger.debug("An error has occurred during translation:", e)
                    
                    if retry_count < max_retries:
                        self.task_logger.info(
                            "Retrying... the script will continue after 30 seconds."
                        )
                        sleep(30)
                    else:
                        self.task_logger.warning(f"Max retries ({max_retries}) reached, skipping translation for: {sentence}")
                        self.task_logger.warning(f"Filtered categories: {' '.join(filtered_categories)} with severity: {' '.join(filtered_with_severity)}")
                        warnings.warn(f"Max retries ({max_retries}) reached, skipping translation for: {sentence}, please check if the video contains any {filtered_categories}")
                        translation = ""
                        
            self.task_logger.info(f"source text: {sentence}")
            self.task_logger.info(f"translate text: {translation}")
            self.srt.set_translation(translation, range_, self.model_name, self.task_id)
            self._log_progress_snapshot(range_, translation)
        
        self.agent_history_logger.info('{"role": "translator", "message": "Whew, translation marathon complete! If you spot a typo, it was totally intentional..."}')

    def translate_parallel(self, max_workers: int = 4, max_retries: int = 2, use_history: bool = True):
        """
        Parallel translation for API-based models (e.g., LLM) using a thread pool.

        Guarantees output ordering by collecting results and applying them in original chunk order.
        - max_workers: number of parallel threads
        - max_retries: retries per chunk on transient errors (e.g., rate limits)
        - use_history: whether to include recent translation history per request
                        (off by default to avoid cross-thread coupling)
        """
        if self.srt is None:
            raise ValueError("SRT file not set")

        # Only enable parallelism for API LLM-style models we know are stateless per request
        api_models = {"gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"}
        if self.model_name not in api_models:
            self.task_logger.info(
                f"Model {self.model_name} is not an API LLM; falling back to sequential translate()."
            )
            return self.translate(max_retries=max_retries)

        if self.system_prompt is None:
            self.system_prompt = "你是一个翻译助理，你的任务是翻译视频，你会被提供一个按行分割的英文段落，你需要在保证句意和行数的情况下输出翻译后的文本。"
            self.task_logger.info(f"translation prompt: {self.system_prompt}")

        self.agent_history_logger.info('{"role": "translator", "message": "Starting parallel translation with knowledge retrieval..."}')

        # Build job list with absolute index ranges (like sequential logic)
        jobs = []  # (idx, sentence, (start, end))
        previous_length = 0
        for idx, (sentence, range_) in enumerate(zip(self.script_arr, self.range_arr)):
            abs_range = (range_[0] + previous_length, range_[1] + previous_length)
            jobs.append((idx, sentence, abs_range))

        if not jobs:
            self.task_logger.info("No chunks to translate; skipping.")
            return

        # local helper: build input with optional knowledge; do not mutate shared state
        def build_input_for(sentence: str, idx: int):
            input_dict = {}
            if use_history and len(self.translation_history) != 0:
                input_dict["history_str"] = "\n".join(self.translation_history[-5:])
            if self.local_knowledge is not None:
                input_dict["context_str"] = self.local_knowledge.retrieve_relevant_nodes(sentence)
            if self.web_search is not None:
                input_dict["supporting_documents"] = self.web_search.retrieve_relevant_nodes(sentence)
            if self.vision_knowledge is not None:
                input_dict["video_clips_description"] = self.vision_knowledge.retrieve_relevant_nodes(sentence)
            input_dict["query_str"] = sentence
            return get_input_prompt(self.domain, self.src_lang, self.tgt_lang, input_dict)

        def translate_one(idx: int, sentence: str):
            retry_count = 0
            last_err = None
            while retry_count <= max_retries:
                try:
                    prompt_input = build_input_for(sentence, idx)
                    translation = self.translator.send_request(prompt_input)
                    return translation
                except openai.BadRequestError as e:
                    # content filter or invalid request; do not retry beyond configured
                    retry_count += 1
                    last_err = e
                    try:
                        error_response = e.response.json()
                        filter_results = error_response['error']['innererror']['content_filter_result']
                        filtered_categories = [
                            category for category, details in filter_results.items()
                            if details.get('filtered') is True
                        ]
                        filtered_with_severity = {
                            category: details['severity']
                            for category, details in filter_results.items()
                            if details.get('filtered') is True
                        }
                        self.task_logger.warning(
                            f"BadRequestError on idx={idx}. Filtered: {filtered_categories} severity={filtered_with_severity}"
                        )
                    except Exception:
                        # best-effort parsing
                        pass
                    if retry_count <= max_retries:
                        # short backoff
                        sleep(2)
                except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.InternalServerError, TimeoutError) as e:
                    retry_count += 1
                    last_err = e
                    backoff = min(10, 1 + retry_count * 2) + random.random()
                    self.task_logger.info(
                        f"Transient error on idx={idx} (attempt {retry_count}/{max_retries}). Backing off {backoff:.1f}s"
                    )
                    sleep(backoff)
                except Exception as e:
                    # Unknown error: do one retry then give up
                    retry_count += 1
                    last_err = e
                    self.task_logger.debug(
                        f"Unexpected error for idx={idx}: {e}\n{traceback.format_exc()}"
                    )
                    sleep(1)
            # Max retries exceeded
            warnings.warn(f"Max retries reached, skipping translation for chunk {idx}")
            if last_err is not None:
                self.task_logger.warning(f"Failed to translate chunk {idx}: {last_err}")
            return ""

        results: dict[int, tuple[str, tuple[int, int]]] = {}

        self.task_logger.info(f"Submitting {len(jobs)} translation chunks with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(translate_one, idx, sentence): (idx, abs_range)
                          for idx, sentence, abs_range in jobs}

            for future in as_completed(future_map):
                idx, abs_range = future_map[future]
                try:
                    translation = future.result()
                except Exception as e:
                    self.task_logger.warning(f"Worker crashed on chunk {idx}: {e}")
                    translation = ""
                results[idx] = (translation, abs_range)
                # optional: update translation_history in submission order only when use_history
                if use_history and translation:
                    self.translation_history.append(translation)

        # Apply results in order to keep deterministic output
        for idx in sorted(results.keys()):
            translation, abs_range = results[idx]
            sentence = self.script_arr[idx]
            self.task_logger.info(f"source text: {sentence}")
            self.task_logger.info(f"translate text: {translation}")
            self.srt.set_translation(translation, abs_range, self.model_name, self.task_id)
            self._log_progress_snapshot(abs_range, translation)

        self.agent_history_logger.info('{"role": "translator", "message": "Parallel translation complete. All chunks processed."}')

    def _resolve_summary_interval(self) -> int:
        """Determine how frequently to emit live translation summaries."""
        default_interval = 5
        env_value = os.getenv("VIDOVE_LIVE_SUMMARY_INTERVAL")
        if env_value:
            try:
                interval = int(env_value)
                if interval > 0:
                    return interval
                self.task_logger.warning(
                    "VIDOVE_LIVE_SUMMARY_INTERVAL must be a positive integer; using default"
                )
            except ValueError:
                self.task_logger.warning(
                    "Invalid VIDOVE_LIVE_SUMMARY_INTERVAL value; using default"
                )
        return default_interval

    def _log_progress_snapshot(self, abs_range, translation: str | None) -> None:
        """Emit a progress message into the agent history at configured intervals."""
        self._processed_chunks += 1

        if translation:
            self._recent_translations.append(translation.strip())

        if self.summary_interval <= 0 or self._processed_chunks % self.summary_interval != 0:
            return

        memory_meta = {}
        memory_message = "Memory stats unavailable"
        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
                process_percent = round(process.memory_percent(), 2)
                system_info = psutil.virtual_memory()
                system_percent = round(system_info.percent, 2)
                memory_meta = {
                    "process_rss_mb": rss_mb,
                    "process_percent": process_percent,
                    "system_percent": system_percent,
                }
                memory_message = (
                    f"Memory usage: {rss_mb} MB RSS (process {process_percent}% / system {system_percent}%)"
                )
            except Exception:
                # Memory inspection is best-effort; keep UI responsive even if psutil is missing.
                pass

        recent_preview = [self._truncate_text(t) for t in self._recent_translations if t]
        if recent_preview:
            summary_preview = " | ".join(recent_preview)
        else:
            summary_preview = "No translations recorded yet."

        range_text = "unknown lines"
        if isinstance(abs_range, tuple) and len(abs_range) == 2:
            range_text = f"lines {abs_range[0]}-{abs_range[1]}"

        payload = {
            "role": "translation_monitor",
            "timestamp": f"{datetime.utcnow().isoformat(timespec='seconds')}Z",
            "message": (
                f"Progress checkpoint after {self._processed_chunks} chunks ({range_text}). "
                f"{memory_message} Recent output: {summary_preview}"
            ),
            "processed_chunks": self._processed_chunks,
        }

        if memory_meta:
            payload["memory"] = memory_meta

        if recent_preview:
            payload["recent_translations"] = recent_preview

        self.agent_history_logger.info(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _truncate_text(text: str, limit: int = 120) -> str:
        """Trim translation text for compact status summaries."""
        clean_text = text.replace("\n", " ").strip()
        if len(clean_text) <= limit:
            return clean_text
        return f"{clean_text[:limit-3]}..."