from typing import Callable, Dict, Optional, List
from src.SRT.srt import SrtScript
from src.memory.abs_api_RAG import AbsApiRAG
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from uuid import uuid4
from datetime import datetime


class EditorAgent:
    def __init__(
        self,
        client,
        srt: SrtScript,
        memory: Optional[AbsApiRAG] = None,
        logger: Optional[logging.Logger] = None,
        history_len: int = 10,
        user_instruction: Optional[str] = None,
        num_workers: int = 4,
        usage_log_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        self.client = client
        self.srt = srt
        self.memory = memory
        self.logger = logger
        self.history_len = history_len
        self.user_instruction = user_instruction
        self.num_workers = max(1, int(num_workers))
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None
        # Optional post-edit handlers registry
        self.handlers: Dict[str, Callable] = {}
        # Lock for thread-safe writes
        self._lock = Lock()
        # Usage logging context
        self.usage_log_path = usage_log_path
        self.task_id = task_id

    def set_agent_history_logger(self, logger):
        self.agent_history_logger = logger

    def register_handler(self, name: str, func: Callable):
        self.handlers[name] = func

    def set_usage_log_path(self, path: Optional[str]):
        self.usage_log_path = path

    def set_task_id(self, task_id: Optional[str]):
        self.task_id = task_id

    def _record_usage(
        self,
        *,
        provider: str,
        model: str,
        category: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        phrase_index: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> None:
        if not self.usage_log_path:
            return
        try:
            rec = {
                "request_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "task_id": self.task_id,
                "provider": provider,
                "model": model,
                "category": category,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "phrase_index": phrase_index,
            }
            if extra:
                rec.update({"extra": extra})
            with open(self.usage_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _snapshot_translations(self) -> List[str]:
        return [seg.translation for seg in self.srt.segments]

    def build_prompt(
        self,
        idx: int,
        src_text: str,
        translation: str,
        base_translations: Optional[List[str]] = None,
    ) -> str:
        seg = self.srt.segments[idx]
        suggestion = getattr(seg, "suggestion", None)
        visual_ctx = getattr(seg, "visual_cues", None)
        visual_ctx = "\n".join(visual_ctx) if visual_ctx else "None"
        audio_ctx = getattr(seg, "audio_cues", None)
        audio_ctx = "\n".join(audio_ctx) if audio_ctx else "None"

        translations = (
            base_translations if base_translations is not None else self._snapshot_translations()
        )
        n = len(translations)
        prev_indices = range(max(0, idx - self.history_len), idx)
        next_indices = range(idx + 1, min(n, idx + self.history_len + 1))
        prev = [translations[i] for i in prev_indices]
        past = [translations[i] for i in next_indices]
        prev_translation_history = "\n".join(prev) if prev else "None"
        past_translation_history = "\n".join(past) if past else "None"

        ltm = []
        if self.memory:
            try:
                nodes = self.memory.retrieve_relevant_nodes(translation)
                ltm = [n.text for n in nodes if getattr(n, "text", None)]
            except Exception:
                ltm = []
        ltm = "\n".join(ltm) if ltm else "None"

        if self.user_instruction and self.agent_history_logger:
            try:
                user_instruction_str = self.user_instruction.replace("\n", "; ")
                self.agent_history_logger.info(
                    json.dumps(
                        {
                            "role": "editor",
                            "message": f"I received the following user instruction: {user_instruction_str}",
                        }
                    )
                )
            except Exception:
                pass

        return f"""You are an Editor ensuring overall translation quality and coherence,
                aligning the translation with the original video content in domain `{self.srt.domain}`, you must ensure the term and style are aligned with the domain's language.
        
                Segment index: {idx}
                Source text:
                {src_text}

                Translated text:
                {translation}

                Here is a provided suggestion for each segment, which may or may not useful for your revision, you may use the suggestion only if necessary (for example, term correctness).
                Note that the suggestion may not be accurate, the proofreader has less information comparing to you, so you need to double check before making revision.
                The proofreader may return "UNCLEAR" if they are not sure about the translation, they will specify the location and you need to check with other information provided to you to solve for unclear.
                If there is no suggestions, you may ignore this part, but still check with other modality context and long-term memory for correctness and coherence.
                Suggestion:
                                {suggestion if suggestion else "No suggestion provided."}
                
                Your edit will also follow the following instruction if provided:
                User instruction:
                {self.user_instruction if self.user_instruction else "No user instruction provided."}                
                
                --- Multimodal Context (Short-Term Memory) ---
                Visual cues:
                You may use visual cues from the video to improve translation or make corrections, the source text might not be accurate, you need to check with the video context if provided:
                {visual_ctx}

                Audio cues:
                {audio_ctx}

                Translation context:
                You will be provided with the previous and next 5 segments' translations, which may help you understand the context and make corrections:
                Previous translation history (up to 5 segments):
                {prev_translation_history}
                Past translation history (up to 5 segments):
                {past_translation_history}

                --- Long-Term Memory ---
                Long-term memory provides broader context and domain-specific knowledge, you may use it to improve translation or make corrections:
                {ltm}

                Notice:
                1. Corrections or adjustments to better align text with the video context.
                2. Suggestions for improving coherence across segments.
                3. Logical consistency and any broader context adjustments.
                4. Ensure the translation is accurate and aligned with the domain `{self.srt.domain}`.
                5. Ensure translation is smooth and fluent across segments.
                6. To ensure the fluency in {self.srt.tgt_lang}, you do not have to ensure translation be word by word accurate, but be sure to convey the same information.

                --- Important ---
                Directly return the revised content only."""

    def send_request(self, prompt: str, phrase_index: Optional[int] = None) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3000,
        )
        # Best-effort usage logging
        try:
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            tt = getattr(usage, "total_tokens", None) if usage else None
            self._record_usage(
                provider="openai",
                model="gpt-4o",
                category="text",
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                phrase_index=phrase_index,
                extra={"agent": "editor"},
            )
        except Exception:
            pass
        return resp.choices[0].message.content

    def srt_iterator(self):
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation

    def edit_all(self) -> Dict[int, str]:
        if self.agent_history_logger:
            try:
                self.agent_history_logger.info(
                    json.dumps(
                        {
                            "role": "editor",
                            "message": "Time to sprinkle some editorial magic. Let us make it smooth as butter!",
                        }
                    )
                )
            except Exception:
                pass

        snapshot_translations = self._snapshot_translations()
        results: Dict[int, str] = {}

        def worker(item):
            idx, src, trans = item
            prompt = self.build_prompt(
                idx, src, trans, base_translations=snapshot_translations
            )
            edits = self.send_request(prompt, phrase_index=idx).strip()
            return idx, edits

        items = list(self.srt_iterator())
        if self.num_workers == 1:
            for item in items:
                idx, edits = worker(item)
                with self._lock:
                    self.srt.segments[idx].translation = edits
                    results[idx] = edits
                if self.logger:
                    self.logger.info(f"Edited segment {idx}: {edits}")
                if self.agent_history_logger:
                    try:
                        self.agent_history_logger.info(
                            json.dumps(
                                {
                                    "role": "editor",
                                    "message": f"Edited segment {idx}: {edits}",
                                }
                            )
                        )
                    except Exception:
                        pass
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                future_map = {ex.submit(worker, item): item[0] for item in items}
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        i, edits = fut.result()
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Editing segment {idx} failed: {e}")
                        if self.agent_history_logger:
                            try:
                                self.agent_history_logger.info(
                                    json.dumps(
                                        {
                                            "role": "editor",
                                            "message": f"Editing segment {idx} failed: {e}",
                                        }
                                    )
                                )
                            except Exception:
                                pass
                        continue
                    with self._lock:
                        self.srt.segments[i].translation = edits
                        results[i] = edits
                    if self.logger:
                        self.logger.info(f"Edited segment {i}: {edits}")
                    if self.agent_history_logger:
                        try:
                            self.agent_history_logger.info(
                                json.dumps(
                                    {
                                        "role": "editor",
                                        "message": f"Edited segment {i}: {edits}",
                                    }
                                )
                            )
                        except Exception:
                            pass

        if self.agent_history_logger:
            try:
                self.agent_history_logger.info(
                    json.dumps(
                        {
                            "role": "editor",
                            "message": "All done! These lines are now as polished as my morning coffee mug.",
                        }
                    )
                )
            except Exception:
                pass
        return results
