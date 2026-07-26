from typing import Dict, List, Optional, Tuple
import re
from src.SRT.srt import SrtScript
from src.memory.abs_api_RAG import AbsApiRAG
from src.openai_responses import (
    DEFAULT_TEXT_MODEL,
    create_response_text,
    extract_usage_tokens,
    normalize_text_model,
    provider_for_client,
)
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
        model_name: str = DEFAULT_TEXT_MODEL,
        user_instruction: Optional[str] = None,
        num_workers: int = 4,
        batch_size: int = 8,
        usage_log_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        self.client = client
        self.srt = srt
        self.memory = memory
        self.logger = logger
        self.history_len = history_len
        self.model_name = normalize_text_model(model_name)
        self.user_instruction = user_instruction
        self.num_workers = max(1, int(num_workers))
        self.batch_size = max(1, int(batch_size))
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None
        # Lock for thread-safe writes
        self._lock = Lock()
        # Usage logging context
        self.usage_log_path = usage_log_path
        self.task_id = task_id

    def set_agent_history_logger(self, logger):
        self.agent_history_logger = logger

    def _record_usage(
        self,
        *,
        provider: str,
        model: str,
        category: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        cached_prompt_tokens: Optional[int] = None,
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
                "cached_prompt_tokens": cached_prompt_tokens,
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

    def _log_history(self, message: str) -> None:
        if self.agent_history_logger:
            try:
                self.agent_history_logger.info(
                    json.dumps({"role": "editor", "message": message})
                )
            except Exception:
                pass

    def _snapshot_translations(self) -> List[str]:
        return [seg.translation for seg in self.srt.segments]

    def _retrieve_long_term_memory(self, query: str) -> str:
        if not self.memory or not query.strip():
            return "None"
        try:
            nodes = self.memory.retrieve_relevant_nodes(query)
            ltm = [n.text for n in nodes if getattr(n, "text", None)]
            return "\n".join(ltm) if ltm else "None"
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Editor memory retrieval failed: {e}")
            return "None"

    def build_batch_prompt(
        self,
        batch: List[Tuple[int, str, str]],
        base_translations: List[str],
    ) -> str:
        n = len(base_translations)
        first_idx = batch[0][0]
        last_idx = batch[-1][0]

        segment_blocks = []
        for idx, src_text, translation in batch:
            seg = self.srt.segments[idx]
            suggestion = getattr(seg, "suggestion", None)
            visual_ctx = getattr(seg, "visual_cues", None)
            visual_ctx = "\n".join(visual_ctx) if visual_ctx else "None"
            audio_ctx = getattr(seg, "audio_cues", None)
            audio_ctx = "\n".join(audio_ctx) if audio_ctx else "None"
            segment_blocks.append(
                f"Segment {idx}:\n"
                f"Source text: {src_text}\n"
                f"Translated text: {translation}\n"
                f"Proofreader suggestion: {suggestion if suggestion else 'None'}\n"
                f"Visual cues: {visual_ctx}\n"
                f"Audio cues: {audio_ctx}\n"
            )
        segments_text = "\n".join(segment_blocks)

        prev_indices = range(max(0, first_idx - self.history_len), first_idx)
        next_indices = range(last_idx + 1, min(n, last_idx + self.history_len + 1))
        prev = [base_translations[i] for i in prev_indices]
        upcoming = [base_translations[i] for i in next_indices]
        prev_translation_history = "\n".join(prev) if prev else "None"
        next_translation_history = "\n".join(upcoming) if upcoming else "None"

        ltm = self._retrieve_long_term_memory(
            " ".join(translation for _, _, translation in batch)
        )

        return f"""You are an Editor ensuring overall translation quality and coherence,
                aligning the translation with the original video content in domain `{self.srt.domain}`, you must ensure the term and style are aligned with the domain's language.

                Below are {len(batch)} consecutive subtitle segments. For each one, revise the translated text for accuracy, fluency, and coherence across segments.

                Proofreader suggestions may or may not be useful; use them only if necessary (for example, term correctness).
                The proofreader has less information than you, so double check before making a revision.
                The proofreader may return "UNCLEAR" if they are not sure about the translation; check the other information provided to you to resolve it.
                The source text might not be accurate; check the visual/audio cues if provided.

                Your edit will also follow the following instruction if provided:
                User instruction:
                {self.user_instruction if self.user_instruction else "No user instruction provided."}

                --- Segments to edit ---
                {segments_text}

                --- Translation context ---
                Previous translation history:
                {prev_translation_history}
                Upcoming translation history:
                {next_translation_history}

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
                Return EXACTLY one line per segment, in this format and nothing else:
                Segment {first_idx}: <revised translation>
                Segment {first_idx + 1 if len(batch) > 1 else first_idx}: <revised translation>
                ...
                Each revised translation must be on a single line. Do not add explanations."""

    def send_request(self, prompt: str, phrase_index: Optional[int] = None) -> str:
        text, resp = create_response_text(
            self.client,
            model=self.model_name,
            input_value=prompt,
        )
        # Best-effort usage logging
        try:
            pt, ct, tt, cpt = extract_usage_tokens(resp)
            self._record_usage(
                provider=provider_for_client(self.client),
                model=self.model_name,
                category="text",
                prompt_tokens=pt,
                cached_prompt_tokens=cpt,
                completion_tokens=ct,
                total_tokens=tt,
                phrase_index=phrase_index,
                extra={"agent": "editor"},
            )
        except Exception:
            pass
        return text

    @staticmethod
    def _parse_batch_response(content: str) -> Dict[int, str]:
        """Parse 'Segment <idx>: <text>' blocks into {idx: text}.

        Lines that do not start a new segment are treated as continuations of
        the current one, so wrapped revisions are not silently truncated.
        """
        edits: Dict[int, str] = {}
        if not content:
            return edits
        current_idx: Optional[int] = None
        current_lines: List[str] = []

        def flush():
            if current_idx is not None:
                text = "\n".join(current_lines).strip()
                if text:
                    edits[current_idx] = text

        for line in content.splitlines():
            match = re.match(r"\s*Segment\s+(\d+)\s*[:：]\s*(.*)", line)
            if match:
                flush()
                current_idx = int(match.group(1))
                current_lines = [match.group(2).strip()]
            elif current_idx is not None:
                current_lines.append(line.strip())
        flush()
        return edits

    def srt_iterator(self):
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation

    def edit_all(self) -> Dict[int, str]:
        self._log_history(
            "Time to sprinkle some editorial magic. Let us make it smooth as butter!"
        )
        if self.user_instruction:
            self._log_history(
                "I received the following user instruction: "
                + self.user_instruction.replace("\n", "; ")
            )

        snapshot_translations = self._snapshot_translations()
        results: Dict[int, str] = {}

        items = list(self.srt_iterator())
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        def apply_edits(batch, edits: Dict[int, str]):
            for idx, _, original in batch:
                revised = edits.get(idx, "").strip()
                if not revised:
                    # Keep the existing translation rather than blanking it out.
                    if self.logger:
                        self.logger.warning(
                            f"Editor returned no revision for segment {idx}; keeping original."
                        )
                    continue
                with self._lock:
                    self.srt.segments[idx].translation = revised
                    results[idx] = revised
                if self.logger:
                    self.logger.info(f"Edited segment {idx}: {revised}")
                self._log_history(f"Edited segment {idx}: {revised}")

        def process_batch(batch):
            prompt = self.build_batch_prompt(batch, snapshot_translations)
            phrase_index = batch[0][0] if batch else None
            content = self.send_request(prompt, phrase_index=phrase_index)
            edits = self._parse_batch_response(content)
            batch_indices = {idx for idx, _, _ in batch}
            unknown = set(edits) - batch_indices
            if unknown and self.logger:
                self.logger.warning(
                    f"Editor response labeled segments {sorted(unknown)} outside batch "
                    f"{sorted(batch_indices)}; those lines are ignored."
                )
            apply_edits(batch, edits)

        if self.num_workers == 1 or len(batches) == 1:
            for batch in batches:
                try:
                    process_batch(batch)
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            f"Editing batch starting at segment {batch[0][0]} failed: {e}; keeping originals."
                        )
                    self._log_history(f"Editing batch failed, keeping originals: {e}")
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                future_map = {ex.submit(process_batch, b): b for b in batches}
                for fut in as_completed(future_map):
                    batch = future_map[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        if self.logger:
                            self.logger.error(
                                f"Editing batch starting at segment {batch[0][0]} failed: {e}; keeping originals."
                            )
                        self._log_history(
                            f"Editing batch failed, keeping originals: {e}"
                        )

        self._log_history(
            "All done! These lines are now as polished as my morning coffee mug."
        )
        return results
