from typing import Optional, List, Tuple
from src.SRT.srt import SrtScript
from src.memory.abs_api_RAG import AbsApiRAG
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from uuid import uuid4
from datetime import datetime

class ProofreaderAgent:
    def __init__(
        self,
        client,
        srt: SrtScript,
        local_knowledge: Optional[AbsApiRAG] = None,
        web_search: Optional[AbsApiRAG] = None,
        logger=None,
        batch_size: int = 5,
        stm_len: int = 10,
        use_short_term_memory: bool = False,
        verbose: int = 2,
        num_workers: int = 4,
        usage_log_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        self.client = client
        self.srt = srt
        self.local_knowledge = local_knowledge
        self.web_search = web_search
        self.logger = logger
        self.batch_size = batch_size
        self.stm_len = stm_len
        self.verbose = verbose
        self.use_short_term_memory = use_short_term_memory
        self.short_term_memory = ""
        self.num_workers = max(1, int(num_workers))
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None
        # Lock for thread-safe SRT writes
        self._lock = Lock()
        # Usage logging context
        self.usage_log_path = usage_log_path
        self.task_id = task_id

    def set_agent_history_logger(self, logger):
        """Set the agent history logger from task"""
        self.agent_history_logger = logger

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

    def set_srt(self, srt: SrtScript):
        self.srt = srt
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "proofreader", "message": "Alright, let me put on my red pen and nitpick these lines. No mercy! ✏️"}')

    def srt_iterator(self):
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation

    def conclude_to_stm(self, translation: str):
        self.short_term_memory = self.send_request(
            f"Briefly conclude the content: '{translation}' with context memory: '{self.short_term_memory}'."
        )
        if self.verbose > 1 and self.logger:
            self.logger.info(f"Updated STM: {self.short_term_memory}")

    def send_request(self, prompt: str, phrase_index: Optional[int] = None):
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
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
                extra={"agent": "proofreader"},
            )
        except Exception:
            pass
        return resp.choices[0].message.content

    def build_batch_prompt(self, batch: List[Tuple[int, str, str]]) -> str:
        segment_block = []
        for idx, src, trans in batch:
            segment_block.append(f"""Segment {idx}:
                                    Source: {src}
                                    Translation: {trans}
                                    """)
        segments_text = "\n".join(segment_block)

        local_ctx = []
        
        for b in batch:
            local_ctx.append("\n".join(
                [n.text for n in self.local_knowledge.retrieve_relevant_nodes(b[1]) if n.text]
            ) if self.local_knowledge else "None")
        
        # DEBUG
        if self.logger:
            self.logger.info('sentences in batch:')
            self.logger.info(" ".join(s for _, s, _ in batch))
            self.logger.info(f"Local context for batch: {local_ctx}") if self.logger else None
            self.logger.info(f"Local context for batch: {local_ctx}")

        web_ctx = "\n".join(
            n.text for n in self.web_search.retrieve_relevant_nodes(
                " ".join(s for _, s, _ in batch)
            )
        ) if self.web_search else "None"

        return f"""You are a translation proofreader. Below are {len(batch)} subtitle segments.
                Some are full sentences, some are fragments. Give **specific advice** for each one, 
                but do not treat each segment separatly you need information across segment.

                Return suggestions in this format:
                Segment 0: [your comment here]
                Segment 1: [your comment here]
                ...

                DO NOT return JSON. DO NOT rewrite the translation. Just return suggestion texts.

                ---
                {segments_text}

                **Short-term memory:**
                {self.short_term_memory}

                **Term context:**
                {local_ctx}

                **Web memory context:**
                {web_ctx}

                Focus on:
                1. Translation accuracy while stick to domain({self.srt.domain}) (missing or incorrect meanings)
                2. Fluency (grammar, spelling, repetition. Only if it affects understanding) and ensure the translation is smooth and fluent across segments.
                3. Terminology (Use term context to edit idioms, ensure every sentence is translated into domain-specific language)
                4. If you have no suggestions, return "PASS" for that segment.
                5. Source text isn't 100% accurate. If you have doubt about the source text, return "UNCLEAR" and specify the location, editor will check the issue.
                6. Only make suggestion if you believe revision is necessary.
                """

    def proofread_all(self):
        if self.agent_history_logger:
            try:
                self.agent_history_logger.info(json.dumps({
                    "role": "proofreader",
                    "message": "Let's see what we've got here... time to hunt for typos and awkward phrasing!",
                }))
            except Exception:
                pass

        segments = list(self.srt_iterator())

        def process_batch(batch):
            prompt = self.build_batch_prompt(batch)
            if self.logger:
                self.logger.info(f"Prompting LLM for segments {[idx for idx, _, _ in batch]}")
            if self.verbose > 1 and self.logger:
                self.logger.info(f"Prompt content:\n{prompt}")
            # Use the first segment index in the batch as phrase_index for attribution
            phrase_index = batch[0][0] if batch else None
            content = self.send_request(prompt, phrase_index=phrase_index)
            lines = content.strip().splitlines()
            suggestions = {}
            for line in lines:
                if line.startswith("Segment "):
                    try:
                        prefix, suggestion = line.split(":", 1)
                        if self.verbose > 0 and self.logger:
                            self.logger.info(f"Processing suggestion for {prefix.strip()}: {suggestion.strip()}")
                        if self.agent_history_logger:
                            try:
                                self.agent_history_logger.info(json.dumps({
                                    "role": "proofreader",
                                    "message": f"My suggestion for {prefix.strip()}: {suggestion.strip()}",
                                }))
                            except Exception:
                                pass
                        idx = int(prefix.replace("Segment ", "").strip())
                        suggestions[idx] = suggestion.strip()
                    except Exception:
                        continue
            # Apply suggestions
            for idx, _, trans in batch:
                suggestion = suggestions.get(idx, "PASS")
                if self.use_short_term_memory:
                    try:
                        with self._lock:
                            self.conclude_to_stm(trans)
                    except Exception:
                        pass
                if suggestion != "PASS":
                    with self._lock:
                        self.srt.segments[idx].suggestion = suggestion
                    if self.logger:
                        self.logger.info(f"Added suggestion for segment {idx}")

        # Build batches
        batches = [segments[i:i + self.batch_size] for i in range(0, len(segments), self.batch_size)]
        if self.num_workers == 1:
            for b in batches:
                process_batch(b)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                futures = [ex.submit(process_batch, b) for b in batches]
                for _ in as_completed(futures):
                    pass

        if self.agent_history_logger:
            try:
                self.agent_history_logger.info(json.dumps({
                    "role": "proofreader",
                    "message": "Proofreading done! If I missed anything, it must be perfect already 😏",
                }))
            except Exception:
                pass
                    