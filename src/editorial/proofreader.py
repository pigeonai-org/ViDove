import os
import openai
from typing import Callable, Dict, Optional, List, Tuple
from src.SRT.srt import SrtScript
from src.memory.abs_api_RAG import AbsApiRAG

class ProofreaderAgent():
    def __init__(
        self,
        client,
        srt: SrtScript,
        local_knowledge: Optional[AbsApiRAG] = None,
        web_search: Optional[AbsApiRAG] = None,
        handlers: Optional[Dict[str, Callable]] = None,
        logger=None,
        batch_size: int = 5,
        stm_len: int = 10,
        verbose: int = 2
    ):
        self.client = client
        self.srt = srt
        self.local_knowledge = local_knowledge
        self.web_search = web_search
        self.handlers = handlers or {}
        self.logger = logger
        self.batch_size = batch_size
        self.stm_len = stm_len
        self.verbose = verbose
        self.short_term_memory = ""

    def set_srt(self, srt: SrtScript):
        self.srt = srt

    def register_handler(self, name: str, func: Callable):
        self.handlers[name] = func

    def srt_iterator(self):
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation

    def apply_handlers(self, idx: int, src: str, trans: str, suggestion: str):
        for name, handler in self.handlers.items():
            updated = handler(idx=idx, src=src, trans=trans, suggestions=suggestion)
            if updated is not None:
                self.srt.segments[idx].translation = updated
            if self.verbose > 0 and self.logger:
                self.logger.info(f"Handler {name} applied to segment {idx}: {updated}")

    def conclude_to_stm(self, translation: str):
        self.short_term_memory = self.send_request(
            f"Briefly conclude this new translation: '{translation}' with context memory: '{self.short_term_memory}'. Focus on names, terminology, key info."
        )
        if self.verbose > 1 and self.logger:
            self.logger.info(f"Updated STM: {self.short_term_memory}")

    def send_request(self, prompt: str):
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
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
        print('sentences in batch:')
        print(" ".join(s for _, s, _ in batch))
        self.logger.info(f"Local context for batch: {local_ctx}") if self.logger else None

        web_ctx = "\n".join(
            n.text for n in self.web_search.retrieve_relevant_nodes(
                " ".join(s for _, s, _ in batch)
            )
        ) if self.web_search else "None"

        return f"""You are a translation proofreader. Below are {len(batch)} subtitle segments.
                Some are full sentences, some are fragments. Give **specific advice** for each one.

                Return suggestions in this format:
                Segment 0: [your comment here]
                Segment 1: [your comment here]
                ...

                DO NOT return JSON. DO NOT rewrite the translation. Just return suggestion texts.

                ---
                {segments_text}

                **Short-term memory:**
                {self.short_term_memory}

                **Local memory context:**
                {local_ctx}

                **Web memory context:**
                {web_ctx}

                Focus on:
                1. Translation accuracy (missing or incorrect meanings)
                2. Fluency (grammar, spelling, repetition)
                3. Terminology (idioms, domain-specific language)
                4. If you have no suggestions, return "PASS" for that segment.
                """

    def proofread_all(self):
        segments = list(self.srt_iterator())
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            prompt = self.build_batch_prompt(batch)

            if self.logger:
                self.logger.info(f"Prompting LLM for segments {[idx for idx, _, _ in batch]}")

            if self.verbose > 1 and self.logger:
                self.logger.info(f"Prompt content:\n{prompt}")

            content = self.send_request(prompt)

            # Parse suggestions back line-by-line
            lines = content.strip().splitlines()
            suggestions = {}
            for line in lines:
                if line.startswith("Segment "):
                    try:
                        prefix, suggestion = line.split(":", 1)
                        if self.verbose > 0 and self.logger:
                            self.logger.info(f"Processing suggestion for {prefix.strip()}: {suggestion.strip()}")
                        idx = int(prefix.replace("Segment ", "").strip())
                        suggestions[idx] = suggestion.strip()
                    except Exception:
                        continue

            for idx, src, trans in batch:
                suggestion = suggestions.get(idx, "PASS")
                if suggestion != "PASS":
                    self.apply_handlers(idx, src, trans, suggestion)
                    self.conclude_to_stm(trans)
                    
        return self.short_term_memory