import os
import openai
from typing import Callable, Dict, Optional
from src.SRT.srt import SrtScript
from src.translators.abs_api_model import AbstractAPIModel
from src.memory.abs_api_RAG import AbsApiRAG

class ProofreaderAgent(AbstractAPIModel):
    def __init__(
        self,
        client,
        srt: SrtScript,
        local_knowledge: Optional[AbsApiRAG] = None,
        web_search: Optional[AbsApiRAG] = None,
        vision_knowledge: Optional[AbsApiRAG] = None,
        handlers: Optional[Dict[str, Callable]] = None,
        logger=None
    ):
        """
        client: an OpenAI or AzureOpenAI instance
        srt: your parsed SrtScript
        local_knowledge: BasicRAG for domain memory
        web_search: TavilySearchRAG for web memory
        vision_knowledge: BasicRAG for visual context
        handlers: dict[name -> function(idx, src, trans, suggestions)->suggestions]
        """
        super().__init__(client)
        self.srt = srt
        self.local_knowledge = local_knowledge
        self.web_search = web_search
        self.handlers = handlers or {}
        self.logger = logger
        self.client = client

    def register_handler(self, name: str, func: Callable):
        """Add or replace a handler by name."""
        self.handlers[name] = func

    def build_prompt(self, src_text: str, translation: str) -> str:
        """Construct a prompt for one subtitle segment."""
        # retrieve contexts if available
        local_ctx = "\n".join(
            n.text for n in self.local_knowledge.retrieve_relevant_nodes(translation)
        ) if self.local_knowledge else "None"
        web_ctx = "\n".join(
            n.text for n in self.web_search.retrieve_relevant_nodes(translation)
        ) if self.web_search else "None"


        return f"""You are a proofreader agent for translation tasks.
                Refine the translation below from {self.srt.src_lang} → {self.srt.tgt_lang} in domain `{self.srt.domain}`.

                Source text:
                {src_text}

                Translated text:
                {translation}

                When writing suggestions, focus on:
                1. Accuracy (fix additions, omissions, mistranslations)
                2. Fluency (grammar, spelling, punctuation, no repetitions)
                3. Terminology (consistent domain terms, correct idioms)

                List specific corrections; if perfect, reply `PASS`.

                ---
                **Local memory context:**
                {local_ctx}

                **Web memory context:**
                {web_ctx}
                """

    def send_request(self, prompt: str):
        """Calls the OpenAI Chat API and returns the assistant's reply."""
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content

    def srt_iterator(self):
        """Yield (index, src_text, translation) for each subtitle segment."""
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation

    def apply_handlers(self, idx: int, src: str, trans: str, suggestions: str):
        """
        Pass suggestions through each registered handler in turn.
        Each handler must accept (idx, src, trans, suggestions) and return new suggestions.
        """
        for name, handler in self.handlers.items():
            suggestions = handler(idx=idx, src=src, trans=trans, suggestions=suggestions)
            if suggestions is not None:
                self.srt.segments[idx].translation = suggestions
            self.logger.info(f"Handler {name} applied to segment {idx}: {suggestions}")

    def proofread_all(self):
        """
        Proofread every segment and run through handlers.
        Returns a dict mapping segment index → final suggestions.
        """
        for idx, src, trans in self.srt_iterator():
            prompt = self.build_prompt(src, trans)
            suggestions = self.send_request(prompt)
            self.apply_handlers(idx, src, trans, suggestions)
