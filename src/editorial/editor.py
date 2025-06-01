from typing import Callable, Dict, Optional
from src.SRT.srt import SrtScript
from src.translators.abs_api_model import AbstractAPIModel
from src.memory.abs_api_RAG import AbsApiRAG

class EditorAgent(AbstractAPIModel):
    def __init__(
        self,
        client,
        srt: SrtScript,
        memory: Optional[AbsApiRAG] = None,
        handlers: Optional[Dict[str, Callable]] = None,
    ):
        """
        client: an OpenAI or AzureOpenAI instance
        srt: your parsed SrtScript
        memory: TavilySearchRAG (long-term memory)
        handlers: dict[name -> function(idx, src, trans, suggestions) -> suggestions]
        """
        super().__init__(client)
        self.srt = srt
        self.memory = memory
        self.handlers = handlers or {}

    def register_handler(self, name: str, func: Callable):
        """Register or replace a post-edit handler by name."""
        self.handlers[name] = func

    def build_prompt(self, idx: int, src_text: str, translation: str) -> str:
        """Construct an editing prompt for one subtitle segment."""
        # Multimodal context from SRT short-term memory
        seg = self.srt.segments[idx]
        visual_ctx = getattr(seg, 'visual_cues', None)
        visual_ctx = "\n".join(visual_ctx) if visual_ctx else "None"
        audio_ctx = getattr(seg, 'audio_cues', None)
        audio_ctx = "\n".join(audio_ctx) if audio_ctx else "None"
        # Translation history up to this segment
        history = [s.translation for s in self.srt.segments[:idx]]
        translation_history = "\n".join(history) if history else "None"
        # Long-term web knowledge
        ltm = []
        if self.memory:
            nodes = self.memory.retrieve_relevant_nodes(translation)
            ltm = [n.text for n in nodes]
        ltm = "\n".join(ltm) if ltm else "None"

        return f"""You are an Editor Agent (L_ed) ensuring overall quality and coherence,
                aligning the translation with the original video content in domain `{self.srt.domain}`.

                Segment index: {idx}
                Source text:
                {src_text}

                Translated text:
                {translation}

                --- Multimodal Context (Short-Term Memory) ---
                Visual cues:
                {visual_ctx}

                Audio cues:
                {audio_ctx}

                Translation history (previous segments):
                {translation_history}

                {ltm}

                Please provide:
                1. Corrections or adjustments to better align text with the video context.
                2. Suggestions for improving coherence across segments.
                3. Logical consistency and any broader context adjustments.

                Return a numbered list of edits. If no edits are needed, reply with `PASS`."""

    def send_request(self, prompt: str) -> str:
        """Calls the LLM and returns the model's response."""
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        return resp.choices[0].message.content

    def srt_iterator(self):
        """Yield (index, src_text, translation) for each subtitle segment."""
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation

    def apply_handlers(self, idx: int, src: str, trans: str, suggestions: str) -> str:
        """
        Pass suggestions through each registered handler in turn.
        Each handler should accept (idx, src, trans, suggestions) and return new suggestions.
        """
        for name, handler in self.handlers.items():
            suggestions = handler(idx=idx, src=src, trans=trans, suggestions=suggestions)
        return suggestions

    def edit_all(self) -> Dict[int, str]:
        """Edit every segment and apply handlers. Returns dict index->final edits."""
        results = {}
        for idx, src, trans in self.srt_iterator():
            prompt = self.build_prompt(idx, src, trans)
            edits = self.send_request(prompt)
            final = self.apply_handlers(idx, src, trans, edits)
            results[idx] = final
        return results
