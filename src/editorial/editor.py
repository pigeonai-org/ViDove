from typing import Callable, Dict, Optional
from src.SRT.srt import SrtScript, SrtSegment
from src.memory.abs_api_RAG import AbsApiRAG

class EditorAgent():
    def __init__(
        self,
        client,
        srt: SrtScript,
        memory: Optional[AbsApiRAG] = None,
        logger=None
    ):
        """
        client: an OpenAI or AzureOpenAI instance
        srt: your parsed SrtScript
        memory: TavilySearchRAG (long-term memory)
        """
        self.client = client
        self.srt = srt
        self.memory = memory
        self.logger = logger

    def register_handler(self, name: str, func: Callable):
        """Register or replace a post-edit handler by name."""
        self.handlers[name] = func

    def build_prompt(self, idx: int, src_text: str, translation: str) -> str:
        """Construct an editing prompt for one subtitle segment."""
        # Multimodal context from SRT short-term memory
        seg = self.srt.segments[idx]
        suggestion = getattr(seg, 'suggestion', None)
        visual_ctx = getattr(seg, 'visual_cues', None)
        visual_ctx = "\n".join(visual_ctx) if visual_ctx else "None"
        audio_ctx = getattr(seg, 'audio_cues', None)
        audio_ctx = "\n".join(audio_ctx) if audio_ctx else "None"
        # Translation history up to this segment
        start = max(0, idx - 5)
        end = min(len(self.srt.segments), idx + 6)
        history = [s.translation for i, s in enumerate(self.srt.segments[start:end]) if i + start != idx]
        translation_history = "\n".join(history) if history else "None"
        ltm = []
        if self.memory:
            nodes = self.memory.retrieve_relevant_nodes(translation)
            ltm = [n.text for n in nodes]
        ltm = "\n".join(ltm) if ltm else "None"

        return f"""You are an Editor ensuring overall translation quality and coherence,
                aligning the translation with the original video content in domain `{self.srt.domain}`, you need to ensure the term and style are aligned with the domain's language.
        
                Segment index: {idx}
                Source text:
                {src_text}

                Translated text:
                {translation}

                Here is a provided suggestion for each segment, which may or may not useful for your revision.
                                {suggestion if suggestion else "No suggestion provided."}

                --- Multimodal Context (Short-Term Memory) ---
                Visual cues:
                You may use visual cues from the video to improve translation or make corrections:
                {visual_ctx}

                Audio cues:
                {audio_ctx}

                Translation history (previous segments):
                {translation_history}

                --- Long-Term Memory (Styling Knowledge) ---
                Long-term memory provides broader context and domain-specific knowledge, you may use it to improve translation or make corrections:
                {ltm}

                Please provide:
                1. Corrections or adjustments to better align text with the video context.
                2. Suggestions for improving coherence across segments.
                3. Logical consistency and any broader context adjustments.

                Directly return the revised content only."""

    def send_request(self, prompt: str) -> str:
        """Calls the LLM and returns the model's response."""
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3000,
        )
        return resp.choices[0].message.content

    def srt_iterator(self):
        """Yield (index, src_text, translation) for each subtitle segment."""
        for idx, seg in enumerate(self.srt.segments):
            yield idx, seg.src_text, seg.translation


    def edit_all(self) -> Dict[int, str]:
        """Edit every segment and apply handlers. Returns dict index->final edits."""
        for idx, src, trans in self.srt_iterator():
            prompt = self.build_prompt(idx, src, trans)
            edits = self.send_request(prompt)
            self.srt.segments[idx].translation = edits.strip()
            self.logger.info(f"Edited segment {idx}: {edits.strip()}") if self.logger else None
