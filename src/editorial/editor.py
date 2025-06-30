from typing import Callable, Dict, Optional
from src.SRT.srt import SrtScript, SrtSegment
from src.memory.abs_api_RAG import AbsApiRAG
import logging

class EditorAgent():
    def __init__(
        self,
        client,
        srt: SrtScript,
        memory: Optional[AbsApiRAG] = None,
        logger=None,
        history_len = 10,
        user_instruction:str = None
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
        self.history_len = history_len
        self.user_instruction = user_instruction
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None

    def set_agent_history_logger(self, logger):
        """Set the agent history logger from task"""
        self.agent_history_logger = logger

    def register_handler(self, name: str, func: Callable):
        """Register or replace a post-edit handler by name."""
        self.handlers[name] = func

    def build_prompt(self, idx: int, src_text: str, translation: str, ) -> str:
        """Construct an editing prompt for one subtitle segment."""
        # Multimodal context from SRT short-term memory
        seg = self.srt.segments[idx]
        suggestion = getattr(seg, 'suggestion', None)
        visual_ctx = getattr(seg, 'visual_cues', None)
        visual_ctx = "\n".join(visual_ctx) if visual_ctx else "None"
        audio_ctx = getattr(seg, 'audio_cues', None)
        audio_ctx = "\n".join(audio_ctx) if audio_ctx else "None"
        # Translation history up to this segment
        start = max(0, idx - self.history_len)
        end = min(len(self.srt.segments), idx + self.history_len + 1)
        prev = [s.translation for i, s in enumerate(self.srt.segments[start:]) if i + start != idx]
        past = [s.translation for i, s in enumerate(self.srt.segments[:end]) if i + start != idx]
        prev_translation_history = "\n".join(prev) if prev else "None"
        past_translation_history = "\n".join(past) if past else "None"
        ltm = []
        if self.memory:
            nodes = self.memory.retrieve_relevant_nodes(translation)
            ltm = [n.text for n in nodes]
        ltm = "\n".join(ltm) if ltm else "None"

        if self.user_instruction:
            user_instruction_str = self.user_instruction.replace("\n", "; ")
            self.agent_history_logger.info(f'{{"role": "editor", "message": "I received the following user instruction: {user_instruction_str}"}}')

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
        self.agent_history_logger.info('{"role": "editor", "message": "Time to sprinkle some editorial magic. Let\'s make it smooth as butter!"}')
        
        for idx, src, trans in self.srt_iterator():
            prompt = self.build_prompt(idx, src, trans)
            edits = self.send_request(prompt)
            self.srt.segments[idx].translation = edits.strip()
            self.logger.info(f"Edited segment {idx}: {edits.strip()}") if self.logger else None
            self.agent_history_logger.info(f'{{"role": "editor", "message": "Edited segment {idx}: {edits.strip()}"}}')
        
        self.agent_history_logger.info('{"role": "editor", "message": "All done! These lines are now as polished as my morning coffee mug."}')
