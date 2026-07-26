import base64

from src.openai_responses import extract_usage_tokens, get_response_text
from src.vision.vision_agent import VisionAgent
from openai import OpenAI

# Optional import - loaded lazily when needed
cv2 = None


def _ensure_cv2():
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    return cv2


class GptVisionAgent(VisionAgent):
    def __init__(self, model_name, model_path, frame_per_seg, cache_dir=None):
        super().__init__(model_name, model_path, frame_per_seg, cache_dir)
        self.client = OpenAI()
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None

    def set_agent_history_logger(self, logger):
        """Set the agent history logger from task"""
        self.agent_history_logger = logger

    def load_model(self, model_path=None):
        # API-backed agent; nothing to load locally.
        return None

    def encode_image_to_base64(self, image):
        _ensure_cv2()
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image

    def describe_frames(self, encoded_images: list[str]) -> str:
        """Describe all frames of a segment in a single multi-image request."""
        model_name = self.model_name if self.model_name in ("gpt-4o", "gpt-4o-mini") else "gpt-4o"
        content = [
            {
                "type": "input_text",
                "text": (
                    "These frames were sampled in order from one video segment. "
                    "Describe the visual content: the scene, recognizable objects, "
                    "on-screen text, and what is happening. Reply with one concise summary."
                ),
            }
        ]
        for encoded_image in encoded_images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                }
            )
        response = self.client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": content}],
        )
        # Best-effort usage logging
        try:
            pt, ct, tt, cpt = extract_usage_tokens(response)
            self._record_usage(
                provider="openai",
                model=model_name,
                prompt_tokens=pt,
                cached_prompt_tokens=cpt,
                completion_tokens=ct,
                total_tokens=tt,
                phrase_index=None,
                extra={},
            )
        except Exception:
            pass
        return get_response_text(response)

    def _extract_frames(self, video_path):
        """Sample up to frame_per_seg frames from the segment without decoding to EOF."""
        _ensure_cv2()
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return []

            if total_frames < self.frame_per_seg:
                extract_indices = list(range(total_frames))
            else:
                # Evenly spaced, strictly within range
                extract_indices = sorted(
                    {
                        min(total_frames - 1, int(i * total_frames / self.frame_per_seg))
                        for i in range(1, self.frame_per_seg + 1)
                    }
                )

            frames = []
            for frame_idx in extract_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            return frames
        finally:
            cap.release()

    def analyze_video(self, video_path):
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "vision_agent", "message": "Alright, rolling up my sleeves to analyze this video. Hope it\'s not a horror flick! 🎬"}')

        frames = self._extract_frames(video_path)
        if not frames:
            if self.agent_history_logger:
                self.agent_history_logger.info('{"role": "vision_agent", "message": "No readable frames in this segment; skipping."}')
            return ""

        encoded_images = [self.encode_image_to_base64(frame) for frame in frames]
        result = self.describe_frames(encoded_images)

        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "vision_agent", "message": "Video frame analysis done! Visual summary ready."}')
        return result
