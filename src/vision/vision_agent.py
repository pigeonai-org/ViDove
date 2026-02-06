from abc import ABC
import json
import time
from typing import Optional

"""
Interface for vision agent.
"""


class VisionAgent(ABC):
    def __init__(self, model_name, model_path=None, frame_per_seg=4, cache_dir=None, logger=None):
        self.model_name = model_name
        self.model_path = model_path
        self.frame_per_seg = frame_per_seg  # frame interval
        self.cache_dir = cache_dir
        self.frames = []
        self.model = None
        self.device = None
        self.logger = logger
        # usage tracking
        self.task_id = None
        self.usage_log_path = None
        self.load_model()

    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def load_model(self):
        # load model from model_path
        # here we can load pretrained CLIP model, pretrained vLLM model, or other models from huggingface
        self.model = ...
        pass

    # --- Usage logging helpers ---
    def set_usage_log_path(self, path: Optional[str]):
        self.usage_log_path = path

    def set_task_id(self, task_id: Optional[str]):
        self.task_id = task_id

    def _record_usage(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        phrase_index: int | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append a JSONL usage record tagged as vision."""
        if not self.usage_log_path:
            return
        try:
            rec = {
                "request_id": f"{self.task_id}:vision:{int(time.time()*1000)}",
                "timestamp": int(time.time()),
                "task_id": self.task_id,
                "provider": provider,
                "model": model,
                "category": "vision",
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "phrase_index": phrase_index,
                "extra": {"agent": "vision", **(extra or {})},
            }
            with open(self.usage_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # best-effort; don't break pipeline
            pass

    def extract_frames(self, video_path, cache_dir=None):
        # extract frames from video
        # if cache_dir is not None, save the frames to the cache_dir
        # return a list of frames
        self.frames = ...

    def analyze_frame(self, frame):
        # analyze frame
        pass

    def analyze_video(self, video_path):
        # analyze video
        visual_cues = ...  # here's the final prompt feed into whisper or translators
        return visual_cues