from abc import ABC

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
        visual_cues = ... # here's the final prompt feed into whisper or translators
        return visual_cues