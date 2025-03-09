from transformers import AutoModel, AutoTokenizer
import torch

"""
    Interface for vision agent.

"""

class VisionAgent:
    def __init__(self, model_name, model_path=None, extract_interval=1800, cache_dir=None):
        self.model_name = model_name
        self.model_path = model_path
        self.extract_interval = extract_interval # frame interval
        self.cache_dir = cache_dir
        self.frames = []
        self.model = None
        self.device = None
        self.load_model()
    
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
