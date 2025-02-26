from transformers import AutoModel, AutoTokenizer

"""
    Interface for vision agent.

"""

class VisionAgent:
    def __init__(self, model_name, model_path, extract_interval, cache_dir=None):
        self.model_name = model_name
        self.model_path = model_path
        self.extract_interval = extract_interval
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


class CLIPVisionAgent(VisionAgent):
    # TODO: implement CLIP vision agent @Zongheng00
    def __init__(self, model_name, model_path, extract_interval, cache_dir=None):
        super().__init__(model_name, model_path, extract_interval, cache_dir)
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def analyze_frame(self, frame):
        # analyze frame
        pass
    
    def analyze_video(self, video_path):
        # analyze video
        visual_cues = ... # here's the final prompt feed into whisper or translators
        return visual_cues
    
class vLLMVisionAgent(VisionAgent):
    # TODO: implement vLLM vision agent @worldqwq
    def __init__(self, model_name, model_path, extract_interval, cache_dir=None):
        super().__init__(model_name, model_path, extract_interval, cache_dir)
        self.model = ...
        self.tokenizer = ...
        self.device = ...
        self.model.to(self.device)

    def analyze_frame(self, frame):
        # analyze frame
        pass
    
    def analyze_video(self, video_path):
        # analyze video 
        visual_cues = ... # here's the final prompt feed into whisper or translators
        return visual_cues

