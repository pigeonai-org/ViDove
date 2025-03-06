import base64
import openai
import torch
import cv2
from .vision_agent import VisionAgent

class GptVisionAgent(VisionAgent):
    def __init__(self, model_name, model_path, extract_interval, cache_dir=None):
        super().__init__(model_name, model_path, extract_interval, cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(model_name)
        self.model.to(self.device)
        self.visual_cues = []
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def load_model(self, model_path):
        # TODO: Implement model loading logic (e.g., OpenAI API or local vLLM model)
        return None
    
    def load_tokenizer(self, model_name):
        # TODO: Implement tokenizer loading logic
        return None
     
    def encode_image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image
    
    def analyze_frame(self, frame):
        encoded_image = self.encode_image_to_base64(frame)
        return self.describe_image(encoded_image)
    
    def describe_image(self, base64_image):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI model that describes images."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_base64", "image_base64": base64_image}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0]['message']['content']
    
    def summarize_cue(self):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI model that summarizes visual cues from a video."},
                {
                    "role": "user", 
                    "content": "Summarize the following visual cues: " + " ".join(self.visual_cues)
                }
            ],
        )
        return response.choices[0]['message']['content']
    
    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.extract_interval == 0:
                description = self.analyze_frame(frame)
                self.visual_cues.append(description)
            
            frame_count += 1
        
        cap.release()
        return self.visual_cues

class CLIPVisionAgent(VisionAgent):
    def __init__(self, model_name, model_path, extract_interval, file_path, cache_dir=None):
        super().__init__(model_name, model_path, extract_interval, cache_dir)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Load category database
        self.category_database = self.load_category_database(file_path)

    def load_category_database(self, file_path):
        """ Load category database from file """
        category_list = []
        try:
            with open(file_path, "r") as file:
                for line in file:
                    category_path = line.strip().split(" ")[0][3:]
                    categories = category_path.split("/")
                    category_list.extend(categories)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []

        return list(set(word.replace("_", " ") for word in category_list))

    def extract_frames(self, video_path, interval=1):
        """ Extract key frames from the video """
        video = cv2.VideoCapture(video_path)
        frames = []
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        success, frame = video.read()
        frame_count = 0

        while success:
            if frame_count % frame_interval == 0:
                frames.append(frame)
            success, frame = video.read()
            frame_count += 1

        video.release()
        return frames

    def analyze_frame(self, frame):
        """ Use CLIP to recognize content in a single frame """
        if not self.category_database:
            print("Warning: Category database is empty.")
            return ""

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(self.category_database).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).squeeze()

        top_keyword_idx = similarity.argmax().item()
        return self.category_database[top_keyword_idx]

    def analyze_video(self, video_path):
        """ Analyze the video and extract key concepts """
        frames = self.extract_frames(video_path, self.extract_interval)
        all_keywords = [self.analyze_frame(frame) for frame in frames]
        keyword_counts = Counter(all_keywords)
        sorted_keywords = keyword_counts.most_common()
        top_keywords = [keyword for keyword, _ in sorted_keywords]

        prompt = f"The key concepts in this video include: {', '.join(top_keywords[:3])}."
        return prompt
