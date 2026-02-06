import os
import time
import base64
from datetime import timedelta
from collections import Counter
import ffmpeg
import torch
from PIL import Image
import openai
from openai import OpenAI
from src.vision.vision_agent import VisionAgent
import logging

# Optional imports - loaded lazily when needed
cv2 = None
clip = None
BlipProcessor = None
BlipForConditionalGeneration = None
pipeline = None

class GptVisionAgent(VisionAgent):
    def __init__(self, model_name, model_path, frame_per_seg, cache_dir=None):
        super().__init__(model_name, model_path, frame_per_seg, cache_dir)
        self.visual_cues = []
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
        # Initialize agent history logger - will be set by task
        self.agent_history_logger = None
    
    def set_agent_history_logger(self, logger):
        """Set the agent history logger from task"""
        self.agent_history_logger = logger
    
    def load_model(self, model_path = None):
        # TODO: Implement model loading logic (e.g., OpenAI API or local vLLM model)
        return None
    
    def load_tokenizer(self, model_name = None):
        # TODO: Implement tokenizer loading logic
        return None
     
    def encode_image_to_base64(self, image):
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image
    
    def analyze_frame(self, frame):
        encoded_image = self.encode_image_to_base64(frame)
        return self.describe_image(encoded_image)
    
    def describe_image(self, base64_image):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the visual cue and recongnize objects in the image:.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        # Best-effort usage logging
        try:
            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            tt = getattr(usage, "total_tokens", None) if usage else None
            if hasattr(self, "_record_usage"):
                self._record_usage(
                    provider="openai",
                    model="gpt-4o",
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    phrase_index=None,
                    extra={}
                )
        except Exception:
            pass
        return response.choices[0].message.content
    
    def summarize_cue(self):
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "vision_agent", "message": "Let me feast my eyes on these frames... Picasso mode: ON 🖼️"}')
        
        prompt = f"Summarize the following visual description: {' '.join(self.visual_cues)}"
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI model that summarizes visual description from a video."},
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
        )
        # Best-effort usage logging
        try:
            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            tt = getattr(usage, "total_tokens", None) if usage else None
            if hasattr(self, "_record_usage"):
                self._record_usage(
                    provider="openai",
                    model="gpt-4o",
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    phrase_index=None,
                    extra={}
                )
        except Exception:
            pass
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "vision_agent", "message": "Visual summary ready! I could do this all day, but I won\'t."}')
        return response.choices[0].message.content

    def analyze_video(self, video_path):
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "vision_agent", "message": "Alright, rolling up my sleeves to analyze this video. Hope it\'s not a horror flick! 🎬"}')
        
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 4:
            extract_indices = list(range(total_frames))
        else:
            extract_indices = [int(i * total_frames / self.frame_per_seg) for i in range(1, self.frame_per_seg + 1) if i * total_frames / self.frame_per_seg > 0]

        frame_count = 0
        self.visual_cues = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in extract_indices:
                description = self.analyze_frame(frame)
                self.visual_cues.append(description)

            frame_count += 1

        cap.release()
        
        if self.agent_history_logger:
            self.agent_history_logger.info('{"role": "vision_agent", "message": "Video frame analysis done! My eyes need a break, but let\'s summarize first."}')
        result = self.summarize_cue()
        return result



class assistant_vision_api(VisionAgent):
    def __init__(self, model_name, model_path, extract_interval, cache_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_cues = []
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=openai.api_key, timeout=600)
        self.tokenizer = self.load_tokenizer(model_name)
        #super().__init__(model_name, model_path, extract_interval, cache_dir)
        self.extract_interval = extract_interval
        self.model = model_path
        self.thread_id = self.client.beta.threads.create().id
        self.file_ids = []
        self.cache_dir = cache_dir

    def load_model(self, model_path=None):
        return None

    def load_tokenizer(self, model_name=None):
        return None

    def upload_image_to_openai(self, image_path):
        """Uploads an image file to OpenAI's file storage."""
        response = self.client.files.create(
            file=open(image_path, "rb"),
            purpose="assistants"
        )
        self.file_ids.append(response.id)
        print(response.status)
        return response.id

    def delete_uploaded_file(self, file_id):
        """Deletes an uploaded file from OpenAI storage."""
        self.client.files.delete(file_id)

    def analyze_image_with_assistant(self, file_id):
        """Analyzes an image using OpenAI Assistants API."""
        # Create a user message with the image file
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=[
                {"type": "text", "text": "analyze the image:"},
                {"type": "image_file", "image_file": {"file_id": file_id}}
            ]
        )

        # Run the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.model
        )

        # Wait for the assistant to finish processing
        while run.status != "completed":
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )

        # Retrieve the assistant's response
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return msg.content[0].text.value

        return "No response from assistant."

    def analyze_frame(self, frame):
        """Processes a single video frame: saves, uploads, analyzes, and deletes it."""
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        timestamp = int(time.time())
        temp_image_path = f"{self.cache_dir}/temp_frame_{timestamp}.jpg"
        
        # Save the frame
        cv2.imwrite(temp_image_path, frame)

        # Upload image
        file_id = self.upload_image_to_openai(temp_image_path)

        # Analyze image
        description = self.analyze_image_with_assistant(file_id)

        # Delete uploaded file from OpenAI storage
        #self.delete_uploaded_file(file_id)

        # Remove local file
        os.remove(temp_image_path)

        return description

    def summarize_cue(self):
        """Requests a summary of all analyzed visual cues."""
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=[{"type": "text", "text": "Conclude the thread."}]
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.model
        )

        # Wait for the assistant to finish processing
        while run.status != "completed":
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )

        # Retrieve the summary response
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return msg.content[0].text.value

        return "No summary available from the assistant."


    def analyze_video(self, video_path):
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 4:
            extract_indices = list(range(total_frames))  # Take all available frames if fewer than 4
        else:
            extract_indices = [int(i * total_frames / 4) for i in range(1, 5)]  # Pick 4 evenly spaced frames

        frame_count = 0
        self.visual_cues = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in extract_indices:
                description = self.analyze_frame(frame)
                self.visual_cues.append(description)

            frame_count += 1

        cap.release()
        return self.summarize_cue()

    
    def cleanup(self):
        while self.file_ids:
            file_id = self.file_ids.pop()
            self.client.files.delete(file_id)
        if self.thread_id:
            self.client.beta.threads.delete(self.thread_id)
        return True

class CLIPVisionAgent(VisionAgent):
    def __init__(self, model_name, model_path, extract_interval, file_path, cache_dir=None):
        super().__init__(model_name, model_path, extract_interval, cache_dir)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model (lazy import)
        global clip
        if clip is None:
            import clip as _clip
            clip = _clip
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Load category database
        self.category_database = self.load_category_database(file_path)

        self.cache_dir = cache_dir

        self.extract_interval = extract_interval

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

    def extract_frames(self, video_path):
        """ Extract key frames from the video """
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        video = cv2.VideoCapture(video_path)
        frames = []
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.extract_interval)
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

        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
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

def get_video_info_ffmpeg(video_path):
    # Retrieve video information using ffmpeg
    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError("No video stream found")

    fps = eval(video_stream['r_frame_rate'])
    if 'nb_frames' in video_stream:
        total_frames = int(video_stream['nb_frames'])
    else:
        duration = float(video_stream['duration'])
        total_frames = int(duration * fps)

    return fps, total_frames
    
class HfVisionAgent:
    def __init__(self, model_name="llava-hf/llava-interleave-qwen-0.5b-hf", api="OPENAI_API_KEY", seconds_per_frame=30, device=None, verbose=True):
        # Initialize the HfVisionAgent with model details, API key, frame interval, and device settings
        self.model_name = model_name
        self.interval_sec = seconds_per_frame  # Interval in seconds between frames to analyze
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_cues = []
        self.verbose = verbose
        self.pipe = None
        self.load_model()
        self.client = OpenAI(api_key=api)

    def load_model(self):
        # Load the multimodal pipeline model (lazy import transformers)
        global pipeline
        if pipeline is None:
            from transformers import pipeline as _pipeline
            pipeline = _pipeline
        if self.verbose:
            print(f"Loading multimodal pipeline: {self.model_name}")
        self.pipe = pipeline("image-text-to-text", model=self.model_name, device=self.device)

    def encode_image_to_caption(self, image):
        # Convert an image to a caption using the loaded pipeline
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": "Briefly describe this image in 1-2 sentences (max 60 tokens)."}
                ],
            }
        ]

        outputs = self.pipe(text=messages, max_new_tokens=60, return_full_text=False)
        caption = outputs[0]["generated_text"]
        return caption

    def summarize_cue(self):
        # Summarize the collected visual descriptions into a coherent summary
        prompt = f"Summarize the following visual description: { ' | '.join(self.visual_cues) }"
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI model that summarizes visual description from a video."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        return response.choices[0].message.content

    def analyze_frame(self, frame):
        # Analyze a single frame and obtain its caption
        caption = self.encode_image_to_caption(frame)
        if self.verbose:
            print(f"[Caption]: {caption}")
        return caption

    def analyze_video(self, video_path):
        # Analyze the video by extracting frames at specified intervals and summarizing their content
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        fps, total_frames = get_video_info_ffmpeg(video_path)
        if self.verbose:
            print(f"Using FFmpeg - FPS: {fps}, Total Frames: {total_frames}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        duration_sec = int(total_frames / fps)
        extract_indices = [int(t * fps) for t in range(0, duration_sec, self.interval_sec)]

        self.visual_cues = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in extract_indices:
                caption = self.analyze_frame(frame)
                self.visual_cues.append(caption)

            frame_count += 1

        cap.release()
        return self.summarize_cue()
