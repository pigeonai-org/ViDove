import base64
import openai
import torch
import cv2
from vision_agent import VisionAgent
import os
from openai import OpenAI
import time

class GptVisionAgent(VisionAgent):
    def __init__(self, model_name, model_path, extract_interval, cache_dir=None):
        super().__init__(model_name, model_path, extract_interval, cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(model_name)
        self.visual_cues = []
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
    
    def load_model(self, model_path = None):
        # TODO: Implement model loading logic (e.g., OpenAI API or local vLLM model)
        return None
    
    def load_tokenizer(self, model_name = None):
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
        #print(response.choices[0]['message']['content'])
        return response.choices[0].message.content
    
    def summarize_cue(self):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI model that summarizes visual cues from a video."},
                {
                    "role": "user", 
                    "content": "Conclude the chain of visual cues: " + " ".join(self.visual_cues)
                }
            ],
        )  
        return response.choices[0].message.content

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.extract_interval == 0:
                if frame_count == 0:
                    frame_count += 1
                    continue
                description = self.analyze_frame(frame)
                #print(description)
                self.visual_cues.append(description)
            
            frame_count += 1
        
        cap.release()
        return self.summarize_cue()
    


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
        temp_image_path = "temp_frame.jpg"
        
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
        """Extracts frames from a video, processes them, and summarizes findings."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.extract_interval == 0:
                if frame_count == 0:
                    frame_count += 1
                    continue
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
