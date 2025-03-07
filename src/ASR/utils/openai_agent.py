import base64
import openai
from openai import OpenAI
import cv2
import numpy as np
import os

# Initialize OpenAI API key
# Replace with your OpenAI API key

def capture_frame(video_path, time_frame):
    """
    Capture a frame from the video at the specified time (in seconds).
    """
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Cannot open video file.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_position = int(fps * time_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Error: Unable to read the frame.")
    
    return frame

def encode_image_to_base64(image):
    """
    Encode a BGR image (as numpy array) to a base64 string.
    """
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def get_image_description(base64_image):
    """
    Send the base64-encoded image to OpenAI's API and get a description.
    """
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What is in this image?",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
    )

    return response.choices[0]

# Example usage
video_file_path = 'test5.mp4'  # Replace with your video file path
time_frame = 5  # Capture frame at the 5-second mark

try:
    frame = capture_frame(video_file_path, time_frame)
    base64_image = encode_image_to_base64(frame)
    description = get_image_description(base64_image)
    print(f"Image Description: {description}")
except Exception as e:
    print(f"An error occurred: {e}")