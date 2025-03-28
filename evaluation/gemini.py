import base64
import os
from google import genai
from google.genai import types
from vertexai.generative_models import Part, GenerationConfig
import sys
from pathlib import Path
import glob

class Gemini:
    def __init__(self, model_name="gemini-2.0-flash", api_key=None, input_dir=None, output_dir=None):
        """
        Initialize the Gemini translator.
        
        Args:
            model_name (str): The name of the Gemini model to use
            api_key (str, optional): API key for Gemini. If None, will use environment variable
        """
        self.model_name = model_name
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided either directly or via GEMINI_API_KEY environment variable")
        
        self.client = genai.Client(api_key=api_key)
        self.input_dir = input_dir
        self.output_dir = output_dir        
        
        self.successful = 0
        self.failed = 0
        self.results = []
        
        
    
    def generate_translation(self, video_path):
        """
        Generate Chinese subtitle translation from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: The generated translation with timestamps
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Read video file as bytes
        with open(video_path, "rb") as f:
            video_data = f.read()
        
        # Generate translation prompt
        
        
        # Please maintain the original style and tone of the speech.
        prompt = """Please translate the speech in this video to Chinese (Simplified).
        Output format should be the same as the following (including the time stamps, and keep your content in Chinese), do not differ from the format:
        1
        00:00:00,240 --> 00:00:02,523
        其次 我建议使用"润滑槽"方法

        2
        00:00:02,523 --> 00:00:04,701
        这意味着每天多次在单杠上悬挂

        3
        00:00:04,701 --> 00:00:07,271
        时间约为你最大悬挂时间的50%
        """
        
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=video_data,
                        mime_type="video/mp4"
                    )
                ]
            )
        ]
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        
        
        output_path = os.path.splitext(video_path)[0] + ".srt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Translation saved to {output_path}")
    
        
        return response.text
    def batch_process_videos(self):
        """
        Batch process all videos in the input directory and save the translations to the output directory.
        
        Args:
            input_dir (str): Path to the directory containing the videos
            output_dir (str): Path to the directory to save the translations
        """
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.input_dir, f"*{ext}")))
        
        # Process each video
        for video_path in video_files:
            # output_srt_file = os.path.splitext(video_path)[0] + ".srt"
            result = self.generate_translation(video_path)

            if result:
                # self.results.append(result)
                self.successful += 1
                print(f"Success: {video_path}")
            else:
                self.failed += 1
                # Record failed file ID
                with open(os.path.join(self.output_dir, "fail.txt"), "a") as f:
                    f.write(f"{os.path.basename(video_path)}\n")
            
                        
def main():
    """Main function to run the video translation."""
    video_path = "./evaluation/test_data/videos/test/_l0SHo7ekoQ_00.mp4"
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./evaluation/test_data/videos/test"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./evaluation/test_data/gemini_results"
    

    translator = Gemini(input_dir=input_dir,output_dir=output_dir)
    
    # 单文件处理
    # translation = translator.generate_translation(video_path)
    # print(translation)
    
    # 多文件处理    
    translator.batch_process_videos()
    

if __name__ == "__main__":
    main()