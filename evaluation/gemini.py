import base64
import os
from google import genai
from google.genai import types
from vertexai.generative_models import Part, GenerationConfig

class Gemini:
    def __init__(self, model_name="gemini-2.0-flash", api_key=None):
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
        prompt = """Please translate the speech in this video to Chinese (Simplified).
        Please maintain the original style and tone of the speech.
        Output format should be the same as the following (including the time stamps, keep in Chinese):
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
        
        # try:
        #     # Using vertexai API
        #     contents = [
        #         Part.from_text(text=prompt),
        #         Part.from_data(data=video_data, mime_type="video/mp4")
        #     ]
            
        #     generation_config = GenerationConfig(
        #         temperature=0.2,
        #         top_p=0.95,
        #         top_k=40,
        #         max_output_tokens=2048
        #     )
            
        #     response = self.client.generate_content(
        #         model=self.model_name,
        #         contents=contents,
        #         generation_config=generation_config
        #     )
            
        #     return response.text
            
        # except Exception as e:
        #     # Fallback to alternative method if the first one fails
        #     try:
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
        
        # generate_content_config = types.GenerateContentConfig(
        #     temperature=0.2,
        #     top_p=0.95,
        #     top_k=40,
        #     max_output_tokens=2048
        # )
    
# 用这个方法      
#                 response = self.client.models.generate_content(
#                     model=self.model,
#                     contents=[
#                         Part.from_text(text=question),
#                         Part.from_bytes(data=audio_data, mime_type="audio/wav"),
#                     ]
#                 )

        
        
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        
        return response.text
                
            # except Exception as inner_e:
            #     raise RuntimeError(f"Failed to generate translation: {str(e)}. Inner exception: {str(inner_e)}")

def main():
    """Main function to run the video translation."""
    video_path = "./evaluation/test_data/videos/test/_l0SHo7ekoQ_00.mp4"
    
    try:
        translator = Gemini()
        translation = translator.generate_translation(video_path)
        print(translation)
        
        # Optionally save the translation to a file
        output_path = os.path.splitext(video_path)[0] + ".srt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translation)
        print(f"Translation saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()