import os
import sys
from pathlib import Path
import logging
import time
import traceback
import glob
from uuid import uuid4

# Vertex AI Gemini API
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import tempfile

sys.path.append("./")

"""
Batch processing of video files using Vertex AI Gemini model

Usage:
1. follow the instructions from https://cloud.google.com/sdk/docs/install to authenticate with the gcloud CLI
2. Set environment variable GEMINI_PROJECT_ID and GOOGLE_CLOUD_PROJECT pointing to your GCP project ID
3. Run the script to process videos: python evaluation/gemini_batch_processing_task.py [input_dir] [output_dir]
"""

class VertexAIGeminiBatchProcessor:
    def __init__(self, input_dir=None, output_dir=None, bucket_name=None, location="us-central1"):
        # Setup logging
        self.logger = self._setup_logging()
        
        # Set paths
        self.input_dir = input_dir or "./evaluation/test_data/videos"
        self.output_dir = output_dir or "./evaluation/test_data/gemini_results"
        
        # Set GCP parameters
        self.project_id = os.environ.get("GEMINI_PROJECT_ID")
        if not self.project_id:
            self.logger.error("Environment variable GEMINI_PROJECT_ID not set. Please set it and try again.")
            raise ValueError("Missing GCP project ID")
            
        self.bucket_name = bucket_name or f"{self.project_id}-gemini-videos"
        self.location = location
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-1.5-flash-002")
        
        # Statistics
        self.successful = 0
        self.failed = 0
        self.results = []

    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger("vertex_ai_gemini_processor")

    def _initialize_vertex_ai(self):
        """Initialize Vertex AI"""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.logger.info(f"Initialized Vertex AI (Project: {self.project_id}, Region: {self.location})")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            raise

    def _ensure_bucket_exists(self):
        """Ensure GCS bucket exists"""
        try:
            bucket = self.storage_client.get_bucket(self.bucket_name)
            self.logger.info(f"Found bucket: {self.bucket_name}")
        except Exception:
            self.logger.info(f"Creating new bucket: {self.bucket_name}")
            bucket = self.storage_client.create_bucket(self.bucket_name, location=self.location)
        return bucket

    def _upload_video_to_gcs(self, video_path):
        """Upload video to GCS"""
        try:
            video_path = Path(video_path)
            blob_name = f"videos/{video_path.stem}_{uuid4()}{video_path.suffix}"
            
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            # Upload video file
            self.logger.info(f"Uploading {video_path} to GCS...")
            blob.upload_from_filename(video_path)
            
            # Generate GCS URI
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            self.logger.info(f"Uploaded to: {gcs_uri}")
            
            return gcs_uri
        except Exception as e:
            self.logger.error(f"Failed to upload video to GCS: {str(e)}")
            raise

    def process_video(self, video_path_str):
        """Process a single video file and generate Chinese subtitles"""
        video_path = Path(video_path_str)
        output_srt = Path(self.output_dir) / f"{video_path.stem}.srt"
        
        self.logger.info(f"Processing video: {video_path}")
        
        try:
            # Upload video to GCS
            video_gcs_uri = self._upload_video_to_gcs(video_path)
            
            # Build prompt
            prompt = """
            Please carefully watch each frame of the video and create Chinese subtitles for it.
            Output the subtitles according to the following requirements:
            1. Output in standard SRT format, including sequence number, timestamp, and Chinese subtitle text
            2. Keep each subtitle entry on a single line
            3. Subtitles should accurately reflect the dialogue and content in the video
            4. Only output the text in SRT format, do not include other explanations or annotations
            5. If there is English dialogue in the video, please translate it to Chinese
            """
            
            # Build request content
            contents = [
                Part.from_uri(video_gcs_uri, mime_type="video/mp4"),
                prompt
            ]
            
            # Generate subtitles
            self.logger.info("Processing video with Gemini...")
            response = self.model.generate_content(contents)
            
            # Save subtitles
            # Create output directory
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Write SRT file
            with open(output_srt, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            self.logger.info(f"Generated subtitle file: {output_srt}")
            
            # Clean up temporary files in GCS
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob_name = video_gcs_uri.replace(f"gs://{self.bucket_name}/", "")
                blob = bucket.blob(blob_name)
                blob.delete()
                self.logger.info(f"Deleted GCS temporary file: {blob_name}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up GCS temporary file: {str(e)}")
            
            return str(output_srt)
            
        except Exception as e:
            self.logger.error(f"Failed to process video: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def process_batch(self):
        """Process all videos in the input directory"""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.input_dir, f"*{ext}")))
        
        if not video_files:
            self.logger.error(f"No video files found in {self.input_dir}")
            return []
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        # Reset statistics
        self.successful = 0
        self.failed = 0
        self.results = []
        
        # Process each video
        for video_path in video_files:
            result = self.process_video(video_path)
            if result:
                self.results.append(result)
                self.successful += 1
                self.logger.info(f"Success: {video_path}")
            else:
                self.failed += 1
                # Record failed file ID
                with open(os.path.join(self.output_dir, "fail.txt"), "a") as f:
                    f.write(f"{os.path.basename(video_path)}\n")
                    
            # Short delay to avoid rate limiting
            time.sleep(2)
        
        self.logger.info(f"Batch processing complete. Generated {self.successful} subtitle files. Failed: {self.failed}.")
        return self.results

def main():
    # Get paths from command line or use defaults
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./evaluation/test_data/videos/test"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./evaluation/test_data/gemini_results"
    
    # Optional: Get bucket name from environment variable
    bucket_name = os.environ.get("GEMINI_GCS_BUCKET")
    
    # Create and run processor
    processor = VertexAIGeminiBatchProcessor(input_dir, output_dir, bucket_name)
    processor.process_batch()

if __name__ == "__main__":
    main() 