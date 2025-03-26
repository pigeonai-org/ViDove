import os
import sys
from pathlib import Path
import logging
from uuid import uuid4
from yaml import Loader, load
import glob
import traceback

sys.path.append("./")
from src.task import Task


"""
Batch processing of translation tasks (using ViDove Agent)

1. Generate srt file names corresponding to the source file id
2. Remove files other than srt files

"""

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("batch_processor")

def load_config(config_path="./configs/task_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return load(f, Loader=Loader)

def process_video(video_path_str, output_dir, task_cfg, logger):
    """Process a single video file and output SRT"""
    # Create a unique task ID
    task_id = str(uuid4())
    
    # Create task directory in a temporary location
    temp_dir = Path("./evaluation/test_data/temp_tasks")  # Create temporary directory outside of output directory
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    task_dir = temp_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing video: {video_path_str}")
    logger.info(f"Task ID: {task_id}")
    
    video_path = Path(video_path_str)
    
    try:
        # Create task - pass Path objects instead of strings
        task = Task.fromVideoFile(video_path, task_id, task_dir, task_cfg)
        
        # Run task (we don't need ASR model preloading for batch processing)
        task.run(None)
        
        # Get original filename (without extension)
        original_filename = Path(video_path).stem
        
        # Copy SRT file to desired output location with original filename
        target_lang = task_cfg["target_lang"]
        source_srt = task_dir / "results" / f"{task_id}_{target_lang}.srt"
        output_srt = Path(output_dir) / f"{original_filename}.srt"
        
        # Copy the file
        if source_srt.exists():
            import shutil
            shutil.copy2(source_srt, output_srt)
            logger.info(f"Generated SRT: {output_srt}")
            
            # Clean up temp directory after successful processing
            try:
                shutil.rmtree(task_dir)
                logger.info(f"Cleaned up temporary directory: {task_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up directory {task_dir}: {str(e)}")
                
            return str(output_srt)
        else:
            logger.error(f"Failed to generate SRT for {video_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing {video_path_str}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to clean up even on failure
        try:
            import shutil
            shutil.rmtree(task_dir)
            logger.info(f"Cleaned up temporary directory after error: {task_dir}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up directory {task_dir}: {str(cleanup_err)}")
            
        return None

def batch_process_videos(input_dir, output_dir, task_cfg, logger):
    """Process all videos in the input directory"""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Process each video
    results = []
    successful = 0
    failed = 0
    
    for video_path in video_files:
        result = process_video(video_path, output_dir, task_cfg, logger)
        if result:
            results.append(result)
            successful += 1
            print(f"success: {video_path}")
        else:
            failed += 1
            # 如果失败的话，把失败了的文件的id记录到一个fail.txt文件中
            with open("./evaluation/test_data/srt_output/fail.txt", "a") as f:
                f.write(f"{video_path.split('/')[-1]}\n")
    
    logger.info(f"Batch processing complete. Generated {successful} SRT files. Failed: {failed}.")
    return results

def main():
    # Get paths from command line or use defaults
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./evaluation/test_data/videos"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./evaluation/test_data/srt_output"
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    task_cfg = load_config()
    
    # Set configuration for SRT-only output
    task_cfg["output_type"]["video"] = False
    task_cfg["output_type"]["subtitle"] = "srt"
    
    # Optional: customize other settings
    # task_cfg["source_lang"] = "EN"
    # task_cfg["target_lang"] = "ZH"
    # task_cfg["translation"]["model"] = "gpt-4o"
    
    # Process videos
    batch_process_videos(input_dir, output_dir, task_cfg, logger)

if __name__ == "__main__":
    main()
