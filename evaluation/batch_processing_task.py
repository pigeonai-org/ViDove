import os
import sys
from pathlib import Path
import logging
from uuid import uuid4
from yaml import Loader, load
import glob

sys.path.append("./")
from src.task import Task


"""
1. 需要根据源文件的id来对应生成srt文件的名字
2. 尽量去掉srt文件以外的文件

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
    
    # Create task directory
    task_dir = Path(output_dir) / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "results").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing video: {video_path_str}")
    logger.info(f"Task ID: {task_id}")
    
    video_path = Path(video_path_str)
    
    # Create task
    # task = Task.fromVideoFile(str(video_path), task_id, str(task_dir), task_cfg)
    task = Task.fromVideoFile(video_path, task_id, task_dir, task_cfg)
    
    # Run task (we don't need ASR model preloading for batch processing)
    task.run(None)
    
    # Get original filename (without extension)
    original_filename = Path(video_path).stem
    
    # Copy SRT file to desired output location with original filename
    target_lang = task_cfg["target_lang"]
    source_srt = task_dir / "results" / f"{task_id}_{target_lang}.srt"
    output_srt = Path(output_dir) / f"{original_filename}_{target_lang}.srt"
    
    # Copy the file
    if source_srt.exists():
        import shutil
        shutil.copy2(source_srt, output_srt)
        logger.info(f"Generated SRT: {output_srt}")
    else:
        logger.error(f"Failed to generate SRT for {video_path}")
    
    return str(output_srt) if source_srt.exists() else None

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
    for video_path in video_files:
        result = process_video(video_path, output_dir, task_cfg, logger)
        if result:
            results.append(result)
    
    logger.info(f"Batch processing complete. Generated {len(results)} SRT files.")
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
