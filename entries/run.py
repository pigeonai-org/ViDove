import argparse
import shutil
from pathlib import Path
from uuid import uuid4

from yaml import Loader, load

import __init_lib_path

from src.task import Task
from config_schema import load_task_config, TaskConfig

"""
    Main entry for terminal environment.
    Use it for debug and development purpose. 
    Usage: python3 entries/run.py [-h] [--link LINK] [--video_file VIDEO_FILE] [--audio_file AUDIO_FILE] [--srt_file SRT_FILE] [--continue CONTINUE]
              [--launch_cfg LAUNCH_CFG] [--task_cfg TASK_CFG]
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--link", help="youtube video link here", default=None, type=str, required=False
    )
    parser.add_argument(
        "--video_file",
        help="local video path here",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--audio_file",
        help="local audio path here",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--srt_file",
        help="srt file input path here",
        default=None,
        type=str,
        required=False,
    )  # Deprecated
    # parser.add_argument("--continue", help="task_id that need to continue", default=None, type=str, required=False) # need implement
    parser.add_argument(
        "--launch_cfg",
        help="launch config path",
        default="./configs/local_launch.yaml",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--is_assistant",
        help="is assistant mode",
        default=False,
        type=bool,
        required=False,
    )
    parser.add_argument(
        "--task_cfg",
        help="task config path",
        default="./configs/task_config.yaml",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # read args and configs
    args = parse_args()
    
    try:
        # Load task configuration using schema
        task_config = load_task_config(args.task_cfg)
        print(f"✅ Successfully loaded task configuration: {args.task_cfg}")
        
        # Validate configuration
        print(f"📋 Configuration validation passed:")
        print(f"   - Source language: {task_config.source_lang}")
        print(f"   - Target language: {task_config.target_lang}")
        print(f"   - Domain: {task_config.domain}")
        print(f"   - Audio processing: {'Enabled' if task_config.audio.enable_audio else 'Disabled'}")
        print(f"   - Vision processing: {'Enabled' if task_config.vision.enable_vision else 'Disabled'}")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        # Fallback to old method
        print("🔄 Falling back to old configuration loading method...")
        task_cfg = load(open(args.task_cfg), Loader=Loader)
        task_config = None
    else:
        # Convert to dict for compatibility
        task_cfg = task_config.to_dict()
    
    launch_cfg = load(open(args.launch_cfg), Loader=Loader)

    # initialize dir
    local_dir = Path(launch_cfg["local_dump"])
    if not local_dir.exists():
        local_dir.mkdir(parents=False, exist_ok=False)

    # Check API source from azure or openai
    if launch_cfg["api_source"] == "openai":
        if task_config:
            task_config.api_source = "openai"
        task_cfg["api_source"] = "openai"
    elif launch_cfg["api_source"] == "azure":
        if task_config:
            task_config.api_source = "azure"
        task_cfg["api_source"] = "azure"

    # get task id
    task_id = str(uuid4())

    # create local dir for the task
    task_dir = local_dir.joinpath(f"task_{task_id}")
    task_dir.mkdir(parents=False, exist_ok=False)
    task_dir.joinpath("results").mkdir(parents=False, exist_ok=False)

    # add is_assistant to task_config
    if task_config:
        task_config.is_assistant = args.is_assistant
    task_cfg["is_assistant"] = args.is_assistant

    # disable spell check and term correct for assistant mode
    if args.is_assistant:
        if task_config:
            task_config.pre_process.spell_check = False
            task_config.pre_process.term_correct = False
        task_cfg["pre_process"]["spell_check"] = False
        task_cfg["pre_process"]["term_correct"] = False
        print("🔧 Assistant mode: Spell checking and term correction disabled")

    # Task create
    if args.link is not None:
        try:
            print(f"🎬 Creating task from YouTube link: {args.link}")
            task = Task.fromYoutubeLink(args.link, task_id, task_dir, task_cfg)
        except Exception:
            shutil.rmtree(task_dir)
            raise RuntimeError("Failed to create task from YouTube link")
    elif args.video_file is not None:
        try:
            print(f"🎬 Creating task from video file: {args.video_file}")
            task = Task.fromVideoFile(args.video_file, task_id, task_dir, task_cfg)
        except Exception:
            shutil.rmtree(task_dir)
            raise RuntimeError("Failed to create task from video file")
    elif args.audio_file is not None:
        try:
            print(f"🎵 Creating task from audio file: {args.audio_file}")
            # Use AudioTask for audio inputs
            task = Task.fromAudioFile(args.audio_file, task_id, task_dir, task_cfg)
        except Exception:
            shutil.rmtree(task_dir)
            raise RuntimeError("Failed to create task from audio file")
    elif args.srt_file is not None:
        try:
            print(f"📝 Creating task from SRT file: {args.srt_file}")
            task = Task.fromSRTFile(args.srt_file, task_id, task_dir, task_cfg)
        except Exception:
            shutil.rmtree(task_dir)
            raise RuntimeError("Failed to create task from SRT file")
    else:
        print("❌ Please provide input source: --link, --video_file, --audio_file or --srt_file")
        exit(1)

    # add task to the status queue
    print(f"🚀 Starting task execution: {task_id}")
    task.run()
