import os
import argparse
import torch
import soundfile as sf
from pathlib import Path
import glob
import subprocess
import tempfile
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# if you want to run this file, torch is required to be at least 2.6.0.

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Recursive Parallel Video Translation with Auto-Segmentation")
    
    # Input options - either single video or folder (recursive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Path to a single video file")
    input_group.add_argument("--input_folder", type=str, help="Path to folder containing video files for recursive batch processing")
    
    parser.add_argument("--output_path", type=str, help="Path to save the translation text (for single file) or output folder (for batch processing)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the model")
    parser.add_argument("--source_lang", type=str, default="auto", help="Source language (auto for auto-detection)")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (en, zh, etc.)")
    parser.add_argument("--use_audio", action="store_true", help="Use audio in video for better translation")
    parser.add_argument("--save_audio", action="store_true", help="Save audio output")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--preserve_structure", action="store_true", help="Preserve directory structure in output folder")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum recursion depth for subdirectories (default: 10)")
    
    # Video segmentation parameters
    parser.add_argument("--segment_duration", type=int, default=10, help="Duration of each segment in seconds (default: 30s)")
    parser.add_argument("--disable_segmentation", action="store_true", help="Disable automatic video segmentation")
    parser.add_argument("--overlap_duration", type=int, default=2, help="Overlap between segments in seconds (default: 2s)")
    
    # Parallel processing parameters
    parser.add_argument("--parallel_workers", type=int, default=5, help="Number of parallel video processing workers (default: 5)")
    parser.add_argument("--shared_model", action="store_true", help="Use shared model for all workers (saves memory)")
    parser.add_argument("--max_concurrent_gpu", type=int, default=2, help="Maximum concurrent GPU operations (default: 2)")
    
    return parser.parse_args()

def get_video_files_recursive(folder_path, max_depth=10):
    """
    Recursively get all video files from the specified folder and its subdirectories
    
    Args:
        folder_path: Path to the folder containing video files
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        List of tuples: (video_file_path, relative_path_from_root)
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Supported video extensions
    video_extensions = [
        '*.mp4', '*.MP4', '*.Mp4', '*.mP4',
        '*.avi', '*.AVI', '*.Avi', '*.aVi',
        '*.mov', '*.MOV', '*.Mov', '*.mOv',
        '*.mkv', '*.MKV', '*.Mkv', '*.mKv',
        '*.wmv', '*.WMV', '*.Wmv', '*.wMv',
        '*.flv', '*.FLV', '*.Flv', '*.fLv',
        '*.webm', '*.WEBM', '*.Webm', '*.wEbm',
        '*.m4v', '*.M4V', '*.M4v', '*.m4V',
        '*.3gp', '*.3GP', '*.3Gp', '*.3gP'
    ]
    
    video_files = []
    
    def _recursive_search(current_path, current_depth):
        if current_depth > max_depth:
            print(f"Warning: Maximum depth ({max_depth}) reached at {current_path}")
            return
        
        # Search for video files in current directory
        for pattern in video_extensions:
            for video_file in current_path.glob(pattern):
                if video_file.is_file():
                    # Calculate relative path from the root input folder
                    relative_path = video_file.relative_to(folder_path)
                    video_files.append((str(video_file), str(relative_path)))
        
        # Recursively search subdirectories
        for subdir in current_path.iterdir():
            if subdir.is_dir():
                _recursive_search(subdir, current_depth + 1)
    
    print(f"🔍 Recursively searching for video files in: {folder_path}")
    _recursive_search(folder_path, 0)
    
    if not video_files:
        print(f"Warning: No video files found in folder: {folder_path}")
        print("Supported formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v, 3gp")
        return []
    
    # Sort by relative path for consistent processing order
    video_files.sort(key=lambda x: x[1])
    
    print(f"📁 Found {len(video_files)} video files across all subdirectories")
    
    # Display directory structure
    dirs = set()
    for _, rel_path in video_files:
        dir_path = str(Path(rel_path).parent)
        if dir_path != '.':
            dirs.add(dir_path)
    
    if dirs:
        print("📂 Directories containing videos:")
        for dir_path in sorted(dirs):
            count = sum(1 for _, rel_path in video_files if str(Path(rel_path).parent) == dir_path)
            print(f"  {dir_path}: {count} files")
    
    return video_files

def get_video_duration(video_path):
    """
    Get video duration using ffprobe
    
    Args:
        video_path: Path to the video file
        
    Returns:
        float: Duration in seconds, or None if failed
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
    except Exception as e:
        print(f"⚠️ Warning: Could not get duration for {os.path.basename(video_path)}: {e}")
    return None

def split_video_into_segments(video_path, segment_duration=10, overlap_duration=2):
    """
    Split a video into smaller segments using ffmpeg
    
    Args:
        video_path: Path to the input video
        segment_duration: Duration of each segment in seconds
        overlap_duration: Overlap between segments in seconds
        
    Returns:
        tuple: (segments_list, temp_directory)
    """
    duration = get_video_duration(video_path)
    if duration is None:
        raise ValueError(f"Could not determine video duration for {video_path}")
    
    # Create temporary directory for segments
    temp_dir = tempfile.mkdtemp(prefix="video_segments_")
    segments = []
    
    print(f"📹 Video duration: {duration:.1f}s, splitting into {segment_duration}s segments...")
    
    segment_count = 0
    start_time = 0
    
    while start_time < duration:
        # Calculate end time for this segment
        end_time = min(start_time + segment_duration, duration)
        actual_duration = end_time - start_time
        
        # Skip very short segments (less than 2 seconds)
        if actual_duration < 2:
            break
        
        # Generate segment filename
        segment_filename = f"segment_{segment_count:04d}.mp4"
        segment_path = os.path.join(temp_dir, segment_filename)
        
        # Use ffmpeg to extract segment
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(actual_duration),
            '-c:v', 'libx264',  # Re-encode to ensure compatibility
            '-c:a', 'aac',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            segment_path
        ]
        
        try:
            print(f"  📼 Creating segment {segment_count + 1}: {start_time:.1f}s - {end_time:.1f}s")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(segment_path):
                segments.append((segment_path, start_time, end_time))
                print(f"  ✅ Segment {segment_count + 1} created successfully")
            else:
                print(f"  ❌ Failed to create segment {segment_count + 1}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout creating segment {segment_count + 1}")
        except Exception as e:
            print(f"  ❌ Error creating segment {segment_count + 1}: {e}")
        
        # Move to next segment with overlap consideration
        start_time = end_time - overlap_duration
        if start_time >= duration - overlap_duration:
            break
        
        segment_count += 1
    
    print(f"📦 Created {len(segments)} segments in {temp_dir}")
    return segments, temp_dir

def generate_output_path(relative_path, output_folder, preserve_structure):
    """
    Generate output file path based on input relative path and options
    
    Args:
        relative_path: Relative path of the video file from input root
        output_folder: Base output folder
        preserve_structure: Whether to preserve directory structure
        
    Returns:
        Output file path for the translation
    """
    video_path = Path(relative_path)
    video_name = video_path.stem
    
    if preserve_structure:
        # Preserve directory structure
        output_dir = Path(output_folder) / video_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_name}.txt"
    else:
        # Flat structure with path-based naming to avoid conflicts
        # Replace directory separators with underscores
        flat_name = str(video_path.with_suffix('')).replace(os.sep, '_').replace('/', '_')
        output_path = Path(output_folder) / f"{flat_name}.txt"
    
    return str(output_path)

# Global model management for parallel processing
_shared_model = None
_shared_processor = None
_model_lock = threading.Lock()
_gpu_semaphore = None

def get_shared_model(model_path, use_flash_attn=False, save_audio=False):
    """
    获取共享模型实例（线程安全）
    
    Args:
        model_path: Path to the model
        use_flash_attn: Whether to use Flash Attention
        save_audio: Whether to save audio output
        
    Returns:
        tuple: (model, processor)
    """
    global _shared_model, _shared_processor
    
    with _model_lock:
        if _shared_model is None:
            print(f"🤖 Loading shared model from {model_path}...")
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            _shared_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            _shared_processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            
            if not save_audio:
                _shared_model.disable_talker()
            
            print("✅ Shared model loaded successfully")
    
    return _shared_model, _shared_processor

def translate_video_segment(video_path, model, processor, source_lang="auto", target_lang="en", 
                           use_audio=True, save_audio=False, segment_info=None):
    """
    Translate a single video segment
    
    Args:
        video_path: Path to the video segment
        model: Pre-loaded model
        processor: Pre-loaded processor
        source_lang: Source language
        target_lang: Target language
        use_audio: Whether to use audio
        save_audio: Whether to save audio
        segment_info: Tuple of (start_time, end_time) for this segment
        
    Returns:
        Translation text
    """
    if segment_info:
        start_time, end_time = segment_info
        print(f"  🔤 Translating segment: {start_time:.1f}s - {end_time:.1f}s")
    
    # Prepare conversation with translation instruction
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": f"翻译提供的视频中的说话内容到中文。只需要输出翻译内容原文，不要输出任何解释。"}
            ],
        },
    ]

    # Process the input
    print("    📊 Processing video segment...")
    try:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio)
        inputs = inputs.to(model.device).to(model.dtype)

        # Generate translation
        print("    🤖 Generating translation...")
        with torch.no_grad():  # Save memory
            if save_audio:
                text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio, max_new_tokens=512)
            else:
                text_ids = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False, max_new_tokens=512)
        
        # Decode translation
        print("    📝 Decoding...")
        translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return translation.strip()
    
    except Exception as e:
        print(f"    ❌ Error translating segment: {e}")
        return ""

def translate_video_with_segmentation_worker(video_path, relative_path, model_path, source_lang="auto", target_lang="en", 
                                           use_audio=True, save_audio=False, use_flash_attn=False,
                                           segment_duration=30, overlap_duration=2, disable_segmentation=False,
                                           shared_model=False, worker_id=0):
    """
    工作线程的视频翻译函数（支持分割）
    
    Args:
        video_path: Path to the video file
        relative_path: Relative path from input root
        model_path: Path to the model
        source_lang: Source language
        target_lang: Target language
        use_audio: Whether to use audio
        save_audio: Whether to save audio
        use_flash_attn: Whether to use Flash Attention
        segment_duration: Duration of each segment
        overlap_duration: Overlap between segments
        disable_segmentation: Whether to disable segmentation
        shared_model: Whether to use shared model
        worker_id: Worker ID for logging
        
    Returns:
        Translation text
    """
    global _gpu_semaphore
    
    try:
        print(f"🔄 Worker {worker_id}: Starting translation for {os.path.basename(video_path)}")
        
        # Check video duration
        duration = get_video_duration(video_path)
        if duration is None:
            print(f"⚠️ Worker {worker_id}: Could not determine video duration, processing as single video")
            disable_segmentation = True
        
        # Get model
        if shared_model:
            model, processor = get_shared_model(model_path, use_flash_attn, save_audio)
        else:
            # Load independent model
            print(f"🤖 Worker {worker_id}: Loading independent model...")
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            
            if not save_audio:
                model.disable_talker()
        
        # Decide whether to segment the video
        should_segment = not disable_segmentation and duration and duration > segment_duration
        
        if not should_segment:
            print(f"✅ Worker {worker_id}: Video duration ({duration:.1f}s) <= segment duration ({segment_duration}s), processing directly")
            
            # GPU concurrency control
            if _gpu_semaphore:
                _gpu_semaphore.acquire()
            
            try:
                translation = translate_video_segment(video_path, model, processor, source_lang, target_lang, 
                                                    use_audio, save_audio)
            finally:
                if _gpu_semaphore:
                    _gpu_semaphore.release()
            
            # Clean up independent model
            if not shared_model:
                del model, processor
                torch.cuda.empty_cache()
            
            print(f"✅ Worker {worker_id}: Completed direct translation for {os.path.basename(video_path)}")
            return translation.strip()
        
        print(f"📹 Worker {worker_id}: Video duration ({duration:.1f}s) > segment duration ({segment_duration}s), segmenting...")
        
        # Split video into segments
        segments, temp_dir = split_video_into_segments(video_path, segment_duration, overlap_duration)
        
        if not segments:
            raise ValueError("Failed to create video segments")
        
        all_translations = []
        
        try:
            # Process each segment
            for i, (segment_path, start_time, end_time) in enumerate(segments):
                print(f"\n🎬 Worker {worker_id}: Processing segment {i+1}/{len(segments)}")
                
                # GPU concurrency control
                if _gpu_semaphore:
                    _gpu_semaphore.acquire()
                
                try:
                    translation = translate_video_segment(
                        segment_path, model, processor, source_lang, target_lang, 
                        use_audio, False, (start_time, end_time)  # Don't save audio for segments
                    )
                finally:
                    if _gpu_semaphore:
                        _gpu_semaphore.release()
                
                if translation and translation.strip():
                    all_translations.append(translation.strip())
                    print(f"  ✅ Worker {worker_id}: Segment {i+1} completed: {translation[:50]}...")
                else:
                    print(f"  ⚠️ Worker {worker_id}: Segment {i+1} produced empty translation")
            
            # Combine all translations
            if all_translations:
                # Use double newline to separate segments
                final_translation = "\n\n".join(all_translations)
                print(f"🎉 Worker {worker_id}: Combined translation from {len(all_translations)}/{len(segments)} segments")
                
                # Clean up independent model
                if not shared_model:
                    del model, processor
                    torch.cuda.empty_cache()
                
                return final_translation
            else:
                raise ValueError("No segments were successfully translated")
                
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Worker {worker_id}: Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"⚠️ Worker {worker_id}: Failed to clean up temporary directory: {e}")
        
    except Exception as e:
        print(f"❌ Worker {worker_id}: Error translating {video_path}: {str(e)}")
        return ""

def recursive_parallel_translate(video_files_info, output_folder, model_path, source_lang="auto", target_lang="en",
                                use_audio=True, save_audio=False, use_flash_attn=False, preserve_structure=True,
                                segment_duration=30, overlap_duration=2, disable_segmentation=False,
                                parallel_workers=5, shared_model=False, max_concurrent_gpu=2):
    """
    递归并行翻译视频文件
    
    Args:
        video_files_info: List of (video_path, relative_path) tuples
        output_folder: Output folder for translations
        model_path: Path to the model
        source_lang: Source language
        target_lang: Target language
        use_audio: Whether to use audio
        save_audio: Whether to save audio
        use_flash_attn: Whether to use Flash Attention
        preserve_structure: Whether to preserve directory structure
        segment_duration: Duration of each segment
        overlap_duration: Overlap between segments
        disable_segmentation: Whether to disable segmentation
        parallel_workers: Number of parallel workers
        shared_model: Whether to use shared model
        max_concurrent_gpu: Maximum concurrent GPU operations
        
    Returns:
        Dict with processing statistics
    """
    global _gpu_semaphore
    
    # Initialize GPU semaphore
    _gpu_semaphore = threading.Semaphore(max_concurrent_gpu)
    
    # Preload shared model if enabled
    if shared_model:
        get_shared_model(model_path, use_flash_attn, save_audio)
    
    # Prepare task queue
    tasks = []
    for video_file, relative_path in video_files_info:
        output_path = generate_output_path(relative_path, output_folder, preserve_structure)
        
        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"⏭️ Skipping existing: {relative_path}")
            continue
        
        tasks.append((video_file, relative_path, output_path))
    
    if not tasks:
        print("No new videos to process.")
        return {"skipped": len(video_files_info), "successful": 0, "failed": 0}
    
    print(f"🚀 Starting recursive parallel processing of {len(tasks)} videos with {parallel_workers} workers")
    print(f"📊 Configuration: Shared Model = {shared_model}, Max GPU Concurrent = {max_concurrent_gpu}")
    print(f"🔪 Segmentation: {'Disabled' if disable_segmentation else f'Enabled ({segment_duration}s segments)'}")
    
    # Statistics counters
    successful_count = 0
    failed_count = 0
    skipped_count = len(video_files_info) - len(tasks)
    segmented_count = 0
    
    start_time = time.time()
    
    # Use thread pool for parallel translation
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        # Submit all tasks
        future_to_video = {}
        for i, (video_file, relative_path, output_path) in enumerate(tasks):
            future = executor.submit(
                translate_video_with_segmentation_worker,
                video_file, relative_path, model_path, source_lang, target_lang,
                use_audio, save_audio, use_flash_attn,
                segment_duration, overlap_duration, disable_segmentation, shared_model, i+1
            )
            future_to_video[future] = (video_file, relative_path, output_path)
        
        # Process completed tasks
        for future in as_completed(future_to_video):
            video_file, relative_path, output_path = future_to_video[future]
            
            try:
                translation = future.result()
                
                if translation and translation.strip():
                    # Create output directory if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save translation result
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translation)
                    
                    print(f"✅ Saved: {relative_path}")
                    print(f"📝 Preview: {translation[:50]}...")
                    successful_count += 1
                    
                    # Check if this video was segmented
                    duration = get_video_duration(video_file)
                    if duration and duration > segment_duration and not disable_segmentation:
                        segmented_count += 1
                else:
                    print(f"⚠️ Empty translation for: {relative_path}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Task failed for {relative_path}: {e}")
                failed_count += 1
    
    elapsed_time = time.time() - start_time
    
    # Output statistics
    print(f"\n{'='*80}")
    print(f"🎉 RECURSIVE PARALLEL PROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"⏱️ Total processing time: {elapsed_time/60:.1f} minutes")
    print(f"📊 Processing Summary:")
    print(f"  📁 Total files found: {len(video_files_info)}")
    print(f"  ⏭️ Files skipped (already exist): {skipped_count}")
    print(f"  ✅ Files successfully processed: {successful_count}")
    print(f"  🔪 Files requiring segmentation: {segmented_count}")
    print(f"  ❌ Files failed to process: {failed_count}")
    print(f"  🚀 Average speed: {successful_count*60/elapsed_time:.1f} videos/hour")
    print(f"  ⚡ Parallel efficiency: {parallel_workers}x workers")
    print(f"  🏗️ Structure preserved: {'Yes' if preserve_structure else 'No (flat)'}")
    print(f"  📂 Results saved in: {output_folder}")
    print(f"{'='*80}")
    
    return {
        "successful": successful_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "segmented": segmented_count,
        "total_time": elapsed_time
    }

def main():
    args = parse_args()
    
    if args.video_path:
        # Single video processing mode
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        # Set default output path if not provided
        if not args.output_path:
            args.output_path = "./evaluation/test_data/qwen_result.txt"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        
        # Check video duration and display info
        duration = get_video_duration(args.video_path)
        if duration:
            print(f"📹 Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            if duration > args.segment_duration and not args.disable_segmentation:
                print(f"🔪 Will segment into {args.segment_duration}s chunks")
        
        # Single-threaded translation with segmentation
        translation = translate_video_with_segmentation_worker(
            args.video_path, 
            os.path.basename(args.video_path),
            args.model_path, 
            args.source_lang, 
            args.target_lang,
            args.use_audio,
            args.save_audio,
            args.use_flash_attn,
            args.segment_duration,
            args.overlap_duration,
            args.disable_segmentation,
            False,  # No shared model for single file
            1
        )
        
        # Save translation to file
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(translation)
        
        print(f"✅ Translation saved to {args.output_path}")
        print("\n📝 Translation result:")
        print("-" * 50)
        print(translation)
        print("-" * 50)
        
    else:
        # Recursive parallel processing mode
        print(f"🚀 Starting recursive parallel processing for folder: {args.input_folder}")
        
        video_files_info = get_video_files_recursive(args.input_folder, args.max_depth)
        
        if not video_files_info:
            print("No video files found to process. Exiting.")
            return
            
        print(f"📁 Found {len(video_files_info)} video files to process")
        
        if not args.output_path:
            args.output_path = "./evaluation/test_data/recursive_parallel_results"
        
        os.makedirs(args.output_path, exist_ok=True)
        
        # Execute recursive parallel translation
        stats = recursive_parallel_translate(
            video_files_info,
            args.output_path,
            args.model_path,
            args.source_lang,
            args.target_lang,
            args.use_audio,
            args.save_audio,
            args.use_flash_attn,
            args.preserve_structure,
            args.segment_duration,
            args.overlap_duration,
            args.disable_segmentation,
            args.parallel_workers,
            args.shared_model,
            args.max_concurrent_gpu
        )

if __name__ == "__main__":
    main() 