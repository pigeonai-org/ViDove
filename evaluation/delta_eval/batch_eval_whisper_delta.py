# DEPENDENCIES FOR LOCAL WHISPER:
# - stable-ts: pip install stable-ts
# - torch: pip install torch torchvision torchaudio
# 
# For local Whisper usage, make sure you have sufficient GPU memory (8GB+ recommended for large models)
# To use local Whisper and avoid 25MB API file size limits, add --use-local-whisper flag

import os
import sys
import json
import subprocess
import argparse
import re
import shutil
import traceback
from pathlib import Path

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to get to ViDove root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules after path setup
try:
    from src.audio.ASR import WhisperAPIASR
    from src.audio.ASR import StableWhisperASR
    from evaluation.utils.dataset_parser.to_big_video_format import to_big_video_format
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you are running from the project root directory")
    sys.exit(1)

# Try to import scoring modules (some may not be available)
try:
    from evaluation.scores.score import BLEUscore, COMETscore, SubERscore, SubSONARscore
    SCORING_AVAILABLE = True
    SUBER_AVAILABLE = True
    SUBSONAR_AVAILABLE = True
    print("All scoring modules imported successfully")
except ImportError as e:
    print(f"Warning: Some scoring modules not available: {e}")
    print("Trying individual imports...")
    
    # Try individual imports
    try:
        from evaluation.scores.score import BLEUscore, COMETscore
        SCORING_AVAILABLE = True
    except ImportError:
        SCORING_AVAILABLE = False
        
    try:
        from evaluation.scores.score import SubERscore
        SUBER_AVAILABLE = True
    except ImportError:
        SUBER_AVAILABLE = False
        
    try:
        from evaluation.scores.score import SubSONARscore
        SUBSONAR_AVAILABLE = True
    except ImportError:
        SUBSONAR_AVAILABLE = False
        
    if not SCORING_AVAILABLE:
        print("Running without evaluation scoring")
        
        # Define dummy scoring functions
        class DummyScore:
            def __init__(self, score=0.0):
                self.score = score
        
        def BLEUscore(sys, refs):
            return DummyScore(0.0)
        
        def COMETscore(src, mt, ref):
            return DummyScore(0.0)
    
    if not SUBER_AVAILABLE:
        def SubERscore(hyp_file, ref_file):
            return 0.0
            
    if not SUBSONAR_AVAILABLE:
        def SubSONARscore(hyp_file, ref_file):
            return 0.0

# Dataset paths
BIGVIDEO_DATA_PATH = Path("./evaluation_experiment/BigVideo-test")
DOVEBENCH_DATA_PATH = Path("./evaluation_experiment/DoveBench")

def extract_audio(video_path, output_path, max_size_mb=25, use_local_whisper=False):
    """
    Extract audio from video file using ffmpeg with size optimization.
    
    Args:
        video_path (str or Path): Path to the video file.
        output_path (str or Path): Path to save the extracted audio.
        max_size_mb (int): Maximum file size in MB for OpenAI API compatibility.
        use_local_whisper (bool): If True, skip size restrictions (local whisper can handle large files).
    
    Returns:
        Path: Path to the extracted audio file if successful, None otherwise.
    """
    try:
        # Choose audio quality based on whether we're using local whisper
        if use_local_whisper:
            # Higher quality for local whisper (no API size limits)
            print("Using higher quality audio extraction for local Whisper...")
            subprocess.run(
                [
                    "ffmpeg", "-y",  # -y to overwrite output files
                    "-i", str(video_path),
                    "-f", "mp3",
                    "-ab", "192000",  # Higher bitrate: 192kbps
                    "-ar", "44100",   # Higher sample rate: 44.1kHz
                    "-ac", "1",       # Mono
                    "-vn",  # no video
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
        else:
            # Lower quality for API whisper (size constraints)
            print("Using compressed audio extraction for API Whisper...")
            subprocess.run(
                [
                    "ffmpeg", "-y",  # -y to overwrite output files
                    "-i", str(video_path),
                    "-f", "mp3",
                    "-ab", "128000",  # Lower bitrate: 128kbps
                    "-ar", "32000",   # Lower sample rate: 32kHz
                    "-ac", "1",       # Mono
                    "-vn",  # no video
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
        
        # Check file size
        output_path = Path(output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Audio extracted: {output_path} ({file_size_mb:.1f} MB)")
        
        # If using local whisper, we can handle large files
        if use_local_whisper:
            print(f"✅ Using local Whisper - file size {file_size_mb:.1f}MB is acceptable")
            return output_path
        
        # For API whisper, apply size restrictions
        if file_size_mb > max_size_mb:
            print(f"File too large ({file_size_mb:.1f} MB), trying lower quality...")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-f", "mp3",
                    "-ab", "32000",  # Even lower bitrate: 32kbps
                    "-ar", "16000",
                    "-ac", "1",
                    "-vn",
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Compressed audio: {output_path} ({file_size_mb:.1f} MB)")
        
        # If still too large for API, split the audio
        if not use_local_whisper and file_size_mb > max_size_mb:
            print(f"File still too large ({file_size_mb:.1f} MB), splitting audio...")
            print("💡 Tip: Use --use-local-whisper to avoid file size limits")
            return split_audio_file(output_path, max_size_mb)
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio from {video_path}: {e}")
        return None

def audio_extractor(video_path, use_local_whisper=False):
    """
    Extract audio from video files for a single video.
    
    Args:
        video_path (str or Path): Path to the video file.
        use_local_whisper (bool): If True, use higher quality extraction for local whisper.
    
    Returns:
        Path or List[Path]: Path(s) to the extracted audio file(s).
    """
    video_path = Path(video_path)
    video_name = video_path.stem  # Get the base name without extension
    output_path = video_path.parent / f"{video_name}.mp3"
    
    result = extract_audio(video_path, output_path, use_local_whisper=use_local_whisper)
    return result

def whisper_transcription(audio_paths, source_lang="en", use_local_whisper=False):
    """
    Transcribe audio using Whisper ASR. Handles both single files and multiple chunks.
    
    Args:
        audio_paths (str, Path, or List): Path(s) to the audio file(s).
        source_lang (str): Source language code.
        use_local_whisper (bool): If True, use StableWhisperASR (local), otherwise use WhisperAPIASR.
    
    Returns:
        str: Transcription result in SRT format.
    """
    try:
        if use_local_whisper:
            print("Using local Whisper model (StableWhisperASR)...")
            asr = StableWhisperASR()
        else:
            print("Using Whisper API (WhisperAPIASR)...")
            asr = WhisperAPIASR()
        
        # Handle single file or list of files
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        
        all_transcriptions = []
        cumulative_time_offset = 0.0
        
        for i, audio_path in enumerate(audio_paths):
            print(f"Transcribing chunk {i+1}/{len(audio_paths)}: {audio_path}")
            
            # Get chunk duration for time offset calculation
            if len(audio_paths) > 1:
                try:
                    result = subprocess.run(
                        [
                            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(audio_path)
                        ],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    chunk_duration = float(result.stdout.strip())
                except Exception:
                    chunk_duration = 300.0  # Default to 5 minutes if can't get duration
            else:
                chunk_duration = 0.0
            
            # Check file size and recommend local whisper for large files
            audio_file_size = Path(audio_path).stat().st_size / (1024 * 1024)  # Size in MB
            if not use_local_whisper and audio_file_size > 25:
                print(f"⚠️ Warning: Audio file is {audio_file_size:.1f}MB (>25MB limit for API)")
                print("   Consider using --use-local-whisper flag to avoid file size limits")
            
            transcription = asr.get_transcript(str(audio_path), source_lang=source_lang)
            
            if transcription:
                # Convert different formats to SRT format
                if use_local_whisper:
                    # StableWhisperASR returns segments format, convert to SRT
                    try:
                        transcription = convert_segments_to_srt(transcription, cumulative_time_offset)
                    except Exception as convert_error:
                        print(f"Error converting segments to SRT: {convert_error}")
                        print(f"Transcription format: {type(transcription)}")
                        if isinstance(transcription, list) and len(transcription) > 0:
                            print(f"First segment example: {transcription[0]}")
                        return None
                else:
                    # WhisperAPIASR returns SRT format, adjust timestamps if needed
                    if cumulative_time_offset > 0:
                        transcription = adjust_srt_timestamps(transcription, cumulative_time_offset)
                
                all_transcriptions.append(transcription)
                cumulative_time_offset += chunk_duration
                print(f"Transcription successful for chunk {i+1}")
            else:
                print(f"Failed to transcribe chunk {i+1}: {audio_path}")
                print(f"ASR returned: {transcription}")
                return None
        
        # Combine all transcriptions
        if len(all_transcriptions) == 1:
            return all_transcriptions[0]
        else:
            return combine_srt_transcriptions(all_transcriptions)
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def save_srt_transcription(transcription, output_path):
    """
    Save transcription to SRT file.
    
    Args:
        transcription (str): Transcription in SRT format.
        output_path (str or Path): Path to save the SRT file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"SRT file saved: {output_path}")
    except Exception as e:
        print(f"Error saving SRT file {output_path}: {e}")


def translate_srt_file(srt_path, src_lang="en", tgt_lang="zh", task_cfg=None):
    """
    Translate an SRT file using DocMTAgent.
    
    Args:
        srt_path (str or Path): Path to the SRT file.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        task_cfg (dict): Task configuration (not used for DocMTAgent).
    
    Returns:
        Path: Path to the translated SRT file if successful, None otherwise.
    """
    try:
        srt_path = Path(srt_path)
        
        # Convert SRT to plain text for DocMTAgent input
        text_input_path = srt_path.parent / f"{srt_path.stem}_input.txt"
        text_lines = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        # Parse SRT and extract text lines (simple parsing)
        # Extract subtitle text (ignore timestamps and sequence numbers)
        subtitle_pattern = r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.+?)(?=\n\d+\n|\n\n|\Z)'
        matches = re.findall(subtitle_pattern, srt_content, re.DOTALL)
        
        for match in matches:
            # Clean up the text (remove extra newlines)
            clean_text = ' '.join(match.strip().split('\n'))
            if clean_text:
                text_lines.append(clean_text)
        
        # Save text input for DocMTAgent (format it properly)
        # Each line should be a single sentence for DocMTAgent
        formatted_lines = []
        for line in text_lines:
            # Split long lines into sentences if needed
            sentences = line.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not sentence.endswith('.') and sentence != sentences[-1]:
                    sentence += '.'
                if sentence:
                    formatted_lines.append(sentence)
        
        with open(text_input_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(formatted_lines))
        
        print(f"Converted SRT to text: {text_input_path}")
        
        # Run DocMTAgent translation
        docmt_script = Path("./evaluation/delta_eval/DocMTAgent/demo/run_gpt.sh")
        if not docmt_script.exists():
            print(f"DocMTAgent script not found: {docmt_script}")
            return None
        
        # Set up language pair
        lang_pair = f"{src_lang}-{tgt_lang}"
        
        # Run the DocMTAgent script
        print(f"Running DocMTAgent translation for {lang_pair}...")
        
        # Change to DocMTAgent demo directory for execution
        original_cwd = os.getcwd()
        docmt_demo_dir = Path("./evaluation/delta_eval/DocMTAgent/demo")
        
        # Get absolute path of input file before changing directory
        abs_input_path = text_input_path.resolve()
        
        try:
            os.chdir(docmt_demo_dir)
            
            # Make sure the input file path is accessible from the demo directory
            print(f"Input file exists: {abs_input_path.exists()}")
            print(f"Input file absolute path: {abs_input_path}")
            
            # Run the translation script
            result = subprocess.run(
                ["bash", "run_gpt.sh", lang_pair, "gpt4omini", str(abs_input_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"DocMTAgent stdout: {result.stdout}")
            if result.stderr:
                print(f"DocMTAgent stderr: {result.stderr}")
            print(f"DocMTAgent return code: {result.returncode}")
            
            # Extract output path from the script output
            output_match = re.search(r'OUTPUT=(.+)', result.stdout)
            if not output_match:
                print("Could not find output file path in DocMTAgent output")
                print(f"Full stdout: {result.stdout}")
                # Check if the script failed
                if result.returncode != 0:
                    print(f"DocMTAgent failed with return code: {result.returncode}")
                    if result.stderr:
                        print(f"Error details: {result.stderr}")
                return None
            
            translated_text_path = Path(output_match.group(1))
            
        finally:
            os.chdir(original_cwd)
        
        if not translated_text_path.exists():
            print(f"Translated text file not found: {translated_text_path}")
            return None
        
        # Read translated text
        with open(translated_text_path, 'r', encoding='utf-8') as f:
            translated_lines = f.readlines()
        
        # Convert back to SRT format
        translated_srt_path = srt_path.parent / f"{srt_path.stem}_translated.srt"
        
        # Parse original SRT for timing information
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract timing information from original SRT
        timing_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n'
        timings = re.findall(timing_pattern, content)
        
        # Create new SRT with translated text
        new_srt_content = []
        for i, (seq_num, timing) in enumerate(timings):
            if i < len(translated_lines):
                translated_text = translated_lines[i].strip()
                new_srt_content.append(f"{seq_num}\n{timing}\n{translated_text}\n")
        
        # Save translated SRT
        with open(translated_srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_srt_content))
        
        print(f"Translation completed: {translated_srt_path}")
        return translated_srt_path
        
    except Exception as e:
        print(f"Error translating SRT file {srt_path}: {e}")
        return None

def process_bigvideo_dataset(test_ids_file, video_dir, output_dir, use_local_whisper=False):
    """
    Process BigVideo dataset: extract audio, transcribe, translate, and evaluate.
    
    Args:
        test_ids_file (str or Path): Path to the test IDs file.
        video_dir (str or Path): Directory containing video files.
        output_dir (str or Path): Output directory for results.
        use_local_whisper (bool): If True, use local Whisper model instead of API.
    
    Returns:
        dict: Evaluation results.
    """
    test_ids_file = Path(test_ids_file)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read test IDs
    with open(test_ids_file, 'r', encoding='utf-8') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(test_ids)} test IDs to process")
    if use_local_whisper:
        print("🏠 Using local Whisper model (no file size limits)")
    else:
        print("☁️ Using Whisper API (25MB file size limit)")
    
    results = {
        "processed_files": [],
        "failed_files": [],
        "translations": {},
        "original_texts": {},
        "reference_texts": {}
    }
    
    # Load reference texts
    ref_en_file = BIGVIDEO_DATA_PATH / "text_data_test.en"
    ref_zh_file = BIGVIDEO_DATA_PATH / "text_data_test.zh"
    
    ref_en_texts = {}
    ref_zh_texts = {}
    
    if ref_en_file.exists():
        with open(ref_en_file, 'r', encoding='utf-8') as f:
            ref_en_lines = f.readlines()
    
    if ref_zh_file.exists():
        with open(ref_zh_file, 'r', encoding='utf-8') as f:
            ref_zh_lines = f.readlines()
    
    # Map reference texts to IDs
    for i, test_id in enumerate(test_ids):
        if i < len(ref_en_lines):
            ref_en_texts[test_id] = ref_en_lines[i].strip()
        if i < len(ref_zh_lines):
            ref_zh_texts[test_id] = ref_zh_lines[i].strip()
    
    for test_id in test_ids:
        try:
            print(f"\nProcessing {test_id}...")
            
            # Find video file
            video_file = video_dir / f"{test_id}.mp4"
            if not video_file.exists():
                print(f"Video file not found: {video_file}")
                results["failed_files"].append(test_id)
                continue
            
            # Extract audio
            audio_file = audio_extractor(video_file, use_local_whisper=use_local_whisper)
            if not audio_file:
                print(f"Audio extraction failed for {test_id}")
                results["failed_files"].append(test_id)
                continue
            
            # Handle case where audio_file might be a list (split files)
            if isinstance(audio_file, list):
                # Check if any of the chunks exist
                if not any(f.exists() for f in audio_file):
                    print(f"Audio extraction failed for {test_id}")
                    results["failed_files"].append(test_id)
                    continue
            else:
                if not audio_file.exists():
                    print(f"Audio extraction failed for {test_id}")
                    results["failed_files"].append(test_id)
                    continue
            
            # Transcribe audio
            transcription = whisper_transcription(audio_file, source_lang="en", use_local_whisper=use_local_whisper)
            if not transcription:
                print(f"Transcription failed for {test_id}")
                results["failed_files"].append(test_id)
                continue
            
            # Save transcription as SRT file
            srt_file = output_dir / f"{test_id}.srt"
            save_srt_transcription(transcription, srt_file)
            
            # Translate SRT file
            translated_srt_path = translate_srt_file(srt_file, src_lang="en", tgt_lang="zh")
            if not translated_srt_path:
                print(f"Translation failed for {test_id}")
                results["failed_files"].append(test_id)
                continue
            
            # Convert to BigVideo format (single line)
            original_text = to_big_video_format(srt_file)
            
            # For translated text, extract from the translated SRT file
            translated_text = to_big_video_format(translated_srt_path)
            
            results["translations"][test_id] = translated_text
            results["original_texts"][test_id] = original_text
            results["reference_texts"][test_id] = ref_zh_texts.get(test_id, "")
            results["processed_files"].append(test_id)
            
            print(f"Successfully processed {test_id}")
            
        except Exception as e:
            print(f"Error processing {test_id}: {e}")
            results["failed_files"].append(test_id)
    
    return results

def process_dovebench_dataset(dataset_dir, output_dir, use_local_whisper=False):
    """
    Process DoveBench dataset: extract audio, transcribe, translate, and evaluate.
    
    Args:
        dataset_dir (str or Path): Directory containing DoveBench data.
        output_dir (str or Path): Output directory for results.
        use_local_whisper (bool): If True, use local Whisper model instead of API.
    
    Returns:
        dict: Evaluation results.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_local_whisper:
        print("🏠 Using local Whisper model for DoveBench processing")
    else:
        print("☁️ Using Whisper API for DoveBench processing")
    
    results = {
        "processed_files": [],
        "failed_files": [],
        "translations": {},
        "original_texts": {},
        "reference_texts": {}
    }
    
    # Process each domain (CS, manga, sc2)
    for domain_dir in dataset_dir.iterdir():
        if not domain_dir.is_dir() or domain_dir.name.startswith('.'):
            continue
        
        print(f"\nProcessing domain: {domain_dir.name}")
        domain_output = output_dir / domain_dir.name
        domain_output.mkdir(parents=True, exist_ok=True)
        
        # Process each video in the domain
        for video_dir in domain_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            try:
                print(f"Processing {video_dir.name}...")
                
                # Find video file
                video_files = list(video_dir.glob("*.mp4"))
                if not video_files:
                    print(f"No video file found in {video_dir}")
                    results["failed_files"].append(str(video_dir))
                    continue
                
                video_file = video_files[0]
                video_id = f"{domain_dir.name}/{video_dir.name}"
                
                # Extract audio
                audio_file = audio_extractor(video_file, use_local_whisper=use_local_whisper)
                if not audio_file:
                    print(f"Audio extraction failed for {video_id}")
                    results["failed_files"].append(video_id)
                    continue
                
                # Handle case where audio_file might be a list (split files)
                if isinstance(audio_file, list):
                    # Check if any of the chunks exist
                    if not any(f.exists() for f in audio_file):
                        print(f"Audio extraction failed for {video_id}")
                        results["failed_files"].append(video_id)
                        continue
                else:
                    if not audio_file.exists():
                        print(f"Audio extraction failed for {video_id}")
                        results["failed_files"].append(video_id)
                        continue
                
                # Transcribe audio
                transcription = whisper_transcription(audio_file, source_lang="en", use_local_whisper=use_local_whisper)
                if not transcription:
                    print(f"Transcription failed for {video_id}")
                    results["failed_files"].append(video_id)
                    continue
                
                # Save transcription as SRT file
                srt_file = domain_output / f"{video_dir.name}.srt"
                save_srt_transcription(transcription, srt_file)
                
                # Translate SRT file
                translated_srt_path = translate_srt_file(srt_file, src_lang="en", tgt_lang="zh")
                if not translated_srt_path:
                    print(f"Translation failed for {video_id}")
                    results["failed_files"].append(video_id)
                    continue
                
                # Copy translated SRT to the original data folder
                translated_srt_file = video_dir / f"{video_dir.name}_translated.srt"
                shutil.copy2(translated_srt_path, translated_srt_file)
                
                # Load reference translation if exists
                ref_files = list(video_dir.glob("*_ZH.ass"))
                ref_text = ""
                if ref_files:
                    # Parse reference ASS file - simplified parsing
                    try:
                        with open(ref_files[0], 'r', encoding='utf-8') as f:
                            ass_content = f.read()
                            # Extract dialogue lines (this is a simplified approach)
                            dialogue_lines = re.findall(r'Dialogue:.*?,(.*)', ass_content)
                            ref_text = ' '.join(dialogue_lines).strip()
                    except Exception as e:
                        print(f"Error reading reference file {ref_files[0]}: {e}")
                
                # Convert to text format
                original_text = to_big_video_format(srt_file)  # Original transcription
                translated_text = to_big_video_format(translated_srt_path)  # Translated text
                
                results["translations"][video_id] = translated_text
                results["original_texts"][video_id] = original_text
                results["reference_texts"][video_id] = ref_text
                results["processed_files"].append(video_id)
                
                print(f"Successfully processed {video_id}")
                
            except Exception as e:
                print(f"Error processing {video_dir}: {e}")
                results["failed_files"].append(str(video_dir))
    
    return results

def calculate_bigvideo_scores(results, output_dir=None):
    """
    Calculate BLEU, dCOMET, SubER, and SubSONAR scores for BigVideo dataset.
    
    Args:
        results (dict): Processing results containing translations and references.
        output_dir (Path): Output directory containing SRT files for SubER/SubSONAR.
    
    Returns:
        dict: Evaluation scores.
    """
    translations = []
    references = []
    sources = []
    srt_pairs = []  # For SubER and SubSONAR
    
    for test_id in results["processed_files"]:
        if test_id in results["translations"] and test_id in results["reference_texts"]:
            translations.append(results["translations"][test_id])
            references.append(results["reference_texts"][test_id])
            sources.append(results["original_texts"].get(test_id, ""))
            
            # Collect SRT file pairs for SubER/SubSONAR
            if output_dir:
                translated_srt = output_dir / f"{test_id}_translated.srt"
                original_srt = output_dir / f"{test_id}.srt"
                if translated_srt.exists() and original_srt.exists():
                    srt_pairs.append((str(translated_srt), str(original_srt)))
    
    if not translations:
        print("No valid translations found for evaluation")
        return {}
    
    print(f"Calculating scores for {len(translations)} translations...")
    
    scores = {}
    
    try:
        # BLEU score
        bleu_score = BLEUscore(translations, [references])
        scores["BLEU"] = bleu_score.score
        print(f"BLEU Score: {bleu_score.score:.4f}")
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
    
    try:
        # COMET score (dCOMET)
        if sources:
            comet_scores = COMETscore(sources, translations, references)
            scores["dCOMET"] = sum(comet_scores.scores) / len(comet_scores.scores)
            print(f"dCOMET Score: {scores['dCOMET']:.4f}")
    except Exception as e:
        print(f"Error calculating COMET score: {e}")
    
    # SubER scoring
    if SUBER_AVAILABLE and srt_pairs:
        try:
            print("Calculating SubER scores...")
            suber_scores = []
            for hyp_file, ref_file in srt_pairs:
                try:
                    score = SubERscore(hyp_file, ref_file)
                    suber_scores.append(score)
                    print(f"SubER for {Path(hyp_file).name}: {score:.4f}")
                except Exception as e:
                    print(f"Error calculating SubER for {hyp_file}: {e}")
            
            if suber_scores:
                scores["SubER"] = sum(suber_scores) / len(suber_scores)
                print(f"Average SubER Score: {scores['SubER']:.4f}")
        except Exception as e:
            print(f"Error calculating SubER scores: {e}")
    
    # SubSONAR scoring
    if SUBSONAR_AVAILABLE and srt_pairs:
        try:
            print("Calculating SubSONAR scores...")
            subsonar_scores = []
            for hyp_file, ref_file in srt_pairs:
                try:
                    score = SubSONARscore(hyp_file, ref_file)
                    subsonar_scores.append(score)
                    print(f"SubSONAR for {Path(hyp_file).name}: {score:.4f}")
                except Exception as e:
                    print(f"Error calculating SubSONAR for {hyp_file}: {e}")
            
            if subsonar_scores:
                scores["SubSONAR"] = sum(subsonar_scores) / len(subsonar_scores)
                print(f"Average SubSONAR Score: {scores['SubSONAR']:.4f}")
        except Exception as e:
            print(f"Error calculating SubSONAR scores: {e}")
    
    return scores

def calculate_dovebench_scores(results, output_dir=None):
    """
    Calculate BLEU, dCOMET, SubER, and SubSORNAR scores for DoveBench dataset.
    
    Args:
        results (dict): Processing results containing translations and references.
        output_dir (Path): Output directory containing SRT files for SubER/SubSONAR.
    
    Returns:
        dict: Evaluation scores.
    """
    translations = []
    references = []
    sources = []
    srt_pairs = []  # For SubER and SubSONAR
    
    for video_id in results["processed_files"]:
        if video_id in results["translations"] and results["reference_texts"].get(video_id):
            translations.append(results["translations"][video_id])
            references.append(results["reference_texts"][video_id])
            sources.append(results["original_texts"].get(video_id, ""))
            
            # Collect SRT file pairs for SubER/SubSONAR
            if output_dir:
                domain, video_name = video_id.split('/')
                translated_srt = output_dir / domain / f"{video_name}_translated.srt"
                original_srt = output_dir / domain / f"{video_name}.srt"
                if translated_srt.exists() and original_srt.exists():
                    srt_pairs.append((str(translated_srt), str(original_srt)))
    
    if not translations:
        print("No valid translations found for evaluation")
        return {}
    
    print(f"Calculating scores for {len(translations)} translations...")
    
    scores = {}
    
    try:
        # BLEU score
        bleu_score = BLEUscore(translations, [references])
        scores["BLEU"] = bleu_score.score
        print(f"BLEU Score: {bleu_score.score:.4f}")
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
    
    try:
        # COMET score (dCOMET)
        if sources:
            comet_scores = COMETscore(sources, translations, references)
            scores["dCOMET"] = sum(comet_scores.scores) / len(comet_scores.scores)
            print(f"dCOMET Score: {scores['dCOMET']:.4f}")
    except Exception as e:
        print(f"Error calculating COMET score: {e}")
    
    # SubER scoring
    if SUBER_AVAILABLE and srt_pairs:
        try:
            print("Calculating SubER scores...")
            suber_scores = []
            for hyp_file, ref_file in srt_pairs:
                try:
                    # Create a reference SRT from the reference text
                    # For now, we'll use the original transcription as reference
                    # In a real scenario, you'd have proper reference SRT files
                    score = SubERscore(hyp_file, ref_file)
                    suber_scores.append(score)
                    print(f"SubER for {Path(hyp_file).name}: {score:.4f}")
                except Exception as e:
                    print(f"Error calculating SubER for {hyp_file}: {e}")
            
            if suber_scores:
                scores["SubER"] = sum(suber_scores) / len(suber_scores)
                print(f"Average SubER Score: {scores['SubER']:.4f}")
        except Exception as e:
            print(f"Error calculating SubER scores: {e}")
    
    # SubSONAR scoring
    if SUBSONAR_AVAILABLE and srt_pairs:
        try:
            print("Calculating SubSONAR scores...")
            subsonar_scores = []
            for hyp_file, ref_file in srt_pairs:
                try:
                    score = SubSONARscore(hyp_file, ref_file)
                    subsonar_scores.append(score)
                    print(f"SubSONAR for {Path(hyp_file).name}: {score:.4f}")
                except Exception as e:
                    print(f"Error calculating SubSONAR for {hyp_file}: {e}")
            
            if subsonar_scores:
                scores["SubSONAR"] = sum(subsonar_scores) / len(subsonar_scores)
                print(f"Average SubSONAR Score: {scores['SubSONAR']:.4f}")
        except Exception as e:
            print(f"Error calculating SubSONAR scores: {e}")
    
    return scores

def save_results(results, scores, output_file):
    """
    Save evaluation results to JSON file.
    
    Args:
        results (dict): Processing results.
        scores (dict): Evaluation scores.
        output_file (str or Path): Path to save results.
    """
    output_data = {
        "evaluation_scores": scores,
        "processing_stats": {
            "total_files": len(results["processed_files"]) + len(results["failed_files"]),
            "processed_successfully": len(results["processed_files"]),
            "failed": len(results["failed_files"]),
            "success_rate": len(results["processed_files"]) / (len(results["processed_files"]) + len(results["failed_files"])) if (len(results["processed_files"]) + len(results["failed_files"])) > 0 else 0
        },
        "processed_files": results["processed_files"],
        "failed_files": results["failed_files"]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

def split_audio_file(audio_path, max_size_mb=20):
    """
    Split audio file into smaller chunks that fit OpenAI's size limit.
    
    Args:
        audio_path (Path): Path to the audio file to split.
        max_size_mb (int): Maximum file size in MB.
    
    Returns:
        List[Path]: List of paths to the split audio files.
    """
    try:
        audio_path = Path(audio_path)
        
        # Get audio duration
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        duration = float(result.stdout.strip())
        
        # Calculate number of chunks needed
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        num_chunks = int(file_size_mb / max_size_mb) + 1
        chunk_duration = duration / num_chunks
        
        print(f"Splitting {audio_path.name} into {num_chunks} chunks of {chunk_duration:.1f}s each")
        
        chunk_files = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_file = audio_path.parent / f"{audio_path.stem}_chunk_{i+1}.mp3"
            
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(audio_path),
                    "-ss", str(start_time),
                    "-t", str(chunk_duration),
                    "-f", "mp3",
                    "-ab", "32000",
                    "-ar", "16000",
                    "-ac", "1",
                    str(chunk_file)
                ],
                check=True,
                capture_output=True
            )
            chunk_files.append(chunk_file)
            print(f"Created chunk: {chunk_file}")
        
        return chunk_files
    except Exception as e:
        print(f"Error splitting audio file {audio_path}: {e}")
        return [audio_path]  # Return original file if splitting fails

def adjust_srt_timestamps(srt_content, time_offset_seconds):
    """
    Adjust SRT timestamps by adding a time offset.
    
    Args:
        srt_content (str): SRT content as string.
        time_offset_seconds (float): Time offset in seconds to add.
    
    Returns:
        str: SRT content with adjusted timestamps.
    """
    
    def adjust_timestamp(match):
        start_time = match.group(1)
        end_time = match.group(2)
        
        # Parse timestamps
        def parse_time(time_str):
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split(',')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            return total_seconds
        
        def format_time(total_seconds):
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            milliseconds = int((total_seconds % 1) * 1000)
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        
        start_seconds = parse_time(start_time) + time_offset_seconds
        end_seconds = parse_time(end_time) + time_offset_seconds
        
        return f"{format_time(start_seconds)} --> {format_time(end_seconds)}"
    
    # Adjust all timestamps
    timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
    adjusted_content = re.sub(timestamp_pattern, adjust_timestamp, srt_content)
    
    return adjusted_content

def combine_srt_transcriptions(transcriptions):
    """
    Combine multiple SRT transcriptions into one.
    
    Args:
        transcriptions (List[str]): List of SRT content strings.
    
    Returns:
        str: Combined SRT content.
    """
    
    combined_lines = []
    subtitle_counter = 1
    
    for transcription in transcriptions:
        # Parse each transcription
        subtitle_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n\n|\Z)'
        matches = re.findall(subtitle_pattern, transcription, re.DOTALL)
        
        for _, timing, text in matches:
            combined_lines.append(f"{subtitle_counter}\n{timing}\n{text.strip()}\n")
            subtitle_counter += 1
    
    return '\n'.join(combined_lines)

def convert_segments_to_srt(segments, time_offset=0.0):
    """
    Convert segments format (from StableWhisperASR) to SRT format.
    
    Args:
        segments (list): List of segment dictionaries with 'start', 'end', 'text' keys.
        time_offset (float): Time offset in seconds to add to all timestamps.
    
    Returns:
        str: SRT formatted transcription.
    """
    if not segments:
        print("Warning: Empty segments list provided")
        return ""
    
    if not isinstance(segments, list):
        print(f"Warning: Expected list of segments, got {type(segments)}")
        return ""
    
    srt_content = []
    entry_counter = 1
    
    for i, segment in enumerate(segments):
        try:
            # Handle different segment formats
            if isinstance(segment, dict):
                # Standard format: {'start': float, 'end': float, 'text': str}
                start_time = float(segment.get('start', 0.0)) + time_offset
                end_time = float(segment.get('end', start_time + 1.0)) + time_offset
                text = str(segment.get('text', '')).strip()
            elif hasattr(segment, 'start') and hasattr(segment, 'end') and hasattr(segment, 'text'):
                # Object format with attributes
                start_time = float(getattr(segment, 'start', 0.0)) + time_offset
                end_time = float(getattr(segment, 'end', start_time + 1.0)) + time_offset
                text = str(getattr(segment, 'text', '')).strip()
            else:
                print(f"Warning: Unrecognized segment format at index {i}: {type(segment)}")
                continue
            
            if not text:
                continue
            
            # Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
            def format_timestamp(seconds):
                if seconds < 0:
                    seconds = 0
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                milliseconds = int((seconds % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
            
            # Create SRT entry
            srt_entry = f"{entry_counter}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n{text}\n"
            srt_content.append(srt_entry)
            entry_counter += 1
            
        except Exception as e:
            print(f"Warning: Error processing segment {i}: {e}")
            print(f"Segment content: {segment}")
            continue
    
    if not srt_content:
        print("Warning: No valid segments could be converted to SRT")
        return ""
    
    return '\n'.join(srt_content)

def main(use_local_whisper=False):
    """
    Main evaluation function for DocMTAgent on DoveBench and BigVideo datasets.
    
    Args:
        use_local_whisper (bool): If True, use StableWhisperASR instead of OpenAI API.
    """
    print("Starting DocMTAgent evaluation...")
    if use_local_whisper:
        print("🔧 Using local Whisper model (StableWhisperASR) - no file size limits")
    else:
        print("🌐 Using OpenAI Whisper API - 25MB file size limit applies")
    
    # Create output directories
    output_dir = Path("./evaluation_results")
    bigvideo_output = output_dir / "bigvideo"
    dovebench_output = output_dir / "dovebench"
    
    # Process BigVideo dataset
    print("\n" + "="*50)
    print("Processing BigVideo dataset...")
    print("="*50)
    
    bigvideo_results = process_bigvideo_dataset(
        test_ids_file=BIGVIDEO_DATA_PATH / "text_data_test.id",
        video_dir=BIGVIDEO_DATA_PATH / "test",
        output_dir=bigvideo_output,
        use_local_whisper=use_local_whisper
    )
    
    bigvideo_scores = calculate_bigvideo_scores(bigvideo_results, bigvideo_output)
    save_results(bigvideo_results, bigvideo_scores, bigvideo_output / "evaluation_results.json")
    
    # Process DoveBench dataset
    print("\n" + "="*50)
    print("Processing DoveBench dataset...")
    print("="*50)
    
    dovebench_results = process_dovebench_dataset(
        dataset_dir=DOVEBENCH_DATA_PATH,
        output_dir=dovebench_output,
        use_local_whisper=use_local_whisper
    )
    
    dovebench_scores = calculate_dovebench_scores(dovebench_results, dovebench_output)
    save_results(dovebench_results, dovebench_scores, dovebench_output / "evaluation_results.json")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print("\nBigVideo Dataset:")
    print(f"  Processed: {len(bigvideo_results['processed_files'])} files")
    print(f"  Failed: {len(bigvideo_results['failed_files'])} files")
    if bigvideo_scores:
        for metric, score in bigvideo_scores.items():
            print(f"  {metric}: {score:.4f}")
    
    print("\nDoveBench Dataset:")
    print(f"  Processed: {len(dovebench_results['processed_files'])} files")
    print(f"  Failed: {len(dovebench_results['failed_files'])} files")
    if dovebench_scores:
        for metric, score in dovebench_scores.items():
            print(f"  {metric}: {score:.4f}")
    
    print(f"\nResults saved to: {output_dir}")

def find_missing_files_bigvideo(test_ids_file, output_dir):
    """
    Find missing translation files for BigVideo dataset by comparing with ID file.
    
    Args:
        test_ids_file (str or Path): Path to the test IDs file.
        output_dir (str or Path): Output directory containing existing results.
    
    Returns:
        tuple: (all_test_ids, missing_ids, existing_ids)
    """
    test_ids_file = Path(test_ids_file)
    output_dir = Path(output_dir)
    
    # Read all test IDs
    with open(test_ids_file, 'r', encoding='utf-8') as f:
        all_test_ids = [line.strip() for line in f if line.strip()]
    
    # Check which files already have translations
    existing_ids = []
    missing_ids = []
    
    for test_id in all_test_ids:
        translated_srt = output_dir / f"{test_id}_translated.srt"
        if translated_srt.exists():
            existing_ids.append(test_id)
        else:
            missing_ids.append(test_id)
    
    print("BigVideo dataset status:")
    print(f"  Total files: {len(all_test_ids)}")
    print(f"  Already processed: {len(existing_ids)}")
    print(f"  Missing: {len(missing_ids)}")
    
    if missing_ids:
        print(f"  Missing files: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")
    
    return all_test_ids, missing_ids, existing_ids

def find_missing_files_dovebench(dataset_dir, output_dir):
    """
    Find missing translation files for DoveBench dataset by checking video directories.
    
    Args:
        dataset_dir (str or Path): Directory containing DoveBench data.
        output_dir (str or Path): Output directory containing existing results.
    
    Returns:
        tuple: (all_video_ids, missing_ids, existing_ids)
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    all_video_ids = []
    existing_ids = []
    missing_ids = []
    
    # Scan all domains and videos
    for domain_dir in dataset_dir.iterdir():
        if not domain_dir.is_dir() or domain_dir.name.startswith('.'):
            continue
        
        for video_dir in domain_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            video_id = f"{domain_dir.name}/{video_dir.name}"
            all_video_ids.append(video_id)
            
            # Check if translation exists
            translated_srt = output_dir / domain_dir.name / f"{video_dir.name}_translated.srt"
            if translated_srt.exists():
                existing_ids.append(video_id)
            else:
                missing_ids.append(video_id)
    
    print("DoveBench dataset status:")
    print(f"  Total files: {len(all_video_ids)}")
    print(f"  Already processed: {len(existing_ids)}")
    print(f"  Missing: {len(missing_ids)}")
    
    if missing_ids:
        print(f"  Missing files: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")
    
    return all_video_ids, missing_ids, existing_ids

def resume_bigvideo_dataset(test_ids_file, video_dir, output_dir, use_local_whisper=False):
    """
    Resume BigVideo dataset processing: only process missing translation files.
    
    Args:
        test_ids_file (str or Path): Path to the test IDs file.
        video_dir (str or Path): Directory containing video files.
        output_dir (str or Path): Output directory for results.
        use_local_whisper (bool): If True, use local Whisper model instead of API.
    
    Returns:
        dict: Evaluation results.
    """
    test_ids_file = Path(test_ids_file)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔄 RESUMING BigVideo dataset processing...")
    
    # Find missing files
    all_test_ids, missing_ids, existing_ids = find_missing_files_bigvideo(test_ids_file, output_dir)
    
    if not missing_ids:
        print("✅ All files already processed! Loading existing results for evaluation...")
        
        # Load existing translations and apply metrics
        results = {
            "processed_files": existing_ids,
            "failed_files": [],
            "translations": {},
            "original_texts": {},
            "reference_texts": {}
        }
        
        # Load reference texts
        ref_en_file = BIGVIDEO_DATA_PATH / "text_data_test.en"
        ref_zh_file = BIGVIDEO_DATA_PATH / "text_data_test.zh"
        
        ref_en_texts = {}
        ref_zh_texts = {}
        
        if ref_en_file.exists():
            with open(ref_en_file, 'r', encoding='utf-8') as f:
                ref_en_lines = f.readlines()
        
        if ref_zh_file.exists():
            with open(ref_zh_file, 'r', encoding='utf-8') as f:
                ref_zh_lines = f.readlines()
        
        # Map reference texts to IDs
        for i, test_id in enumerate(all_test_ids):
            if i < len(ref_en_lines):
                ref_en_texts[test_id] = ref_en_lines[i].strip()
            if i < len(ref_zh_lines):
                ref_zh_texts[test_id] = ref_zh_lines[i].strip()
        
        # Load existing results
        for test_id in existing_ids:
            srt_file = output_dir / f"{test_id}.srt"
            translated_srt_file = output_dir / f"{test_id}_translated.srt"
            
            if srt_file.exists() and translated_srt_file.exists():
                original_text = to_big_video_format(srt_file)
                translated_text = to_big_video_format(translated_srt_file)
                
                results["translations"][test_id] = translated_text
                results["original_texts"][test_id] = original_text
                results["reference_texts"][test_id] = ref_zh_texts.get(test_id, "")
        
        return results
    
    if use_local_whisper:
        print("🏠 Using local Whisper model for missing files")
    else:
        print("☁️ Using Whisper API for missing files")
    
    # Process only missing files
    results = {
        "processed_files": list(existing_ids),  # Start with existing files
        "failed_files": [],
        "translations": {},
        "original_texts": {},
        "reference_texts": {}
    }
    
    # Load reference texts
    ref_en_file = BIGVIDEO_DATA_PATH / "text_data_test.en"
    ref_zh_file = BIGVIDEO_DATA_PATH / "text_data_test.zh"
    
    ref_en_texts = {}
    ref_zh_texts = {}
    
    if ref_en_file.exists():
        with open(ref_en_file, 'r', encoding='utf-8') as f:
            ref_en_lines = f.readlines()
    
    if ref_zh_file.exists():
        with open(ref_zh_file, 'r', encoding='utf-8') as f:
            ref_zh_lines = f.readlines()
    
    # Map reference texts to IDs
    for i, test_id in enumerate(all_test_ids):
        if i < len(ref_en_lines):
            ref_en_texts[test_id] = ref_en_lines[i].strip()
        if i < len(ref_zh_lines):
            ref_zh_texts[test_id] = ref_zh_lines[i].strip()
    
    # Load existing processed files first
    for test_id in existing_ids:
        srt_file = output_dir / f"{test_id}.srt"
        translated_srt_file = output_dir / f"{test_id}_translated.srt"
        
        if srt_file.exists() and translated_srt_file.exists():
            original_text = to_big_video_format(srt_file)
            translated_text = to_big_video_format(translated_srt_file)
            
            results["translations"][test_id] = translated_text
            results["original_texts"][test_id] = original_text
            results["reference_texts"][test_id] = ref_zh_texts.get(test_id, "")
    
    # Process missing files
    for test_id in missing_ids:
        try:
            print(f"\n📝 Processing missing file: {test_id}...")
            
            # Find video file
            video_file = video_dir / f"{test_id}.mp4"
            if not video_file.exists():
                print(f"❌ Video file not found: {video_file}")
                results["failed_files"].append(test_id)
                continue
            
            # Extract audio
            audio_file = audio_extractor(video_file, use_local_whisper=use_local_whisper)
            if not audio_file:
                print(f"❌ Audio extraction failed for {test_id}")
                results["failed_files"].append(test_id)
                continue
            
            # Handle case where audio_file might be a list (split files)
            if isinstance(audio_file, list):
                if not any(f.exists() for f in audio_file):
                    print(f"❌ Audio extraction failed for {test_id}")
                    results["failed_files"].append(test_id)
                    continue
            else:
                if not audio_file.exists():
                    print(f"❌ Audio extraction failed for {test_id}")
                    results["failed_files"].append(test_id)
                    continue
            
            # Transcribe audio
            transcription = whisper_transcription(audio_file, source_lang="en", use_local_whisper=use_local_whisper)
            if not transcription:
                print(f"❌ Transcription failed for {test_id}")
                results["failed_files"].append(test_id)
                continue
            
            # Save transcription as SRT file
            srt_file = output_dir / f"{test_id}.srt"
            save_srt_transcription(transcription, srt_file)
            
            # Translate SRT file
            translated_srt_path = translate_srt_file(srt_file, src_lang="en", tgt_lang="zh")
            if not translated_srt_path:
                print(f"❌ Translation failed for {test_id}")
                results["failed_files"].append(test_id)
                continue
            
            # Convert to BigVideo format
            original_text = to_big_video_format(srt_file)
            translated_text = to_big_video_format(translated_srt_path)
            
            results["translations"][test_id] = translated_text
            results["original_texts"][test_id] = original_text
            results["reference_texts"][test_id] = ref_zh_texts.get(test_id, "")
            results["processed_files"].append(test_id)
            
            print(f"✅ Successfully processed {test_id}")
            
        except Exception as e:
            print(f"❌ Error processing {test_id}: {e}")
            results["failed_files"].append(test_id)
    
    print("\n📊 Resume completed!")
    print(f"  Total files: {len(all_test_ids)}")
    print(f"  Previously processed: {len(existing_ids)}")
    print(f"  Newly processed: {len([f for f in results['processed_files'] if f not in existing_ids])}")
    print(f"  Failed: {len(results['failed_files'])}")
    
    return results

def resume_dovebench_dataset(dataset_dir, output_dir, use_local_whisper=False):
    """
    Resume DoveBench dataset processing: only process missing translation files.
    
    Args:
        dataset_dir (str or Path): Directory containing DoveBench data.
        output_dir (str or Path): Output directory for results.
        use_local_whisper (bool): If True, use local Whisper model instead of API.
    
    Returns:
        dict: Evaluation results.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔄 RESUMING DoveBench dataset processing...")
    
    # Find missing files
    all_video_ids, missing_ids, existing_ids = find_missing_files_dovebench(dataset_dir, output_dir)
    
    if not missing_ids:
        print("✅ All files already processed! Loading existing results for evaluation...")
        
        # Load existing translations and apply metrics
        results = {
            "processed_files": existing_ids,
            "failed_files": [],
            "translations": {},
            "original_texts": {},
            "reference_texts": {}
        }
        
        # Load existing results
        for video_id in existing_ids:
            domain, video_name = video_id.split('/')
            srt_file = output_dir / domain / f"{video_name}.srt"
            translated_srt_file = output_dir / domain / f"{video_name}_translated.srt"
            
            if srt_file.exists() and translated_srt_file.exists():
                original_text = to_big_video_format(srt_file)
                translated_text = to_big_video_format(translated_srt_file)
                
                results["translations"][video_id] = translated_text
                results["original_texts"][video_id] = original_text
                
                # Load reference translation if exists
                video_dir = dataset_dir / domain / video_name
                ref_files = list(video_dir.glob("*_ZH.ass"))
                ref_text = ""
                if ref_files:
                    try:
                        with open(ref_files[0], 'r', encoding='utf-8') as f:
                            ref_content = f.read()
                            # Simple ASS parsing - extract dialogue lines
                            dialogue_lines = []
                            for line in ref_content.split('\n'):
                                if line.startswith('Dialogue:'):
                                    parts = line.split(',', 9)
                                    if len(parts) >= 10:
                                        dialogue_lines.append(parts[9].strip())
                            ref_text = ' '.join(dialogue_lines)
                    except Exception as e:
                        print(f"Error reading reference file {ref_files[0]}: {e}")
                
                results["reference_texts"][video_id] = ref_text
        
        return results
    
    if use_local_whisper:
        print("🏠 Using local Whisper model for missing files")
    else:
        print("☁️ Using Whisper API for missing files")
    
    # Process missing files
    results = {
        "processed_files": list(existing_ids),  # Start with existing files
        "failed_files": [],
        "translations": {},
        "original_texts": {},
        "reference_texts": {}
    }
    
    # Load existing processed files first
    for video_id in existing_ids:
        domain, video_name = video_id.split('/')
        srt_file = output_dir / domain / f"{video_name}.srt"
        translated_srt_file = output_dir / domain / f"{video_name}_translated.srt"
        
        if srt_file.exists() and translated_srt_file.exists():
            original_text = to_big_video_format(srt_file)
            translated_text = to_big_video_format(translated_srt_file)
            
            results["translations"][video_id] = translated_text
            results["original_texts"][video_id] = original_text
            
            # Load reference translation if exists
            video_dir = dataset_dir / domain / video_name
            ref_files = list(video_dir.glob("*_ZH.ass"))
            ref_text = ""
            if ref_files:
                try:
                    with open(ref_files[0], 'r', encoding='utf-8') as f:
                        ref_content = f.read()
                        # Simple ASS parsing - extract dialogue lines
                        dialogue_lines = []
                        for line in ref_content.split('\n'):
                            if line.startswith('Dialogue:'):
                                parts = line.split(',', 9)
                                if len(parts) >= 10:
                                    dialogue_lines.append(parts[9].strip())
                        ref_text = ' '.join(dialogue_lines)
                except Exception as e:
                    print(f"Error reading reference file {ref_files[0]}: {e}")
            
            results["reference_texts"][video_id] = ref_text
    
    # Process missing files
    for video_id in missing_ids:
        try:
            domain, video_name = video_id.split('/')
            print(f"\n📝 Processing missing file: {video_id}...")
            
            # Create domain output directory
            domain_output = output_dir / domain
            domain_output.mkdir(parents=True, exist_ok=True)
            
            # Find video file
            video_dir = dataset_dir / domain / video_name
            video_files = list(video_dir.glob("*.mp4"))
            if not video_files:
                print(f"❌ No video file found in {video_dir}")
                results["failed_files"].append(video_id)
                continue
            
            video_file = video_files[0]
            
            # Extract audio
            audio_file = audio_extractor(video_file, use_local_whisper=use_local_whisper)
            if not audio_file:
                print(f"❌ Audio extraction failed for {video_id}")
                results["failed_files"].append(video_id)
                continue
            
            # Handle case where audio_file might be a list (split files)
            if isinstance(audio_file, list):
                if not any(f.exists() for f in audio_file):
                    print(f"❌ Audio extraction failed for {video_id}")
                    results["failed_files"].append(video_id)
                    continue
            else:
                if not audio_file.exists():
                    print(f"❌ Audio extraction failed for {video_id}")
                    results["failed_files"].append(video_id)
                    continue
            
            # Transcribe audio
            transcription = whisper_transcription(audio_file, source_lang="en", use_local_whisper=use_local_whisper)
            if not transcription:
                print(f"❌ Transcription failed for {video_id}")
                results["failed_files"].append(video_id)
                continue
            
            # Save transcription as SRT file
            srt_file = domain_output / f"{video_name}.srt"
            save_srt_transcription(transcription, srt_file)
            
            # Translate SRT file
            translated_srt_path = translate_srt_file(srt_file, src_lang="en", tgt_lang="zh")
            if not translated_srt_path:
                print(f"❌ Translation failed for {video_id}")
                results["failed_files"].append(video_id)
                continue
            
            # Copy translated SRT to the original data folder
            translated_srt_file = video_dir / f"{video_name}_translated.srt"
            shutil.copy2(translated_srt_path, translated_srt_file)
            
            # Load reference translation if exists
            ref_files = list(video_dir.glob("*_ZH.ass"))
            ref_text = ""
            if ref_files:
                try:
                    with open(ref_files[0], 'r', encoding='utf-8') as f:
                        ref_content = f.read()
                        # Simple ASS parsing - extract dialogue lines
                        dialogue_lines = []
                        for line in ref_content.split('\n'):
                            if line.startswith('Dialogue:'):
                                parts = line.split(',', 9)
                                if len(parts) >= 10:
                                    dialogue_lines.append(parts[9].strip())
                        ref_text = ' '.join(dialogue_lines)
                except Exception as e:
                    print(f"Error reading reference file {ref_files[0]}: {e}")
            
            # Convert to text format
            original_text = to_big_video_format(srt_file)  # Original transcription
            translated_text = to_big_video_format(translated_srt_path)  # Translated text
            
            results["translations"][video_id] = translated_text
            results["original_texts"][video_id] = original_text
            results["reference_texts"][video_id] = ref_text
            results["processed_files"].append(video_id)
            
            print(f"✅ Successfully processed {video_id}")
            
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            results["failed_files"].append(video_id)
    
    print("\n📊 Resume completed!")
    print(f"  Total files: {len(all_video_ids)}")
    print(f"  Previously processed: {len(existing_ids)}")
    print(f"  Newly processed: {len([f for f in results['processed_files'] if f not in existing_ids])}")
    print(f"  Failed: {len(results['failed_files'])}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DocMTAgent Evaluation Script")
    parser.add_argument("--mode", choices=["full", "test", "test-scoring", "bigvideo", "dovebench", "resume", "resume-bigvideo", "resume-dovebench"], 
                        default="full", help="Evaluation mode")
    parser.add_argument("--test-video", type=str, help="Path to a single video file for testing")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--reference-text", type=str, help="Reference translation for scoring test")
    parser.add_argument("--source-text", type=str, help="Source text for COMET scoring test")
    parser.add_argument("--use-local-whisper", action="store_true", 
                        help="Use local Whisper model (StableWhisperASR) instead of OpenAI API. "
                             "Avoids 25MB file size limit but requires local model installation.")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        main(use_local_whisper=args.use_local_whisper)
    elif args.mode == "bigvideo":
        print("Processing BigVideo dataset only...")
        output_dir = Path(args.output_dir)
        bigvideo_output = output_dir / "bigvideo"
        
        bigvideo_results = process_bigvideo_dataset(
            test_ids_file=BIGVIDEO_DATA_PATH / "text_data_test.id",
            video_dir=BIGVIDEO_DATA_PATH / "test",
            output_dir=bigvideo_output,
            use_local_whisper=args.use_local_whisper
        )
        
        bigvideo_scores = calculate_bigvideo_scores(bigvideo_results, bigvideo_output)
        save_results(bigvideo_results, bigvideo_scores, bigvideo_output / "evaluation_results.json")
        
        print("\nBigVideo Results:")
        print(f"  Processed: {len(bigvideo_results['processed_files'])} files")
        print(f"  Failed: {len(bigvideo_results['failed_files'])} files")
        if bigvideo_scores:
            for metric, score in bigvideo_scores.items():
                print(f"  {metric}: {score:.4f}")
    
    elif args.mode == "dovebench":
        print("Processing DoveBench dataset only...")
        output_dir = Path(args.output_dir)
        dovebench_output = output_dir / "dovebench"
        
        dovebench_results = process_dovebench_dataset(
            dataset_dir=DOVEBENCH_DATA_PATH,
            output_dir=dovebench_output,
            use_local_whisper=args.use_local_whisper
        )
        
        dovebench_scores = calculate_dovebench_scores(dovebench_results, dovebench_output)
        save_results(dovebench_results, dovebench_scores, dovebench_output / "evaluation_results.json")
        
        print("\nDoveBench Results:")
        print(f"  Processed: {len(dovebench_results['processed_files'])} files")
        print(f"  Failed: {len(dovebench_results['failed_files'])} files")
        if dovebench_scores:
            for metric, score in dovebench_scores.items():
                print(f"  {metric}: {score:.4f}")
    
    elif args.mode == "resume":
        print("Resuming evaluation for both datasets...")
        output_dir = Path(args.output_dir)
        bigvideo_output = output_dir / "bigvideo"
        dovebench_output = output_dir / "dovebench"
        
        # Resume BigVideo dataset
        print("\n" + "="*50)
        print("Resuming BigVideo dataset...")
        print("="*50)
        
        bigvideo_results = resume_bigvideo_dataset(
            test_ids_file=BIGVIDEO_DATA_PATH / "text_data_test.id",
            video_dir=BIGVIDEO_DATA_PATH / "test",
            output_dir=bigvideo_output,
            use_local_whisper=args.use_local_whisper
        )
        
        bigvideo_scores = calculate_bigvideo_scores(bigvideo_results, bigvideo_output)
        save_results(bigvideo_results, bigvideo_scores, bigvideo_output / "evaluation_results.json")
        
        # Resume DoveBench dataset
        print("\n" + "="*50)
        print("Resuming DoveBench dataset...")
        print("="*50)
        
        dovebench_results = resume_dovebench_dataset(
            dataset_dir=DOVEBENCH_DATA_PATH,
            output_dir=dovebench_output,
            use_local_whisper=args.use_local_whisper
        )
        
        dovebench_scores = calculate_dovebench_scores(dovebench_results, dovebench_output)
        save_results(dovebench_results, dovebench_scores, dovebench_output / "evaluation_results.json")
        
        # Print summary
        print("\n" + "="*50)
        print("RESUME EVALUATION SUMMARY")
        print("="*50)
        
        print("\nBigVideo Dataset:")
        print(f"  Processed: {len(bigvideo_results['processed_files'])} files")
        print(f"  Failed: {len(bigvideo_results['failed_files'])} files")
        if bigvideo_scores:
            for metric, score in bigvideo_scores.items():
                print(f"  {metric}: {score:.4f}")
        
        print("\nDoveBench Dataset:")
        print(f"  Processed: {len(dovebench_results['processed_files'])} files")
        print(f"  Failed: {len(dovebench_results['failed_files'])} files")
        if dovebench_scores:
            for metric, score in dovebench_scores.items():
                print(f"  {metric}: {score:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
    
    elif args.mode == "resume-bigvideo":
        print("Resuming BigVideo dataset only...")
        output_dir = Path(args.output_dir)
        bigvideo_output = output_dir / "bigvideo"
        
        bigvideo_results = resume_bigvideo_dataset(
            test_ids_file=BIGVIDEO_DATA_PATH / "text_data_test.id",
            video_dir=BIGVIDEO_DATA_PATH / "test",
            output_dir=bigvideo_output,
            use_local_whisper=args.use_local_whisper
        )
        
        bigvideo_scores = calculate_bigvideo_scores(bigvideo_results, bigvideo_output)
        save_results(bigvideo_results, bigvideo_scores, bigvideo_output / "evaluation_results.json")
        
        print("\nBigVideo Resume Results:")
        print(f"  Processed: {len(bigvideo_results['processed_files'])} files")
        print(f"  Failed: {len(bigvideo_results['failed_files'])} files")
        if bigvideo_scores:
            for metric, score in bigvideo_scores.items():
                print(f"  {metric}: {score:.4f}")
    
    elif args.mode == "resume-dovebench":
        print("Resuming DoveBench dataset only...")
        output_dir = Path(args.output_dir)
        dovebench_output = output_dir / "dovebench"
        
        dovebench_results = resume_dovebench_dataset(
            dataset_dir=DOVEBENCH_DATA_PATH,
            output_dir=dovebench_output,
            use_local_whisper=args.use_local_whisper
        )
        
        dovebench_scores = calculate_dovebench_scores(dovebench_results, dovebench_output)
        save_results(dovebench_results, dovebench_scores, dovebench_output / "evaluation_results.json")
        
        print("\nDoveBench Resume Results:")
        print(f"  Processed: {len(dovebench_results['processed_files'])} files")
        print(f"  Failed: {len(dovebench_results['failed_files'])} files")
        if dovebench_scores:
            for metric, score in dovebench_scores.items():
                print(f"  {metric}: {score:.4f}")
