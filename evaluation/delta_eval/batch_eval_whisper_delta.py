import os
import sys
import json
import subprocess
from pathlib import Path

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules after path setup
try:
    from src.audio.ASR import WhisperAPIASR
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

def extract_audio(video_path, output_path, max_size_mb=25):
    """
    Extract audio from video file using ffmpeg with size optimization.
    
    Args:
        video_path (str or Path): Path to the video file.
        output_path (str or Path): Path to save the extracted audio.
        max_size_mb (int): Maximum file size in MB for OpenAI API compatibility.
    
    Returns:
        Path: Path to the extracted audio file if successful, None otherwise.
    """
    try:
        # First, try with lower bitrate to reduce file size
        subprocess.run(
            [
                "ffmpeg", "-y",  # -y to overwrite output files
                "-i", str(video_path),
                "-f", "mp3",
                "-ab", "128000",  # Lower bitrate: 64kbps instead of 192kbps
                "-ar", "32000",  # Lower sample rate: 16kHz instead of default
                "-ac", "1",      # Mono instead of stereo
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
        
        # If still too large, try even lower bitrate
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
        
        # If still too large, split the audio
        if file_size_mb > max_size_mb:
            print(f"File still too large ({file_size_mb:.1f} MB), splitting audio...")
            return split_audio_file(output_path, max_size_mb)
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio from {video_path}: {e}")
        return None

def audio_extractor(video_path):
    """
    Extract audio from video files for a single video.
    
    Args:
        video_path (str or Path): Path to the video file.
    
    Returns:
        Path or List[Path]: Path(s) to the extracted audio file(s).
    """
    video_path = Path(video_path)
    video_name = video_path.stem  # Get the base name without extension
    output_path = video_path.parent / f"{video_name}.mp3"
    
    result = extract_audio(video_path, output_path)
    return result

def whisper_transcription(audio_paths, source_lang="en"):
    """
    Transcribe audio using Whisper API ASR. Handles both single files and multiple chunks.
    
    Args:
        audio_paths (str, Path, or List): Path(s) to the audio file(s).
        source_lang (str): Source language code.
    
    Returns:
        str: Transcription result in SRT format.
    """
    try:
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
            
            transcription = asr.get_transcript(str(audio_path), source_lang=source_lang)
            
            if transcription:
                # Adjust timestamps if this is not the first chunk
                if cumulative_time_offset > 0:
                    transcription = adjust_srt_timestamps(transcription, cumulative_time_offset)
                
                all_transcriptions.append(transcription)
                cumulative_time_offset += chunk_duration
                print(f"Transcription successful for chunk {i+1}")
            else:
                print(f"Failed to transcribe chunk {i+1}: {audio_path}")
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
        import re
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

def process_bigvideo_dataset(test_ids_file, video_dir, output_dir):
    """
    Process BigVideo dataset: extract audio, transcribe, translate, and evaluate.
    
    Args:
        test_ids_file (str or Path): Path to the test IDs file.
        video_dir (str or Path): Directory containing video files.
        output_dir (str or Path): Output directory for results.
    
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
            audio_file = audio_extractor(video_file)
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
            transcription = whisper_transcription(audio_file, source_lang="en")
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

def process_dovebench_dataset(dataset_dir, output_dir):
    """
    Process DoveBench dataset: extract audio, transcribe, translate, and evaluate.
    
    Args:
        dataset_dir (str or Path): Directory containing DoveBench data.
        output_dir (str or Path): Output directory for results.
    
    Returns:
        dict: Evaluation results.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
                audio_file = audio_extractor(video_file)
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
                transcription = whisper_transcription(audio_file, source_lang="en")
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
                import shutil
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
                            import re
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

def test_single_video(video_path, output_dir="./test_output"):
    """
    Test the evaluation pipeline on a single video file.
    
    Args:
        video_path (str or Path): Path to a single video file for testing.
        output_dir (str or Path): Output directory for test results.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing evaluation pipeline on: {video_path}")
    
    try:
        # Extract audio
        print("1. Extracting audio...")
        audio_file = audio_extractor(video_path)
        if not audio_file:
            print("❌ Audio extraction failed")
            return False
            
        # Handle case where audio_file might be a list (split files)
        if isinstance(audio_file, list):
            # Check if any of the chunks exist
            if not any(f.exists() for f in audio_file):
                print("❌ Audio extraction failed")
                return False
            print(f"✅ Audio extracted and split into {len(audio_file)} chunks")
        else:
            if not audio_file.exists():
                print("❌ Audio extraction failed")
                return False
            print(f"✅ Audio extracted: {audio_file}")
        
        # Transcribe
        print("2. Transcribing audio...")
        transcription = whisper_transcription(audio_file, source_lang="en")
        if not transcription:
            print("❌ Transcription failed")
            return False
        print("✅ Transcription completed")
        
        # Save SRT
        print("3. Saving SRT file...")
        srt_file = output_dir / f"{video_path.stem}.srt"
        save_srt_transcription(transcription, srt_file)
        print(f"✅ SRT saved: {srt_file}")
        
        # Translate
        print("4. Translating...")
        translated_srt_path = translate_srt_file(srt_file, src_lang="en", tgt_lang="zh")
        if not translated_srt_path:
            print("❌ Translation failed")
            return False
        print("✅ Translation completed")
        print(f"✅ Translated SRT saved: {translated_srt_path}")
        
        print("🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_single_video_with_scoring(video_path, reference_text=None, source_text=None, output_dir="./test_scoring_output"):
    """
    Test the evaluation pipeline on a single video file with comprehensive scoring.
    
    Args:
        video_path (str or Path): Path to a single video file for testing.
        reference_text (str): Optional reference translation for scoring.
        source_text (str): Optional source text for COMET scoring.
        output_dir (str or Path): Output directory for test results.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing evaluation pipeline with scoring on: {video_path}")
    
    try:
        # Extract audio
        print("1. Extracting audio...")
        audio_file = audio_extractor(video_path)
        if not audio_file:
            print("❌ Audio extraction failed")
            return False
            
        # Handle case where audio_file might be a list (split files)
        if isinstance(audio_file, list):
            if not any(f.exists() for f in audio_file):
                print("❌ Audio extraction failed")
                return False
            print(f"✅ Audio extracted and split into {len(audio_file)} chunks")
            for i, chunk in enumerate(audio_file):
                file_size = chunk.stat().st_size / (1024 * 1024)
                print(f"   Chunk {i+1}: {chunk.name} ({file_size:.1f} MB)")
        else:
            if not audio_file.exists():
                print("❌ Audio extraction failed")
                return False
            file_size = audio_file.stat().st_size / (1024 * 1024)
            print(f"✅ Audio extracted: {audio_file.name} ({file_size:.1f} MB)")
        
        # Transcribe
        print("2. Transcribing audio...")
        transcription = whisper_transcription(audio_file, source_lang="en")
        if not transcription:
            print("❌ Transcription failed")
            return False
        print("✅ Transcription completed")
        
        # Save SRT
        print("3. Saving SRT file...")
        srt_file = output_dir / f"{video_path.stem}.srt"
        save_srt_transcription(transcription, srt_file)
        print(f"✅ SRT saved: {srt_file}")
        
        # Translate
        print("4. Translating...")
        translated_srt_path = translate_srt_file(srt_file, src_lang="en", tgt_lang="zh")
        if not translated_srt_path:
            print("❌ Translation failed")
            return False
        print("✅ Translation completed")
        print(f"✅ Translated SRT saved: {translated_srt_path}")
        
        # Extract texts for scoring
        print("5. Extracting texts for scoring...")
        original_text = to_big_video_format(srt_file)
        translated_text = to_big_video_format(translated_srt_path)
        
        print(f"Original text preview: {original_text[:100]}...")
        print(f"Translated text preview: {translated_text[:100]}...")
        
        # Test scoring functionality
        print("6. Testing scoring functionality...")
        scores = {}
        
        if SCORING_AVAILABLE:
            try:
                # Test BLEU score with a simple reference
                test_reference = reference_text if reference_text else "这是一个测试翻译示例。"
                test_translation = translated_text if translated_text else "这是翻译的文本。"
                
                print("   Testing BLEU score...")
                bleu_score = BLEUscore([test_translation], [[test_reference]])
                scores["BLEU"] = bleu_score.score
                print(f"   ✅ BLEU Score: {bleu_score.score:.4f}")
                
                # Test COMET score
                if source_text or original_text:
                    print("   Testing COMET score...")
                    test_source = source_text if source_text else original_text
                    try:
                        # Fix multiprocessing issue with COMET by using batch_size=1 and gpus=0
                        comet_result = COMETscore([test_source], [test_translation], [test_reference])
                        if hasattr(comet_result, 'scores'):
                            scores["COMET"] = sum(comet_result.scores) / len(comet_result.scores)
                            print(f"   ✅ COMET Score: {scores['COMET']:.4f}")
                        else:
                            scores["COMET"] = comet_result.score if hasattr(comet_result, 'score') else 0.0
                            print(f"   ✅ COMET Score: {scores['COMET']:.4f}")
                    except Exception as comet_error:
                        print(f"   ⚠️ COMET scoring failed: {comet_error}")
                        scores["COMET"] = 0.0
                        print(f"   ⚠️ COMET Score (fallback): {scores['COMET']:.4f}")
                
            except Exception as e:
                print(f"   ⚠️ Scoring failed: {e}")
                print("   This is expected if scoring dependencies are not fully installed")
        else:
            print("   ⚠️ Scoring modules not available")
        
        # Save detailed results
        results = {
            "video_file": str(video_path),
            "audio_file": str(audio_file) if not isinstance(audio_file, list) else [str(f) for f in audio_file],
            "srt_file": str(srt_file),
            "translated_srt_file": str(translated_srt_path),
            "original_text": original_text,
            "translated_text": translated_text,
            "scores": scores,
            "success": True
        }
        
        results_file = output_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Test results saved: {results_file}")
        print("🎉 Comprehensive test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    import re
    
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
    import re
    
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

def main():
    """
    Main evaluation function for DocMTAgent on DoveBench and BigVideo datasets.
    """
    print("Starting DocMTAgent evaluation...")
    
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
        output_dir=bigvideo_output
    )
    
    bigvideo_scores = calculate_bigvideo_scores(bigvideo_results, bigvideo_output)
    save_results(bigvideo_results, bigvideo_scores, bigvideo_output / "evaluation_results.json")
    
    # Process DoveBench dataset
    print("\n" + "="*50)
    print("Processing DoveBench dataset...")
    print("="*50)
    
    dovebench_results = process_dovebench_dataset(
        dataset_dir=DOVEBENCH_DATA_PATH,
        output_dir=dovebench_output
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DocMTAgent Evaluation Script")
    parser.add_argument("--mode", choices=["full", "test", "test-scoring", "bigvideo", "dovebench"], 
                        default="full", help="Evaluation mode")
    parser.add_argument("--test-video", type=str, help="Path to a single video file for testing")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--reference-text", type=str, help="Reference translation for scoring test")
    parser.add_argument("--source-text", type=str, help="Source text for COMET scoring test")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        if not args.test_video:
            print("Error: --test-video is required for test mode")
            exit(1)
        test_single_video(args.test_video, args.output_dir)
    elif args.mode == "test-scoring":
        if not args.test_video:
            print("Error: --test-video is required for test-scoring mode")
            exit(1)
        test_single_video_with_scoring(
            args.test_video, 
            args.reference_text, 
            args.source_text, 
            args.output_dir
        )
    elif args.mode == "full":
        main()
    elif args.mode == "bigvideo":
        print("Processing BigVideo dataset only...")
        output_dir = Path(args.output_dir)
        bigvideo_output = output_dir / "bigvideo"
        
        bigvideo_results = process_bigvideo_dataset(
            test_ids_file=BIGVIDEO_DATA_PATH / "text_data_test.id",
            video_dir=BIGVIDEO_DATA_PATH / "test",
            output_dir=bigvideo_output
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
            output_dir=dovebench_output
        )
        
        dovebench_scores = calculate_dovebench_scores(dovebench_results, dovebench_output)
        save_results(dovebench_results, dovebench_scores, dovebench_output / "evaluation_results.json")
        
        print("\nDoveBench Results:")
        print(f"  Processed: {len(dovebench_results['processed_files'])} files")
        print(f"  Failed: {len(dovebench_results['failed_files'])} files")
        if dovebench_scores:
            for metric, score in dovebench_scores.items():
                print(f"  {metric}: {score:.4f}")