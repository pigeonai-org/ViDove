import sys
import os
import glob
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import from local module (same directory)
# Assume all imports are successful as requested
# from scores.score import SubSONARscore

# from scores.SubER_main.suber.file_readers import read_input_file
# from scores.SubER_main.suber.metrics.suber import calculate_SubER

# try:
# # from scores.subsonar.src.subsonar.sonar_metric import SonarAudioTextMetric
#     from subsonar import SonarAudioTextMetric
#     # from scores.subsonar.src.subsonar.srt_reader import SrtReader
#     from subsonar import SrtReader
#     print("t1")
# except:
#     from subsonar.sonar_metric import SonarAudioTextMetric
#     from subsonar.srt_reader import SrtReader
#     print("t2")
    

# Set availability flags to True as requested
SUBER_AVAILABLE = True
SUBSONAR_AVAILABLE = False
ASS_EXTRACTOR_AVAILABLE = False

# Language codes for SubSONAR
# Audio language codes: "eng" (English), "cmn" (Mandarin Chinese), etc.
# Text language codes in Flores 200 format: "eng_Latn" (English), "zho_Hans" (Simplified Chinese), etc.

def SubERscore(hypothesis_file: str, reference_file: str) -> float:
    """
    计算SubER分数
    
    Args:
        hypothesis_file: 假设SRT文件路径
        reference_file: 参考SRT文件路径
        
    Returns:
        float: SubER分数
    """
    from scores.SubER_main.suber.file_readers import read_input_file
    from scores.SubER_main.suber.metrics.suber import calculate_SubER
    
    hypo_segments = read_input_file(hypothesis_file, file_format="SRT")
    ref_segments = read_input_file(reference_file, file_format="SRT")

    score = calculate_SubER(hypo_segments, ref_segments)

    return score


def find_matching_ref_file(eval_filename: str, ref_folder: str) -> Optional[str]:
    """Find matching reference SRT file for evaluation file.
    
    Args:
        eval_filename (str): Name of evaluation file (without extension)
        ref_folder (str): Path to reference folder containing SRT files
        
    Returns:
        Optional[str]: Path to matching SRT file, or None if not found
    """
    ref_folder_path = Path(ref_folder)
    
    # Try different matching strategies for SRT files
    patterns_to_try = [
        f"{eval_filename}*.srt",
        f"*{eval_filename}*.srt",
        f"*{eval_filename.replace('_', ' ')}*.srt",
        f"*{eval_filename.split('_')[0]}*.srt" if '_' in eval_filename else f"{eval_filename}*.srt"
    ]
    
    for pattern in patterns_to_try:
        matches = list(ref_folder_path.glob(pattern))
        if matches:
            return str(matches[0])  # Return first match
    
    return None


def find_matching_audio_file(srt_file_path: str) -> Optional[str]:
    """Find matching audio file for a given SRT file.
    
    Looks for audio files with the same base name as the SRT file.
    
    Args:
        srt_file_path (str): Path to SRT file
        
    Returns:
        Optional[str]: Path to matching audio file, or None if not found
    """
    srt_path = Path(srt_file_path)
    srt_folder = srt_path.parent
    srt_stem = srt_path.stem  # filename without extension
    
    # Common audio file extensions
    audio_extensions = ['.mp3', '.mp4', '.wav', '.m4a', '.flac', '.aac']
    
    # Try to find audio file with same base name
    for ext in audio_extensions:
        audio_path = srt_folder / f"{srt_stem}{ext}"
        if audio_path.exists():
            return str(audio_path)
    
    # If exact match not found, try fuzzy matching
    for ext in audio_extensions:
        pattern = f"*{srt_stem}*{ext}"
        matches = list(srt_folder.glob(pattern))
        if matches:
            return str(matches[0])  # Return first match
    
    return None


def batch_eval_suber(eval_folder: str, ref_folder: str) -> Dict[str, float]:
    """Batch evaluate SubER scores for SRT files against reference SRT files.
    
    Args:
        eval_folder (str): Path to folder containing evaluation SRT files
        ref_folder (str): Path to folder containing reference SRT files
        
    Returns:
        Dict[str, float]: Dictionary mapping filenames to SubER scores
    """
    eval_folder_path = Path(eval_folder)
    ref_folder_path = Path(ref_folder)
    
    if not eval_folder_path.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {eval_folder}")
    if not ref_folder_path.exists():
        raise FileNotFoundError(f"Reference folder not found: {ref_folder}")
    
    # Get all SRT files from eval folder
    eval_files = list(eval_folder_path.glob("*.srt"))
    
    if not eval_files:
        print(f"No .srt files found in {eval_folder}")
        return {}
    
    results = {}
    
    for eval_file in eval_files:
        eval_filename = eval_file.stem  # filename without extension
        print(f"Processing SubER for: {eval_filename}")
        
        # Find matching reference SRT file
        ref_srt_path = find_matching_ref_file(eval_filename, str(ref_folder_path))
        
        if not ref_srt_path:
            print(f"  Warning: No matching reference SRT file found for {eval_filename}")
            continue
        
        try:
            # Calculate SubER score using SRT files directly
            score = SubERscore(str(eval_file), ref_srt_path)
            results[eval_filename] = score
            
            print(f"  SubER score: {score:.4f}")
            print(f"  Reference file: {Path(ref_srt_path).name}")
            
        except Exception as e:
            print(f"  Error processing {eval_filename}: {e}")
            continue
    
    return results


def batch_eval_subsonar(eval_folder: str, ref_folder: str, audio_lang: str = "eng", text_lang: str = "zho_Hans") -> Dict[str, float]:
    """Batch evaluate SubSONAR scores for SRT files against reference files with audio.
    
    SubSONAR requires audio files to be present in the same directory as SRT files.
    
    Args:
        eval_folder (str): Path to folder containing evaluation SRT files
        ref_folder (str): Path to folder containing reference SRT files and audio files
        audio_lang (str): Language of the speech in audio files (default: "eng" for English)
        text_lang (str): Language of the text in Flores 200 format (default: "zho_Hans" for Chinese)
        
    Returns:
        Dict[str, float]: Dictionary mapping filenames to SubSONAR scores
    """
    eval_folder_path = Path(eval_folder)
    ref_folder_path = Path(ref_folder)
    
    if not eval_folder_path.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {eval_folder}")
    if not ref_folder_path.exists():
        raise FileNotFoundError(f"Reference folder not found: {ref_folder}")
    
    # Get all SRT files from eval folder
    eval_files = list(eval_folder_path.glob("*.srt"))
    
    if not eval_files:
        print(f"No .srt files found in {eval_folder}")
        return {}
    
    results = {}
    
    for eval_file in eval_files:
        eval_filename = eval_file.stem  # filename without extension
        print(f"Processing SubSONAR for: {eval_filename}")
        
        # Find matching reference SRT file
        ref_srt_path = find_matching_ref_file(eval_filename, str(ref_folder_path))
        
        if not ref_srt_path:
            print(f"  Warning: No matching reference SRT file found for {eval_filename}")
            continue
        
        # Find matching audio file
        audio_file_path = find_matching_audio_file(ref_srt_path)
        
        if not audio_file_path:
            print(f"  Warning: No matching audio file found for {Path(ref_srt_path).name}")
            continue
        
        try:
            # Calculate SubSONAR score using SRT files directly
            print(f"  Using audio file: {Path(audio_file_path).name}")
            print(f"  Audio language: {audio_lang}, Text language: {text_lang}")
            
            score = SubSONARscore(
                hypothesis_file=str(eval_file),
                audio_file=audio_file_path,
                audio_lang=audio_lang,
                text_lang=text_lang
            )
            
            results[eval_filename] = score
            
            print(f"  SubSONAR score: {score:.4f}")
            print(f"  Reference file: {Path(ref_srt_path).name}")
            
        except Exception as e:
            print(f"  Error processing {eval_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def calculate_average_scores(scores: Dict[str, float]) -> Tuple[float, Dict[str, any]]:
    """Calculate average and statistics for evaluation scores.
    
    Args:
        scores (Dict[str, float]): Dictionary of filename to score mappings
        
    Returns:
        Tuple[float, Dict[str, any]]: Average score and statistics dictionary
    """
    if not scores:
        return 0.0, {"count": 0, "min": 0, "max": 0, "avg": 0}
    
    score_values = list(scores.values())
    
    stats = {
        "count": len(score_values),
        "min": min(score_values),
        "max": max(score_values),
        "avg": sum(score_values) / len(score_values)
    }
    
    return stats["avg"], stats


def batch_eval(eval_folder: str, ref_folder: str, eval_type: str = "suber") -> Dict[str, any]:
    """Main batch evaluation function.
    
    Args:
        eval_folder (str): Path to folder containing evaluation SRT files
        ref_folder (str): Path to folder containing reference files
        eval_type (str): Type of evaluation ("suber" or "subsonar")
        
    Returns:
        Dict[str, any]: Evaluation results with scores and statistics
    """
    print(f"Starting batch evaluation with {eval_type.upper()}")
    print(f"Eval folder: {eval_folder}")
    print(f"Ref folder: {ref_folder}")
    
    if eval_type.lower() == "suber":
        scores = batch_eval_suber(eval_folder, ref_folder)
    elif eval_type.lower() == "subsonar":
        scores = batch_eval_subsonar(eval_folder, ref_folder)
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")
    
    # Calculate statistics
    avg_score, stats = calculate_average_scores(scores)
    
    results = {
        "eval_type": eval_type,
        "individual_scores": scores,
        "statistics": stats,
        "summary": f"Evaluated {stats['count']} files with average {eval_type.upper()} score: {avg_score:.4f}"
    }
    
    return results


if __name__ == "__main__":
    # Adjust paths based on where script is run from
    if os.path.basename(os.getcwd()) == "scores":
        # Running from evaluation/scores directory
        eval_folder_path = "../../test_data/dovebench"
        ref_folder_path = "../../test_data/dovebench/ref"
    else:
        # Running from project root directory
        eval_folder_path = "evaluation/test_data/dovebench"
        ref_folder_path = "evaluation/test_data/dovebench/ref"
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Evaluation folder path: {eval_folder_path}")
    print(f"Reference folder path: {ref_folder_path}")
    print(f"Evaluation folder exists: {os.path.exists(eval_folder_path)}")
    print(f"Reference folder exists: {os.path.exists(ref_folder_path)}")
    
    try:
        # Run SubER evaluation
        print("="*60)
        print("RUNNING SUBER EVALUATION")
        print("="*60)
        suber_results = batch_eval(eval_folder=eval_folder_path, ref_folder=ref_folder_path, eval_type="suber")
        
        print("\n" + "="*50)
        print("SUBER EVALUATION RESULTS")
        print("="*50)
        print(suber_results["summary"])
        print(f"\nStatistics:")
        print(f"  Files processed: {suber_results['statistics']['count']}")
        print(f"  Average score: {suber_results['statistics']['avg']:.4f}")
        print(f"  Min score: {suber_results['statistics']['min']:.4f}")
        print(f"  Max score: {suber_results['statistics']['max']:.4f}")
        
        print(f"\nIndividual scores:")
        for filename, score in suber_results["individual_scores"].items():
            print(f"  {filename}: {score:.4f}")
        
        # Optionally run SubSONAR evaluation (comment out if SubSONAR is not available)
        print("\n" + "="*60)
        print("RUNNING SUBSONAR EVALUATION")
        print("="*60)
        
        try:
            subsonar_results = batch_eval(eval_folder=eval_folder_path, ref_folder=ref_folder_path, eval_type="subsonar")
            
            print("\n" + "="*50)
            print("SUBSONAR EVALUATION RESULTS")
            print("="*50)
            print(subsonar_results["summary"])
            print(f"\nStatistics:")
            print(f"  Files processed: {subsonar_results['statistics']['count']}")
            print(f"  Average score: {subsonar_results['statistics']['avg']:.4f}")
            print(f"  Min score: {subsonar_results['statistics']['min']:.4f}")
            print(f"  Max score: {subsonar_results['statistics']['max']:.4f}")
            
            print(f"\nIndividual scores:")
            for filename, score in subsonar_results["individual_scores"].items():
                print(f"  {filename}: {score:.4f}")
                
        except Exception as subsonar_error:
            print(f"SubSONAR evaluation failed: {subsonar_error}")
            print("This is expected if SubSONAR dependencies are not installed.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()