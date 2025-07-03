import sys
import os
import glob
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import from local module (same directory)
# Assume all imports are successful as requested
from scores.score import SubERscore

# Set availability flags to True as requested
SUBER_AVAILABLE = True
SUBSONAR_AVAILABLE = True
ASS_EXTRACTOR_AVAILABLE = True


def extract_text_from_eval_file(eval_file_path: str) -> str:
    """Extract translation text from evaluation text file.
    
    The evaluation files contain AI model outputs in a specific format.
    We need to extract only the assistant responses that contain translations.
    
    Args:
        eval_file_path (str): Path to evaluation text file
        
    Returns:
        str: Extracted translation text, one sentence per line
    """
    translations = []
    
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "assistant" markers to get AI responses
    assistant_responses = content.split('assistant\n')[1:]  # Skip first empty part
    
    for response in assistant_responses:
        # Get text before next "system" or "user" marker
        if 'system\n' in response:
            text = response.split('system\n')[0].strip()
        elif 'user\n' in response:
            text = response.split('user\n')[0].strip()
        else:
            text = response.strip()
        
        if text and len(text) > 5:  # Filter out very short responses
            # Clean up the text
            text = text.replace('\n', ' ').strip()
            if text:
                translations.append(text)
    
    return '\n'.join(translations)


def create_temp_srt_from_text(text_content: str) -> str:
    """Create temporary SRT file from text content.
    
    Args:
        text_content (str): Text content, one sentence per line
        
    Returns:
        str: Path to temporary SRT file
    """
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    
    if not lines:
        raise ValueError("No content to create SRT file")
    
    # Create temporary SRT file
    temp_fd, temp_srt_path = tempfile.mkstemp(suffix='.srt', text=True)
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines, 1):
                # Create simple SRT format with dummy timestamps
                start_time = f"00:00:{i:02d},000"
                end_time = f"00:00:{i+1:02d},000"
                f.write(f"{i}\n{start_time} --> {end_time}\n{line}\n\n")
    except:
        # Clean up on error
        if os.path.exists(temp_srt_path):
            os.unlink(temp_srt_path)
        raise
    
    return temp_srt_path


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


def batch_eval_suber(eval_folder: str, ref_folder: str) -> Dict[str, float]:
    """Batch evaluate SubER scores for evaluation files against reference SRT files.
    
    Args:
        eval_folder (str): Path to folder containing evaluation text files
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
    
    # Get all text files from eval folder
    eval_files = list(eval_folder_path.glob("*.txt"))
    
    if not eval_files:
        print(f"No .txt files found in {eval_folder}")
        return {}
    
    results = {}
    temp_files_to_cleanup = []
    
    try:
        for eval_file in eval_files:
            eval_filename = eval_file.stem  # filename without extension
            print(f"Processing: {eval_filename}")
            
            # Find matching reference SRT file
            ref_srt_path = find_matching_ref_file(eval_filename, str(ref_folder_path))
            
            if not ref_srt_path:
                print(f"  Warning: No matching reference SRT file found for {eval_filename}")
                continue
            
            try:
                # Extract text from evaluation file
                eval_text = extract_text_from_eval_file(str(eval_file))
                
                if not eval_text.strip():
                    print(f"  Warning: No text extracted from {eval_filename}")
                    continue
                
                # Create SRT from evaluation text
                eval_srt_path = create_temp_srt_from_text(eval_text)
                temp_files_to_cleanup.append(eval_srt_path)
                
                # Calculate SubER score using existing SRT files directly
                score = SubERscore(eval_srt_path, ref_srt_path)
                results[eval_filename] = score
                
                print(f"  SubER score: {score:.4f}")
                print(f"  Reference file: {Path(ref_srt_path).name}")
                
            except Exception as e:
                print(f"  Error processing {eval_filename}: {e}")
                continue
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_file}: {e}")
    
    return results


def batch_eval_subsonar(eval_folder: str, ref_folder: str) -> Dict[str, float]:
    """Batch evaluate SubSONAR scores for evaluation files against reference files.
    
    Note: SubSONAR requires audio files, which may not be available.
    This function is a placeholder and may need audio file handling.
    
    Args:
        eval_folder (str): Path to folder containing evaluation text files
        ref_folder (str): Path to folder containing reference files and audio
        
    Returns:
        Dict[str, float]: Dictionary mapping filenames to SubSONAR scores
    """
    print("SubSONAR evaluation requires audio files.")
    print("This function is not yet implemented as audio files may not be available.")
    print("Please implement SubSONAR evaluation based on your specific audio file setup.")
    
    # TODO: Implement SubSONAR evaluation when audio files are available
    # You would need to:
    # 1. Find matching audio files (mp3/mp4) for each evaluation file
    # 2. Convert evaluation text to SRT format
    # 3. Call SubSONARscore with the SRT file, audio file, and language codes
    
    return {}


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
        eval_folder (str): Path to folder containing evaluation files
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
        results = batch_eval(eval_folder=eval_folder_path, ref_folder=ref_folder_path, eval_type="suber")
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(results["summary"])
        print(f"\nStatistics:")
        print(f"  Files processed: {results['statistics']['count']}")
        print(f"  Average score: {results['statistics']['avg']:.4f}")
        print(f"  Min score: {results['statistics']['min']:.4f}")
        print(f"  Max score: {results['statistics']['max']:.4f}")
        
        print(f"\nIndividual scores:")
        for filename, score in results["individual_scores"].items():
            print(f"  {filename}: {score:.4f}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()