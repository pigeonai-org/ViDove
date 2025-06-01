import os
import sys
import argparse
from pathlib import Path

# Add the root directory to the path to import to_big_video_format
sys.path.append("./")
from evaluation.utils.dataset_parser.to_big_video_format import to_big_video_format



def process_srt_files(id_file_path, output_file_path, srt_dir):
    """
    Process SRT files according to an ID file
    
    Parameters:
        id_file_path: Path to the ID file
        output_file_path: Path to the output file
        srt_dir: Directory containing SRT files
    """
    id_file_path = Path(id_file_path)
    output_file_path = Path(output_file_path)
    srt_dir = Path(srt_dir)
    
    if not id_file_path.exists():
        print(f"Error: ID file not found at {id_file_path}")
        return
    
    with open(id_file_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(ids)} IDs to process")
    
    # Process each SRT file and collect results
    results = []
    missing_files = []
    
    for file_id in ids:
        srt_path = srt_dir / f"{file_id}.srt"
        if not srt_path.exists():
            # Try without the extension in the ID
            base_id = file_id.split(".")[0] if "." in file_id else file_id
            srt_path = srt_dir / f"{base_id}.srt"
            
        if not srt_path.exists():
            print(f"Warning: SRT file not found for ID: {file_id}")
            missing_files.append(file_id)
            results.append("")  # Add empty entry to maintain order
            continue

        # Process using the existing to_big_video_format function
        # This function takes a file path, not file content
        processed_content = to_big_video_format(srt_path)
        results.append(processed_content)
        print(f"Processed: {file_id}")
    
    # Write results to the output file
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result + "\n")
    
    print(f"Results written to {output_file_path}")
    if missing_files:
        print(f"Warning: {len(missing_files)} files were missing: {', '.join(missing_files)}")

def generate_eval_result(id_file="./evaluation/test_data/text_data_test.id",
                  output="./evaluation/test_data/eval_result.zh",
                  srt_dir="./evaluation/test_data/srt_output"):
    """把所有多行的srt文件转换为big video format（只有一行），然后批量写入到output文件中
    
    Args:
        id_file: Path to file containing IDs
        output: Path to output file
        srt_dir: Directory containing SRT files
    """
    # Process SRT files
    process_srt_files(id_file, output, srt_dir)

if __name__ == "__main__":
    # TEST
    # to_eval_result(id_file="./evaluation/test_data/test_eval/text_data_test.id",
    #               output="./evaluation/test_data/test_eval/eval_result.zh",
    #               srt_dir="./evaluation/test_data/test_eval/srt_output") 
    
    generate_eval_result(id_file="./evaluation/test_data/text_data_test.id",
                output="./evaluation/test_data/gemini_eval_result.zh",
                srt_dir="./evaluation/test_data/gemini_results") 