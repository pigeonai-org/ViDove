from pathlib import Path
import re

def remove_timestamp_and_num(lines):
    """Remove timestamp and number lines from SRT file content
    
    Args:
        lines: List of lines from SRT file
        
    Returns:
        List of subtitle text lines with timestamps and numbers removed
    """
    text_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Skip number and timestamp lines
        if line.isdigit():
            i += 1
            if i < len(lines) and '-->' in lines[i]:
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                    text_line = lines[i].strip()
                    if text_line:
                        text_lines.append(text_line)
                    i += 1
            continue
        
        # Keep non-timestamp text lines
        if '-->' not in line:
            text_lines.append(line)
        
        i += 1
        
    return text_lines

def align_to_one_line(text_lines):
    """Combine multiple lines into one line
    
    Args:
        text_lines: List of subtitle text lines
        
    Returns:
        Single string with all lines combined
    """
    return ' '.join(text_lines)

def add_comma(text):
    """Add commas between text segments and period at end
    
    Args:
        text: Combined text string
        
    Returns:
        Text with commas between segments and period at end
    """
    # Replace spaces with commas
    text = text.replace(' ', '，')
    
    # Handle consecutive commas
    text = re.sub('，+', '，', text)
    
    # Remove trailing comma and add period
    if text.endswith('，'):
        text = text[:-1]
        
    if not text.endswith('。'):
        text += '。'
        
    return text

def to_big_video_format(srt_path):
    """Convert SRT file to big video format
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        Text in big video format (single line with commas)
    """
    srt_path = Path(srt_path)
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text_lines = remove_timestamp_and_num(lines)
    combined_text = align_to_one_line(text_lines)
    formatted_text = add_comma(combined_text)
    
    # 需要remove comma就用这个
    # return formatted_text

    # 不需要remove comma就用这个
    return combined_text

def batch_convert_srt_files(input_dir, output_dir=None, output_suffix="_plain"):
    """
    Batch convert all SRT files in a directory
    
    Parameters:
        input_dir: Directory containing SRT files
        output_dir: Output directory, if None use input directory
        output_suffix: Suffix for output files
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all SRT files
    srt_files = list(input_dir.glob("**/*.srt"))
    
    for srt_file in srt_files:
        # Convert SRT to plain text
        plain_text = to_big_video_format(srt_file)
        
        # Create output file path
        relative_path = srt_file.relative_to(input_dir)
        output_file = output_dir / f"{relative_path.stem}{output_suffix}.txt"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write converted text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(plain_text)
        
        print(f"Converted {srt_file} to {output_file}")

if __name__ == "__main__":
    # Test if the function works properly

    input_path = "evaluation/test_data/28e6c89c-2a04-45d9-8fdb-6b4ed23f6087_ZH.srt"
    output_dir = "evaluation/test_data/"

    # Convert string paths to Path objects
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Single file conversion
    plain_text = to_big_video_format(input_path)
    
    if output_dir is None:
        output_file = input_path.with_suffix('.txt')
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / input_path.with_suffix('.txt').name
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(plain_text)
    
    print(f"Converted {input_path} to {output_file}")
    
    # elif input_path.is_dir():
    #     # Batch convert directory
    #     batch_convert_srt_files(input_path, output_dir)
    
    # else:
    #     print(f"Error: {input_path} is not a valid SRT file or directory")
    #     sys.exit(1) 