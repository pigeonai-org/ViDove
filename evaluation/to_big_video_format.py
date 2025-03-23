from pathlib import Path
import re

def srt_to_plain_text(srt_path):
    """
    original file example:
    1
    00:00:00,000 --> 00:00:01,000
    Hello
    2
    00:00:01,000 --> 00:00:02,000
    World
    
    output example:
    Hello, World
    """
    
    srt_path = Path(srt_path)
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text_lines = []
    i = 0
    
    # Iterate through each line, skip number lines and timestamp lines
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check if it's a number line (pure digit)
        if line.isdigit():
            i += 1
            # Next line should be a timestamp line
            if i < len(lines) and '-->' in lines[i]:
                i += 1
                # The following non-empty lines are subtitle text until next number or end of file
                while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                    text_line = lines[i].strip()
                    if text_line:
                        text_lines.append(text_line)
                    i += 1
            continue
        
        # If not a number line and not a known marker, treat as text line
        if '-->' not in line:
            text_lines.append(line)
        
        i += 1
    
    # Combine all text lines into one line
    combined_text = ' '.join(text_lines)
    
    # Replace spaces with commas
    combined_text = combined_text.replace(' ', ',')
    
    # Handle possible consecutive commas
    combined_text = re.sub(',+', ',', combined_text)
    
    # Remove trailing comma (if any) and add period
    if combined_text.endswith(','):
        combined_text = combined_text[:-1]
    
    if not combined_text.endswith('.'):
        combined_text += '.'
    
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
        plain_text = srt_to_plain_text(srt_file)
        
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

    input_path = "evaluation/test_data/c0a62310-998d-4c52-b2be-e94ce7f0f3b2_ZH.srt"
    output_dir = "evaluation/test_data/"

    # Convert string paths to Path objects
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Single file conversion
    plain_text = srt_to_plain_text(input_path)
    
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