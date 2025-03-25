from pathlib import Path
import re
import argparse

class VidoveFormatConverter:
    """Class for converting SRT files to Vidove dataset format"""
    
    def __init__(self, id_file=None, srt_dir=None, output_file=None, single_file=None):
        """
        Initialize the converter with file paths
        
        Args:
            id_file: Path to file containing IDs
            srt_dir: Directory containing SRT files
            output_file: Path to output file
            single_file: Path to a single SRT file to process
        """
        self.id_file = Path(id_file) if id_file else None
        self.srt_dir = Path(srt_dir) if srt_dir else None
        self.output_file = Path(output_file) if output_file else None
        self.single_file = Path(single_file) if single_file else None
        
    def convert_srt_file(self, srt_path):
        """
        Convert an SRT file to Vidove dataset format
        
        Example output:
        欢迎回来 我们看到的是位于北美服务器上的
        一场星际争霸II顶级五局三胜系列赛
        是时候来点ZVP
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            Formatted string with each subtitle as a separate line
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
                    subtitle_text = []
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        text_line = lines[i].strip()
                        if text_line:
                            subtitle_text.append(text_line)
                        i += 1
                    
                    # Add the complete subtitle as one line
                    if subtitle_text:
                        text_lines.append(' '.join(subtitle_text))
                continue
            
            # If not a number line and not a known marker, treat as text line
            if '-->' not in line:
                text_lines.append(line)
            
            i += 1
        
        # Join lines with newlines - no comma conversion needed
        formatted_text = '\n'.join(text_lines)
        
        return formatted_text

    def batch_convert_directory(self, input_dir, output_file):
        """
        Batch convert all SRT files in a directory and combine into one output file
        
        Args:
            input_dir: Directory containing SRT files
            output_file: Path to output file
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        
        # Get all SRT files
        srt_files = sorted(list(input_dir.glob("**/*.srt")))
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        all_texts = []
        for srt_file in srt_files:
            # Convert SRT to Vidove format
            formatted_text = self.convert_srt_file(srt_file)
            all_texts.append(formatted_text)
            print(f"Processed {srt_file}")
        
        # Write all converted texts to single file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_texts))
        
        print(f"Combined {len(srt_files)} files into {output_file}")
        return output_file

    def process_id_list(self):
        """
        Process SRT files according to IDs in a file and combine into one output file
        """
        if not self.id_file or not self.id_file.exists():
            print(f"Error: ID file not found at {self.id_file}")
            return
        
        with open(self.id_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(ids)} IDs to process")
        
        # Process each SRT file and collect results
        all_texts = []
        missing_files = []
        
        for file_id in ids:
            srt_path = self.srt_dir / f"{file_id}.srt"
            if not srt_path.exists():
                # Try without the extension in the ID
                base_id = file_id.split(".")[0] if "." in file_id else file_id
                srt_path = self.srt_dir / f"{base_id}.srt"
                
            if not srt_path.exists():
                print(f"Warning: SRT file not found for ID: {file_id}")
                missing_files.append(file_id)
                all_texts.append("")  # Add empty entry to maintain order
                continue
            
            # Process SRT file to Vidove format
            formatted_text = self.convert_srt_file(srt_path)
            all_texts.append(formatted_text)
            print(f"Processed: {file_id}")
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write all converted texts to single file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_texts))
        
        print(f"Combined {len(ids) - len(missing_files)} files into {self.output_file}")
        if missing_files:
            print(f"Warning: {len(missing_files)} files were missing: {', '.join(missing_files)}")
        
        return self.output_file
    
    def process_single_file(self):
        """Process a single SRT file"""
        if not self.single_file or not self.single_file.exists():
            print(f"Error: Single file not found at {self.single_file}")
            return
            
        formatted_text = self.convert_srt_file(self.single_file)
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
            
        print(f"Converted {self.single_file} to {self.output_file}")
        return self.output_file
    
    def run(self):
        """Run the converter based on input parameters"""
        if self.single_file:
            return self.process_single_file()
        elif self.id_file and self.srt_dir and self.output_file:
            return self.process_id_list()
        else:
            print("Error: Missing required parameters")
            return None

class AssSubtitleExtractor:
    """Class for extracting translation content from ASS subtitle format"""
    
    def __init__(self, input_file=None, output_file=None):
        """
        Initialize the extractor with file paths
        
        Args:
            input_file: Path to ASS subtitle file or text file containing ASS dialogues
            output_file: Path to output file
        """
        self.input_file = Path(input_file) if input_file else None
        self.output_file = Path(output_file) if output_file else None
    
    def extract_text_from_line(self, line):
        """
        Extract only the subtitle text from an ASS dialogue line
        
        Args:
            line: A line in ASS format starting with "Dialogue:"
            
        Returns:
            Extracted subtitle text after the last comma
        """
        if not line.startswith("Dialogue:"):
            return line.strip()
        
        # Split by commas and get the last part (the actual subtitle text)
        parts = line.split(',')
        if len(parts) > 8:  # Ensure there are enough parts
            return parts[-1].strip()
        return ""
    
    def extract_from_file(self):
        """
        Extract subtitle text from an ASS file
        
        Returns:
            List of extracted subtitle text lines
        """
        if not self.input_file or not self.input_file.exists():
            print(f"Error: Input file not found at {self.input_file}")
            return []
        
        extracted_lines = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Dialogue:"):
                    text = self.extract_text_from_line(line)
                    if text:
                        extracted_lines.append(text)
        
        return extracted_lines
    
    def process(self):
        """
        Process the input file and write extracted text to output file
        
        Returns:
            Path to the output file if successful, None otherwise
        """
        if not self.input_file or not self.output_file:
            print("Error: Both input and output files must be specified")
            return None
        
        extracted_lines = self.extract_from_file()
        
        if not extracted_lines:
            print("Warning: No subtitle text extracted")
            return None
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write extracted lines to output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(extracted_lines))
        
        print(f"Extracted {len(extracted_lines)} subtitle lines to {self.output_file}")
        return self.output_file

def convert_srt_files(id_file=None, srt_dir=None, output_file=None, single_file=None):
    """
    Function to convert SRT files to Vidove dataset format
    
    Args:
        id_file: Path to file containing IDs
        srt_dir: Directory containing SRT files
        output_file: Path to output file
        single_file: Path to a single SRT file to process
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    converter = VidoveFormatConverter(
        id_file=id_file,
        srt_dir=srt_dir,
        output_file=output_file,
        single_file=single_file
    )
    return converter.run()

def extract_ass_subtitles(input_file=None, output_file=None):
    """
    Function to extract subtitle text from ASS format
    
    Args:
        input_file: Path to ASS subtitle file
        output_file: Path to output file
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    extractor = AssSubtitleExtractor(
        input_file=input_file,
        output_file=output_file
    )
    return extractor.process()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert subtitles to Vidove dataset format")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # SRT converter subcommand
    srt_parser = subparsers.add_parser("srt", help="Convert SRT files to Vidove format")
    srt_parser.add_argument("--id-file", default="./evaluation/test_data/text_data_test.id",
                     help="Path to file containing IDs (default: ./evaluation/test_data/text_data_test.id)")
    srt_parser.add_argument("--output", default="./evaluation/test_data/eval_result.zh",
                     help="Path to output file (default: ./evaluation/test_data/eval_result.zh)")
    srt_parser.add_argument("--srt-dir", default="./evaluation/test_data/srt_output",
                     help="Directory containing SRT files (default: ./evaluation/test_data/srt_output)")
    srt_parser.add_argument("--single-file", help="Process a single SRT file instead of using ID file")
    
    # ASS extractor subcommand
    ass_parser = subparsers.add_parser("ass", help="Extract subtitle text from ASS format")
    ass_parser.add_argument("--input", required=True, help="Path to input ASS subtitle file")
    ass_parser.add_argument("--output", required=True, help="Path to output text file")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    convert_srt_files(
        id_file=args.id_file,
        srt_dir=args.srt_dir,
        output_file=args.output,
        single_file=args.single_file
    )

    # extract_ass_subtitles(
    #     input_file=args.input,
    #     output_file=args.output
    # )