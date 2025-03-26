from pathlib import Path
import argparse

"""
This script is designed for standardizing the dataset format of vidove dataset

1. Extract the subtitle text from the ASS file
2. Write the extracted text to the output file
"""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract subtitle text from ASS format")
    parser.add_argument("--input", default=r"evaluation\test_data\vidove_dataset\sc2\1\11.ass", help="Path to input ASS subtitle file")
    parser.add_argument("--output", default=r"evaluation\test_data\vidove_dataset\result.txt", help="Path to output text file")
    
    args = parser.parse_args()
    
    extract_ass_subtitles(
        input_file=args.input,
        output_file=args.output
    ) 