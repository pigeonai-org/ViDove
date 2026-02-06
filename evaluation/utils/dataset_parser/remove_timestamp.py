from pathlib import Path
import re
import argparse
from extract_ass_subtitles import extract_ass_subtitles, AssSubtitleExtractor

class RemoveTimestampConverter:
    """Class for converting SRT files to Vidove dataset format（remove timestamp）"""
    
    def __init__(self, id_list=None, target_srt_dir=None, single_file=None, output_file=None, output_dir=None):
        """
        Initialize the converter with file paths
        
        Args:
            id_list: Path to file containing IDs
            target_srt_dir: Directory containing SRT files to process
            single_file: Path to a single SRT file to process
            output_file: Path to output file (for single file output)
            output_dir: Directory to save individual processed files (for batch processing)
        """
        self.id_list = Path(id_list) if id_list else None
        self.target_srt_dir = Path(target_srt_dir) if target_srt_dir else None
        self.single_file = Path(single_file) if single_file else None
        self.output_file = Path(output_file) if output_file else None
        self.output_dir = Path(output_dir) if output_dir else None
        
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

    def process_id_list_to_separate_files(self):
        """
        Process SRT files according to IDs in a file and save each to a separate output file
        """
        if not self.id_list or not self.id_list.exists():
            print(f"Error: ID file not found at {self.id_list}")
            return
        
        if not self.output_dir:
            print("Error: output_dir must be specified for separate file output")
            return
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.id_list, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(ids)} IDs to process")
        
        # Process each SRT file and save to separate output file
        processed_files = []
        missing_files = []
        
        for file_id in ids:
            srt_path = self.target_srt_dir / f"{file_id}.srt"
            if not srt_path.exists():
                # Try without the extension in the ID
                base_id = file_id.split(".")[0] if "." in file_id else file_id
                srt_path = self.target_srt_dir / f"{base_id}.srt"
                
            if not srt_path.exists():
                print(f"Warning: SRT file not found for ID: {file_id}")
                missing_files.append(file_id)
                continue
            
            # Process SRT file to Vidove format
            formatted_text = self.convert_srt_file(srt_path)
            
            # Create output file with same name but .txt extension
            output_file = self.output_dir / f"{file_id}.txt"
            
            # Write formatted text to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            processed_files.append(str(output_file))
            print(f"Processed: {file_id} -> {output_file}")
        
        print(f"Processed {len(processed_files)} files. Output saved to {self.output_dir}")
        if missing_files:
            print(f"Warning: {len(missing_files)} files were missing: {', '.join(missing_files)}")
        
        return processed_files
    
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

if __name__ == "__main__":

    # 单个文件处理
    converter = RemoveTimestampConverter(
        single_file=r"evaluation\test_data\28e6c89c-2a04-45d9-8fdb-6b4ed23f6087_ZH.srt", # for single file processing
        output_file=r"evaluation\test_data\remove_timestamp_result.txt",
    )
    converter.process_single_file()
    
    # 批量处理
    # converter = RemoveTimestampConverter(
    #     id_list=r"evaluation\test_data\text_data_test.id",
    #     target_srt_dir=r"evaluation\test_data\test\srt_output",
    #     output_dir=r"evaluation\test_data\batch_result",
    # )
    # converter.process_id_list_to_separate_files()

