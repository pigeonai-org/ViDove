import re
import sys

def clean_srt_file(input_path, output_path):
    """
    Clean an SRT subtitle file:
    - Remove meaningless subtitle blocks (e.g., only symbols, underscores, signatures, ads)
    - Keep meaningful subtitle lines
    - Renumber the subtitles sequentially
    """
    # Define a list of keywords to filter out unwanted subtitles
    blacklist_keywords = ["字幕", "翻译", "Bilibili", "更新","@"]

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content into blocks separated by empty lines
    blocks = re.split(r'\n\s*\n', content.strip())

    cleaned_blocks = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            timestamp = lines[1]
            text = ' '.join(lines[2:]).strip()

            # Check if the text is meaningful
            if text and not re.fullmatch(r'[\W_]+', text):
                # Filter out blocks containing any blacklist keywords
                if not any(keyword in text for keyword in blacklist_keywords):
                    cleaned_blocks.append((timestamp, text))

    # Rebuild the cleaned subtitle content
    cleaned_content = ""
    for idx, (timestamp, text) in enumerate(cleaned_blocks, start=1):
        cleaned_content += f"{idx}\n{timestamp}\n{text}\n\n"

    # Write the cleaned content to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    print(f"Cleaning complete! {len(cleaned_blocks)} subtitle blocks retained. Output path: {output_path}")