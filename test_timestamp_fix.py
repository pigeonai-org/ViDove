#!/usr/bin/env python3
"""
Test script to verify the Gemini timestamp normalization fix.
This simulates the timestamp parsing issue and verifies the fix.
"""

def seconds_to_srt_time(secs: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    if secs is None:
        secs = 0.0
    if secs < 0:
        secs = 0.0
    total_ms = int(round(float(secs) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_timestamp(time_str, audio_duration_secs):
    """
    Normalize timestamp from Gemini response to HH:MM:SS,mmm format.
    """
    import re
    
    if not time_str:
        return "00:00:00,000"
    
    time_str = str(time_str).strip()
    
    # Case 1: Plain decimal seconds (e.g., "1.250", "65.5")
    try:
        secs = float(time_str)
        # Clamp to audio duration
        secs = max(0.0, min(secs, audio_duration_secs))
        return seconds_to_srt_time(secs)
    except ValueError:
        pass
    
    # Case 2: Time format with separators
    normalized = re.sub(r'[^0-9]', ':', time_str)
    parts = [p for p in normalized.split(':') if p]
    
    if not parts:
        return "00:00:00,000"
    
    # Pad to ensure we have at least 4 parts [HH, MM, SS, mmm]
    while len(parts) < 4:
        parts.insert(0, '0')
    
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2])
        ms = int(parts[3])
        
        # Convert to total seconds
        total_secs = hh * 3600 + mm * 60 + ss + ms / 1000.0
        
        # If the calculated time exceeds audio duration by a lot,
        # the format is likely wrong
        if total_secs > audio_duration_secs * 2:
            # Try alternative: treat first part as minutes (MM:SS:mmm format)
            total_secs = hh * 60 + mm + ss / 1000.0 + ms / 1000000.0
            
            # Still too large? Last resort: everything in milliseconds
            if total_secs > audio_duration_secs * 2:
                total_secs = (hh * 60000 + mm * 1000 + ss + ms / 1000.0) / 1000.0
        
        # Clamp to valid range
        total_secs = max(0.0, min(total_secs, audio_duration_secs))
        return seconds_to_srt_time(total_secs)
        
    except (ValueError, IndexError):
        return "00:00:00,000"


def test_timestamp_normalization():
    """Test various timestamp formats"""
    
    # Assume a 10-second audio clip
    audio_duration = 10.0
    
    test_cases = [
        # (input_timestamp, expected_output_description)
        ("00:00:000", "Start of clip"),
        ("00:00:500", "0.5 seconds"),
        ("00:01:250", "1.25 seconds"),
        ("00:05:500", "5.5 seconds"),
        ("00:10:000", "10 seconds (end of clip)"),
        ("0.500", "0.5 seconds (decimal format)"),
        ("5.250", "5.25 seconds (decimal format)"),
        ("10.0", "10 seconds (decimal format)"),
        ("00:00:805", "0.805 seconds (problematic format)"),
        ("00:00:885", "0.885 seconds (problematic format)"),
    ]
    
    print("Testing timestamp normalization with 10-second audio clip:")
    print("=" * 70)
    
    for input_ts, description in test_cases:
        output = normalize_timestamp(input_ts, audio_duration)
        print(f"Input: {input_ts:15s} -> Output: {output} ({description})")
    
    print("\n" + "=" * 70)
    print("Testing the specific issue from the SRT file:")
    print("=" * 70)
    
    # The issue: segments at 9:33 with only 80ms duration
    # This happens when a VAD segment starts at 573 seconds (9:33)
    segment_start_time = 573.0  # 9 minutes 33 seconds
    
    # Gemini returns timestamps relative to the clip
    # Let's say the clip is 5 seconds long
    clip_duration = 5.0
    
    # Gemini might return these timestamps for text within the clip:
    gemini_timestamps = [
        ("00:00:805", "00:00:885"),  # This is the problematic format
        ("00:01:500", "00:02:200"),
        ("0.805", "0.885"),  # If it returns decimal format
        ("1.500", "2.200"),
    ]
    
    print(f"\nVAD segment starts at: {segment_start_time} seconds ({seconds_to_srt_time(segment_start_time)})")
    print(f"Clip duration: {clip_duration} seconds\n")
    
    for start_ts, end_ts in gemini_timestamps:
        start_normalized = normalize_timestamp(start_ts, clip_duration)
        end_normalized = normalize_timestamp(end_ts, clip_duration)
        
        # Convert to seconds for offset calculation
        start_secs = parse_srt_to_seconds(start_normalized)
        end_secs = parse_srt_to_seconds(end_normalized)
        
        # Add offset (this is what task.py does)
        final_start = segment_start_time + start_secs
        final_end = segment_start_time + end_secs
        
        duration_ms = (end_secs - start_secs) * 1000
        
        print(f"Gemini returns: {start_ts} -> {end_ts}")
        print(f"  Normalized: {start_normalized} -> {end_normalized}")
        print(f"  After offset: {seconds_to_srt_time(final_start)} -> {seconds_to_srt_time(final_end)}")
        print(f"  Duration: {duration_ms:.0f} ms")
        print()


def parse_srt_to_seconds(srt_time):
    """Convert SRT format HH:MM:SS,mmm to seconds"""
    import datetime
    dt = datetime.datetime.strptime(srt_time, "%H:%M:%S,%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


if __name__ == "__main__":
    test_timestamp_normalization()
