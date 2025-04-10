AUDIO_TRANSCRIBE_PROMPT = """
Transcribe the audio into text, maintaining the original text and timestamps.
Please follow these guidelines:
- Split sentences naturally at sentence boundaries
- Keep sentences concise and not too long
- Pay special attention to technical terms and proper nouns
- Ensure timestamps align with sentence breaks

output in json format:
{
    [
        {
            "index": 0,
            "text": <text> (text of the transcript),
            "start": <start_time> (hh:mm:ss, ms), 
            "end": <end_time> (hh:mm:ss, ms)
        },
        {
            "index": 1,
            "text": <text> (text of the transcript),
            "start": <start_time> (hh:mm:ss, ms),
            "end": <end_time> (hh:mm:ss, ms)
        }
    ]
}
"""

AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES = """
Transcribe the audio into text, maintaining the original text and timestamps.
Please follow these guidelines:
- Split sentences naturally at sentence boundaries
- Keep sentences concise and not too long
- Use the provided visual context to improve transcription accuracy
- Pay special attention to technical terms and proper nouns that appear in the video
- Ensure timestamps align with sentence breaks

Visual information: {visual_cues}

output in json format:
{{
    [
        {{
            "index": 0
            "text": <text> (text of the transcript),
            "start": <start_time> (hh:mm:ss, ms),
            "end": <end_time> (hh:mm:ss, ms)
        }},
        {{
            "index": 1,
            "text": <text> (text of the transcript),
            "start": <start_time> (hh:mm:ss, ms),
            "end": <end_time> (hh:mm:ss, ms)
        }},
        ...        
    ]
}}
"""

AUDIO_ANALYZE_PROMPT = """Analyze the given audio and provide a detailed, structured description by addressing each of the following components. Your final output should be well-organized and result in a rich, natural-sounding caption.

Speaker Profile:
Identify key characteristics of the speaker:
- Gender: Male / Female
- Accent: (Specify if identifiable)
- Age Group:
  - Toddler (0–3)
  - Child (4–12)
  - Teenager (13–19)
  - Adult (19–39)
  - Middle-aged (40–59)
  - Senior (60+)

Emotional Tone:
Describe the emotional tone of the speech, including any notable shifts in expression or tonality.
Example: "Calm and explanatory: the voice is steady, with a soft sense of resignation."

Timbre:
Summarize the voice's quality in a few words.
Example: "Warm and clear"


Paralinguistic Cues:
Identify any non-verbal vocal elements present in the audio. Use the list below, or assign a custom label if needed:
["Throat Clearing", "Groan", "Whispering", "Crying", "Singing", "Grunt", "Yawn", "Screaming", "Yell", "Breathing", "Sniff", "Sneeze", "Sigh", "Gasp", "Blow", "Suction", "Hawk", "Cough"]


Final Caption:
Compose a 1–2 sentence natural description that:
1. Integrates all relevant vocal characteristics.
2. Focuses on how the speech is delivered, not what is said.
3. Is clear, concise, and cohesive.
Example: "A confident adult male speaks with steady energy and moderate pitch. A soft sigh at the end adds emotional depth to his composed delivery."

Final Output Format (JSON):
{
  "gender": "<Male/Female>",
  "accent": "<Accent>",
  "age": "<Toddler/Child/Teenager/Adult/Middle-aged/Senior>",
  "emotion_caption": "<Detailed emotion description>",
  "timbre": "<Timbre description>",
  "paralinguistic_features": ["<Detected features>"],
  "final_caption": "<1-2 sentence enriched description based on attributes>"
}
"""
