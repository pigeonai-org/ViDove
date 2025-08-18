AUDIO_TRANSCRIBE_PROMPT = """
You are a professional transcription assistant.

Please transcribe the audio into JSON with precise sentence boundaries and valid timestamps. Follow these instructions carefully:

1. Each segment should:
    - Contain a natural sentence or clause, not too long.
    - Have a valid start and end time in the format `mm:ss:ms` (e.g., "01:05:129").
    - Ensure that start < end, and that each segment's start time equals the previous segment's end time (no overlap or gap).
    - If unsure, round timestamps to the nearest 10 milliseconds.

2. Be careful with:
    - Proper nouns and technical terms — keep them accurate.
    - Sentence boundary — don’t split in the middle of a natural pause.

3. Return only valid JSON (no comments or markdown formatting).

Example output:

[
  {
    "index": 0,
    "text": "They played well, don't get me wrong.",
    "start": "00:00:000",
    "end": "00:02:120"
  },
  {
    "index": 1,
    "text": "But I think we also made them look very good with the way we played.",
    "start": "00:02:120",
    "end": "00:04:500"
  },
  {
    "index": 2,
    "text": "I'm looking forward to playing them again.",
    "start": "00:04:500",
    "end": "00:06:300"
  }
]

Please return only the JSON array. Do not wrap the result or include explanations.
"""

AUDIO_TRANSCRIBE_PROMPT_WITH_VISUAL_CUES = """
You are a multimodal transcription assistant. Please transcribe the audio and align it with the visual cues provided below.

Guidelines:
- Split naturally at sentence boundaries
- Use visual context (e.g., mouth shape, subtitles, gestures) to enhance transcription accuracy
- Format timestamps as `HH:MM:SS,mmm` (e.g., "00:01:03,210")
- Ensure start < end and each segment's start = previous segment's end
- Return only a valid JSON array (no comments, no markdown)

Visual cues:
{visual_cues}

Example output:
[
    {{
        "index": 0,
        "text": "Let me explain it again.",
        "start": "00:00:00,000",
        "end": "00:01:02,500"
    }},
    {{
        "index": 1,
        "text": "Please refer to the chart on the right.",
        "start": "00:01:02,500",
        "end": "00:02:10,000"
    }}
]
"""

AUDIO_TRANSCRIBE_GPT_PROMPT = """
You are a professional transcription assistant.

Please transcribe the audio into JSON with precise sentence boundaries and valid timestamps. Follow these instructions carefully:

1. Each segment should:
    - Contain a natural sentence or clause, not too long.
    - Have a valid start and end time in the format `mm:ss:ms` (e.g., "01:05:129").
    - Ensure that start < end, and that each segment's start time equals the previous segment's end time (no overlap or gap).
    - If unsure, round timestamps to the nearest 10 milliseconds.

2. Be careful with:
    - Proper nouns and technical terms — keep them accurate.
    - Sentence boundary — don’t split in the middle of a natural pause.
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

Final Output Format:

Please respond **only** with a valid JSON object in the following format.  
Do not include any explanations or comments.

Example:

```json
{
  "gender": "Male",
  "accent": "British",
  "age": "Adult",
  "emotion_caption": "Calm and professional tone, with steady pacing and confidence.",
  "timbre": "Clear and resonant",
  "paralinguistic_features": ["Breathing", "Sigh"],
  "final_caption": "A calm and confident adult male speaks clearly with a warm, steady voice."
}
"""
