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

AUDIO_ANALYZE_PROMPT = """
You are an expert audio analyst.

Analyze the provided audio carefully and produce a structured analysis.

Your task includes:

1. CONTENT ANALYSIS
- What is happening in the audio?
- Summarize the main events or speech content.
- Identify key topics.

2. SPEECH ANALYSIS (if speech exists)
- Language(s) detected
- Speaker count (estimate)
- Speaker roles (if inferable)
- Emotion or tone
- Speech clarity (clear, noisy, overlapping)

3. ACOUSTIC ANALYSIS
- Background environment (indoor, outdoor, office, street, etc.)
- Background sounds (music, traffic, typing, crowd, etc.)
- Noise level
- Audio quality assessment

4. TEMPORAL EVENTS
List important timestamps:

[ timestamp_start, timestamp_end, description ]

5. CONFIDENCE
Provide confidence score (0–1).

Return output strictly in JSON format:

{
  "summary": "...",
  "speech": {...},
  "environment": {...},
  "events": [...],
  "confidence": ...
}
"""
