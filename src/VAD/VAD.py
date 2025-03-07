from pydub import AudioSegment
from pyannote.audio import Pipeline
import subprocess
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.SRT.srt import SrtScript, SrtSegment
import datetime

class VAD:
    def __init__(self, model_name_or_path: str, src_lang: str, tgt_lang: str):
        self.model = Pipeline.from_pretrained(
        model_name_or_path,
        use_auth_token="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",
    )
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.srt = None
    
    @staticmethod
    def load_audio(audio_path: str):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        return audio

    def get_speaker_segments(self, audio_path: str):
        print(f"Processing audio file: {audio_path}")
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        segments = self.model(audio_path)  
        for turn, _, speaker in segments.itertracks(yield_label=True):
            srt.segments.append(SrtSegment(src_lang=self.src_lang, tgt_lang=self.tgt_lang, src_text="", translation="", speaker=speaker, start_time=turn.start, end_time=turn.end))  
        self.srt = srt
        return srt
    
    @staticmethod
    def clip_audio_and_save(srt: SrtScript, audio_path: str, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)

        for segment in srt.segments:
            start_time = segment.start_time
            end_time = segment.end_time
            # Convert time to milliseconds for ffmpeg
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            duration_ms = end_ms - start_ms

            # Format timestamps for ffmpeg
            start_time_str = str(datetime.timedelta(milliseconds=start_ms))
            duration_str = str(datetime.timedelta(milliseconds=duration_ms))

            # Generate output filename
            output_filename = os.path.join(output_dir, f"segment_{start_ms}_{end_ms}.wav")

            # Use ffmpeg to extract segment
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ss", start_time_str,
                "-t", duration_str,
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_filename
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing segment {start_ms}-{end_ms}: {e}")

if __name__ == "__main__":
    vad = VAD("pyannote/speaker-diarization-3.1", "en", "en")
    segments = vad.get_speaker_segments("test.wav")
    VAD.clip_audio_and_save(segments, "test.wav", ".output")