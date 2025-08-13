from pydub import AudioSegment
from pyannote.audio import Pipeline
import subprocess
import sys
import os
import requests
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.SRT.srt import SrtScript, SrtSegment
import datetime

class VAD:
    def __init__(self, model_name_or_path: str, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.srt = None

        self.backend_endpoint = os.getenv("VAD_ENDPOINT_URL")  # e.g., http://localhost:8000/vad/segment
        if model_name_or_path == "API":
            # API mode: either call our backend endpoint if configured, or call pyannote API directly
            self.model = None
            if not self.backend_endpoint:
                self.api_key = os.getenv("PYANNOTE_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "PYANNOTE_API_KEY is required when using API mode without VAD_ENDPOINT_URL"
                    )
            else:
                self.api_key = None
        else:
            self.model = Pipeline.from_pretrained(
                model_name_or_path,
                use_auth_token=os.getenv("HF_TOKEN"),
            )
    
    @staticmethod
    def load_audio(audio_path: str):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        return audio

    def get_speaker_segments_api(self, audio_path: str, webhook_url: str = None):
        print(f"Processing audio file via API: {audio_path}")
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)

        # Prefer calling our backend VAD endpoint if configured
        if self.backend_endpoint:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                params = {"src_lang": self.src_lang, "tgt_lang": self.tgt_lang}
                resp = requests.post(self.backend_endpoint, files=files, data=params)
            if resp.status_code != 200:
                raise Exception(
                    f"Backend VAD endpoint failed ({resp.status_code}): {resp.text}"
                )
            data = resp.json() or {}
            segments = data.get("segments", [])
            for seg in segments:
                start_time = seg.get("start")
                end_time = seg.get("end")
                speaker = seg.get("speaker")
                if start_time is None or end_time is None:
                    continue
                if end_time - start_time < 1:
                    continue
                srt.segments.append(
                    SrtSegment(
                        src_lang=self.src_lang,
                        tgt_lang=self.tgt_lang,
                        src_text="",
                        translation="",
                        speaker=speaker,
                        start_time=start_time,
                        end_time=end_time,
                        idx=len(srt.segments),
                    )
                )
            self.srt = srt
            return srt

        # Fallback: call pyannote API directly (requires audio_path to be a URL)
        file_url = audio_path
        url = "https://api.pyannote.ai/v1/diarize"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"url": file_url}
        if webhook_url:
            data["webhook"] = webhook_url

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

        job_data = response.json()
        job_id = job_data["jobId"]
        while True:
            status_response = requests.get(f"{url}/{job_id}", headers=headers)
            if status_response.status_code != 200:
                raise Exception(
                    f"Failed to get job status: {status_response.text}"
                )
            status_data = status_response.json()
            if status_data["status"] == "completed":
                for segment in status_data["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    speaker = segment["speaker"]
                    if end_time - start_time < 1:
                        continue
                    srt.segments.append(
                        SrtSegment(
                            src_lang=self.src_lang,
                            tgt_lang=self.tgt_lang,
                            src_text="",
                            translation="",
                            speaker=speaker,
                            start_time=start_time,
                            end_time=end_time,
                            idx=len(srt.segments),
                        )
                    )
                break
            elif status_data["status"] == "failed":
                raise Exception(
                    f"Job failed: {status_data.get('error', 'Unknown error')}"
                )
            time.sleep(5)

        self.srt = srt
        return srt

    def get_speaker_segments(self, audio_path: str, webhook_url: str = None):
        if self.model is None:
            return self.get_speaker_segments_api(audio_path, webhook_url)
            
        print(f"Processing audio file: {audio_path}")
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        segments = self.model(audio_path)  
        for turn, _, speaker in segments.itertracks(yield_label=True):
            if turn.end - turn.start < 1:
                continue
            srt.segments.append(SrtSegment(src_lang=self.src_lang, tgt_lang=self.tgt_lang, src_text="", translation="", speaker=speaker, start_time=turn.start, end_time=turn.end, idx=len(srt.segments)))  
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
            segment.audio_path = output_filename
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
    
    @staticmethod
    def clip_video_and_save(srt: SrtScript, video_path: str, output_dir: str):
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

            output_filename = os.path.join(output_dir, f"segment_{start_time}_{end_time}.mp4")
            segment.video_path = output_filename
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", start_time_str,
                "-t", duration_str,
                output_filename
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing segment {start_ms}-{end_ms}: {e}")

if __name__ == "__main__":
    vad = VAD("API", "en", "en")
    segments = vad.get_speaker_segments_api("/home/mlp/eason/Y0000003046_74RJksRTFSo_S00085.wav")
    # VAD.clip_audio_and_save(segments, "/home/mlp/eason/Y0000003046_74RJksRTFSo_S00085.wav", ".output")