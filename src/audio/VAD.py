from pydub import AudioSegment
import subprocess
import sys
import os
import requests
import time
import uuid
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.SRT.srt import SrtScript, SrtSegment
import datetime

class VAD:
    def __init__(self, model_name_or_path: str, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.srt = None
        
        if model_name_or_path == "API":
            self.model = None
            # Support official env var name from tutorial and a few common fallbacks
            self.api_token = (
                os.getenv("PYANNOTEAI_API_TOKEN")
                or os.getenv("PYANNOTE_API_TOKEN")
                or os.getenv("PYANNOTE_API_KEY")
            )
            if not self.api_token:
                raise ValueError(
                    "Set PYANNOTEAI_API_TOKEN (or PYANNOTE_API_TOKEN/PYANNOTE_API_KEY) to use API mode"
                )
            # Back-compat attribute if referenced elsewhere
            self.api_key = self.api_token
        else:
            # Lazy import to avoid requiring pyannote when using API mode only
            from pyannote.audio import Pipeline  # type: ignore
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

    def _auth_headers_json(self):
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _is_url(path) -> bool:
        s = str(path)
        return s.startswith("http://") or s.startswith("https://") or s.startswith("media://")

    def upload_media(self, audio_path: str) -> str:
        """Upload a local file to pyannote temporary media storage and return the media:// URL.

        If audio_path is already an http(s) or media:// URL, return it unchanged.
        """
        if os.path.isfile(audio_path) and not self._is_url(audio_path):
            input_path = Path(audio_path)
            # API expects media://object-key where object-key is (per error) alpha-numeric.
            # Use a safe, unique key without slashes.
            object_key = f"vidove{uuid.uuid4().hex}"
            media_input_url = f"media://{object_key}"

            presign_endpoint = "https://api.pyannote.ai/v1/media/input"
            body = {"url": media_input_url}
            resp = requests.post(presign_endpoint, json=body, headers=self._auth_headers_json())
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise Exception(
                    f"Failed to create pre-signed URL ({resp.status_code}): {resp.text}\n"
                    f"Sent body: {body}"
                ) from e
            presigned = resp.json().get("url")
            if not presigned:
                raise Exception("Pre-signed URL response missing 'url' field")

            with open(input_path, "rb") as f:
                put_resp = requests.put(presigned, data=f, headers={"Content-Type": "application/octet-stream"})
                try:
                    put_resp.raise_for_status()
                except requests.HTTPError as e:
                    raise Exception(
                        f"Upload to pre-signed URL failed ({put_resp.status_code}): {put_resp.text}"
                    ) from e
            return media_input_url
        return audio_path

    def create_diarization_job(self, media_url: str, webhook_url: str | None = None) -> str:
        """Create a diarization job for the given media:// or http(s) URL and return job id."""
        jobs_endpoint = "https://api.pyannote.ai/v1/diarize"
        job_body = {"url": media_url}
        if webhook_url:
            job_body["webhook"] = webhook_url
        create_resp = requests.post(jobs_endpoint, json=job_body, headers=self._auth_headers_json())
        create_resp.raise_for_status()
        job_data = create_resp.json()
        job_id = job_data.get("jobId") or job_data.get("id")
        if not job_id:
            raise Exception(f"Unexpected job creation response: {job_data}")
        return job_id

    def poll_diarization_results(self, job_id: str, initial_delay: float = 2.0, poll_interval: float = 1.0, timeout: float = 600.0) -> SrtScript:
        """Poll diarization job until completion and return an SrtScript with segments.

        Adds a short initial delay and handles 404 responses (job not yet registered)
        by retrying until the timeout expires.
        """
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        jobs_endpoint = "https://api.pyannote.ai/v1/jobs"
        poll_endpoint = f"{jobs_endpoint}/{job_id}"
        deadline = time.time() + timeout

        if initial_delay and initial_delay > 0:
            time.sleep(initial_delay)

        while True:
            status_resp = requests.get(poll_endpoint, headers={"Authorization": f"Bearer {self.api_token}"})

            # Handle race condition where the job is not yet registered
            if status_resp.status_code == 404:
                if time.time() >= deadline:
                    raise TimeoutError(f"Job not registered before timeout: {job_id}")
                time.sleep(min(2.0, poll_interval))
                continue

            # Optional: polite backoff on rate limiting
            if status_resp.status_code == 429:
                if time.time() >= deadline:
                    raise TimeoutError("Polling rate-limited until timeout expired")
                time.sleep(5.0)
                continue

            if status_resp.status_code != 200:
                raise Exception(f"Failed to get job status: {status_resp.status_code} {status_resp.text}")

            status_data = status_resp.json()
            status = status_data.get("status")
            if status == "succeeded":
                segments = (
                    status_data.get("output").get("diarization") or []
                )
                for segment in segments or []:
                    start_time = segment.get("start")
                    end_time = segment.get("end")
                    speaker = segment.get("speaker")
                    if start_time is None or end_time is None:
                        continue
                    if (end_time - start_time) < 1:
                        continue
                    srt.segments.append(
                        SrtSegment(
                            src_lang=self.src_lang,
                            tgt_lang=self.tgt_lang,
                            src_text="",
                            translation="",
                            speaker=speaker,
                            start_time=float(start_time),
                            end_time=float(end_time),
                            idx=len(srt.segments),
                        )
                    )
                break
            elif status == "failed":
                raise Exception(f"Job failed: {status_data.get('error') or status_data}")

            if time.time() >= deadline:
                raise TimeoutError("Diarization job polling timed out")
            time.sleep(poll_interval)
        self.srt = srt
        return srt

    def get_speaker_segments_api(self, audio_path: str, webhook_url: str = None):
        """High-level wrapper: upload -> create job -> poll for results."""
        print(f"Processing audio file via API: {audio_path}")
        media_url = self.upload_media(audio_path)
        job_id = self.create_diarization_job(media_url, webhook_url)
        return self.poll_diarization_results(job_id)

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