from pydub import AudioSegment
import subprocess
import sys
import os
import requests
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.SRT.srt import SrtScript, SrtSegment
import datetime

class VAD:
    def __init__(self, model_name_or_path: str, src_lang: str, tgt_lang: str, min_segment_seconds: float = 1.0):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.srt = None
        # Drop segments shorter than this many seconds; default 0.0 means keep all
        self.min_segment_seconds = max(0.0, float(min_segment_seconds))

        if model_name_or_path == "API":
            self.model = None
            # Support official env var name from tutorial and a few common fallbacks
            self.api_token = os.getenv("PYANNOTE_API_KEY")
            if not self.api_token:
                raise ValueError(
                    "Set PYANNOTE_API_KEY to use API mode"
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
                output = status_data.get("output") or {}
                diar = output.get("diarization") or []
                segments = diar if isinstance(diar, list) else []
                # Ensure chronological order
                try:
                    segments.sort(key=lambda s: float(s.get("start") or 0.0))
                except Exception:
                    pass
                # print(len(segments), "segments found")
                for segment in segments or []:
                    # API returns seconds as floats already – keep them as floats
                    start_sec = float(segment.get("start") or 0.0)
                    end_sec = float(segment.get("end") or 0.0)
                    speaker = segment.get("speaker")

                    # Basic sanity checks and consistent min-duration filter
                    if end_sec <= start_sec:
                        continue
                    if (end_sec - start_sec) < self.min_segment_seconds:
                        continue

                    srt.segments.append(
                        SrtSegment(
                            src_lang=self.src_lang,
                            tgt_lang=self.tgt_lang,
                            src_text="",
                            translation="",
                            speaker=speaker,
                            start_time=start_sec,
                            end_time=end_sec,
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
    
    def _seconds_to_srt_time(self, secs: float) -> str:
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
            if (turn.end - turn.start) < self.min_segment_seconds:
                continue
            srt.segments.append(SrtSegment(src_lang=self.src_lang, tgt_lang=self.tgt_lang, src_text="", translation="", speaker=speaker, start_time=turn.start, end_time=turn.end, idx=len(srt.segments)))  
        self.srt = srt
        return srt
    
    @staticmethod
    def clip_audio_and_save(srt: SrtScript, audio_path: str, output_dir: str):
        """Cut audio segments quickly.
        Optimization:
        - Normalize once to 16k mono PCM WAV in the output_dir and then stream-copy per segment.
        - Use fast seek (-ss before -i) with sample-accurate PCM.
        - Parallelize clipping with a bounded thread pool.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1) Normalize input once to 16k/mono PCM WAV to avoid re-decode/re-sample N times
        norm_wav = os.path.join(output_dir, "source_16k_mono.wav")
        if not (os.path.isfile(norm_wav) and os.path.getsize(norm_wav) > 0):
            norm_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", audio_path,
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                norm_wav,
            ]
            try:
                subprocess.run(norm_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Error normalizing audio for clipping: {e}")
                # Fallback to original audio if normalization fails
                norm_wav = audio_path

        # 2) Prepare all commands and execute in parallel
        tasks = []
        segments = list(srt.segments)
        # Optional throttling to avoid overloading backend when many segments
        sample_rate = max(1, int(os.getenv("VAD_VIDEO_SAMPLE_RATE", "1")))
        max_clips_env = os.getenv("VAD_VIDEO_MAX_CLIPS", "")
        try:
            max_clips = int(max_clips_env) if max_clips_env.strip() != "" else None
        except Exception:
            max_clips = None
        if sample_rate > 1:
            segments = [seg for i, seg in enumerate(segments) if i % sample_rate == 0]
        if max_clips is not None and max_clips >= 0:
            segments = segments[:max_clips]

        for segment in segments:
            start_time = segment.start_time
            end_time = segment.end_time
            start_ms = int(max(0.0, start_time) * 1000)
            end_ms = int(max(0.0, end_time) * 1000)
            if end_ms <= start_ms:
                continue
            duration_ms = end_ms - start_ms

            start_time_str = str(datetime.timedelta(milliseconds=start_ms))
            duration_str = str(datetime.timedelta(milliseconds=duration_ms))

            output_filename = os.path.join(output_dir, f"segment_{start_ms}_{end_ms}.wav")
            segment.audio_path = output_filename

            # With PCM WAV source, we can stream copy for blazing fast cuts
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", start_time_str, "-i", norm_wav,
                "-t", duration_str,
                "-c", "copy",
                output_filename,
            ]
            tasks.append((segment, cmd, start_ms, end_ms))

        max_workers = int(os.getenv("VAD_FFMPEG_WORKERS", str(min(32, (os.cpu_count() or 4)))))
        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(subprocess.run, cmd, check=True, capture_output=True): (seg, start_ms, end_ms)
                       for (seg, cmd, start_ms, end_ms) in tasks}
            for fut in as_completed(futures):
                seg, s_ms, e_ms = futures[fut]
                try:
                    _ = fut.result()
                except subprocess.CalledProcessError as e:
                    errors.append((s_ms, e_ms, str(e)))
        for s_ms, e_ms, msg in errors:
            print(f"Error processing audio segment {s_ms}-{e_ms}: {msg}")
    
    @staticmethod
    def clip_video_and_save(srt: SrtScript, video_path: str, output_dir: str):
        """Cut video segments efficiently for vision cues.
        Optimization:
        - Use fast seek with stream copy: -ss before -i, -t duration, -c copy.
        - Parallelize clipping with bounded threads.
        - Fallback to re-encode on rare failures.
        Note: Keyframe alignment may cause small timing drift; acceptable for visual cues.
        """
        os.makedirs(output_dir, exist_ok=True)

        tasks = []
        for segment in srt.segments:
            start_time = max(0.0, float(segment.start_time or 0.0))
            end_time = max(0.0, float(segment.end_time or 0.0))
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            if end_ms <= start_ms:
                continue
            duration_ms = end_ms - start_ms

            start_time_str = str(datetime.timedelta(milliseconds=start_ms))
            duration_str = str(datetime.timedelta(milliseconds=duration_ms))

            output_filename = os.path.join(output_dir, f"segment_{start_ms}_{end_ms}.mp4")
            segment.video_path = output_filename

            fast_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", start_time_str, "-i", video_path,
                "-t", duration_str,
                "-c", "copy",
                "-avoid_negative_ts", "1",
                "-movflags", "+faststart",
                output_filename,
            ]
            # Slower but accurate fallback
            slow_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", video_path,
                "-ss", start_time_str, "-t", duration_str,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                output_filename,
            ]
            tasks.append((segment, start_ms, end_ms, fast_cmd, slow_cmd))

        max_workers = int(os.getenv("VAD_FFMPEG_WORKERS", str(min(16, (os.cpu_count() or 4)))))
        errors = []
        def run_with_fallback(fast_cmd, slow_cmd):
            try:
                subprocess.run(fast_cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(slow_cmd, check=True, capture_output=True)
                    return True
                except subprocess.CalledProcessError as e2:
                    return e2

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_with_fallback, fast, slow): (s, s_ms, e_ms)
                       for (s, s_ms, e_ms, fast, slow) in tasks}
            for fut in as_completed(futures):
                seg, s_ms, e_ms = futures[fut]
                try:
                    res = fut.result()
                    if res is not True:
                        errors.append((s_ms, e_ms, str(res)))
                except Exception as e:
                    errors.append((s_ms, e_ms, str(e)))
        for s_ms, e_ms, msg in errors:
            print(f"Error processing video segment {s_ms}-{e_ms}: {msg}")

if __name__ == "__main__":
    vad = VAD("API", "en", "en")
    segments = vad.get_speaker_segments_api("/home/mlp/eason/Y0000003046_74RJksRTFSo_S00085.wav")
    # VAD.clip_audio_and_save(segments, "/home/mlp/eason/Y0000003046_74RJksRTFSo_S00085.wav", ".output")