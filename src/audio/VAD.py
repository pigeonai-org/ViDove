from abc import ABC, abstractmethod
import datetime
import io
import os
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger

from pydub import AudioSegment

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.SRT.srt import SrtScript

logger = getLogger(__name__)

class VAD(ABC):
    def __init__(self, src_lang: str, tgt_lang: str, min_segment_seconds: float = 1.0):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.srt: SrtScript | None = None
        # Drop segments shorter than this many seconds; default 0.0 means keep all
        self.min_segment_seconds = max(0.0, float(min_segment_seconds))

    @abstractmethod
    def get_speaker_segments(self, audio_path: str, webhook_url: str | None = None) -> SrtScript:
        """Return an ``SrtScript`` describing speaker segments for ``audio_path``."""
        raise NotImplementedError

    @staticmethod
    def load_audio(audio_path: str) -> AudioSegment:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        return audio

    @staticmethod
    def _audiosegment_to_wav_bytes(segment: AudioSegment) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(segment.channels)
            wf.setsampwidth(segment.sample_width)
            wf.setframerate(segment.frame_rate)
            wf.writeframes(segment.raw_data)
        return buffer.getvalue()

    @staticmethod
    def _write_audiosegment_to_wav(segment: AudioSegment, output_path: str) -> None:
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(segment.channels)
            wf.setsampwidth(segment.sample_width)
            wf.setframerate(segment.frame_rate)
            wf.writeframes(segment.raw_data)

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
    
    @staticmethod
    def clip_audio_and_save(srt: SrtScript, audio_path: str, output_dir: str):
        """Cut audio segments quickly.
        Optimization:
        - Load the full track once with pydub and operate in-memory to avoid repeated ffmpeg I/O.
        - Normalize to 16k mono PCM via AudioSegment utilities.
        - Parallelize file writes with a bounded thread pool.
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            base_audio = AudioSegment.from_file(audio_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load audio %s: %s", audio_path, exc)
            raise

        normalized = (
            base_audio.set_channels(1)
            .set_frame_rate(16000)
            .set_sample_width(2)
        )

        tasks = []
        segments = list(srt.segments)
        # Optional throttling to avoid overloading backend when many segments
        sample_rate = max(
            1,
            int(os.getenv("VAD_AUDIO_SAMPLE_RATE", os.getenv("VAD_VIDEO_SAMPLE_RATE", "1"))),
        )
        max_clips_env = os.getenv(
            "VAD_AUDIO_MAX_CLIPS",
            os.getenv("VAD_VIDEO_MAX_CLIPS", ""),
        )
        try:
            max_clips = int(max_clips_env) if max_clips_env.strip() != "" else None
        except Exception:
            max_clips = None
        if sample_rate > 1:
            segments = [seg for i, seg in enumerate(segments) if i % sample_rate == 0]
        if max_clips is not None and max_clips >= 0:
            segments = segments[:max_clips]

        full_duration_ms = len(normalized)

        for segment in segments:
            start_time = segment.start_time
            end_time = segment.end_time
            start_ms = int(max(0.0, start_time) * 1000)
            end_ms = int(max(0.0, end_time) * 1000)
            if end_ms <= start_ms:
                continue
            end_ms = min(end_ms, full_duration_ms)
            if end_ms <= start_ms:
                continue
            output_filename = os.path.join(output_dir, f"segment_{start_ms}_{end_ms}.wav")
            segment.audio_path = output_filename

            tasks.append((start_ms, end_ms, output_filename))

        max_workers = int(os.getenv("VAD_FFMPEG_WORKERS", str(min(32, (os.cpu_count() or 4)))))
        errors = []

        def write_segment(start_ms: int, end_ms: int, output_filename: str):
            clip = normalized[start_ms:end_ms]
            if len(clip) == 0:
                return
            VAD._write_audiosegment_to_wav(clip, output_filename)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(write_segment, start_ms, end_ms, output_filename): (start_ms, end_ms)
                for (start_ms, end_ms, output_filename) in tasks
            }
            for fut in as_completed(futures):
                s_ms, e_ms = futures[fut]
                try:
                    fut.result()
                except Exception as e:  # pragma: no cover - defensive
                    errors.append((s_ms, e_ms, str(e)))
        for s_ms, e_ms, msg in errors:
            logger.error("Error processing audio segment %s-%s: %s", s_ms, e_ms, msg)
    
    @staticmethod
    def clip_video_and_save(srt: SrtScript, video_path: str, output_dir: str):
        """Cut video segments efficiently for vision cues.
        Optimization:
        - Pre-slice the audio track with AudioSegment to limit ffmpeg invocations.
        - Use fast seek stream copies for the video stream while piping fresh audio bytes.
        Note: Keyframe alignment may cause small timing drift; acceptable for visual cues.
        """
        os.makedirs(output_dir, exist_ok=True)

        video_audio = AudioSegment.from_file(video_path).set_sample_width(2)
        full_duration_ms = len(video_audio)

        tasks = []
        segments = list(srt.segments)
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
            start_time = max(0.0, float(segment.start_time or 0.0))
            end_time = max(0.0, float(segment.end_time or 0.0))
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            if end_ms <= start_ms:
                continue
            end_ms = min(end_ms, full_duration_ms)
            if end_ms <= start_ms:
                continue
            duration_ms = end_ms - start_ms

            start_time_str = str(datetime.timedelta(milliseconds=start_ms))
            duration_str = str(datetime.timedelta(milliseconds=duration_ms))

            output_filename = os.path.join(output_dir, f"segment_{start_ms}_{end_ms}.mp4")
            segment.video_path = output_filename

            tasks.append((start_ms, end_ms, start_time_str, duration_str, output_filename))

        try:
            affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - platform specific
            affinity = os.cpu_count() or 1
        logger.info("Preparing %s video segments for clipping with %s CPU cores", len(tasks), affinity)

        max_workers = int(os.getenv("VAD_FFMPEG_WORKERS", str(min(16, (os.cpu_count() or 4)))))
        errors = []

        def process_segment(start_ms: int, end_ms: int, start_time_str: str, duration_str: str, output_filename: str):
            audio_clip = video_audio[start_ms:end_ms]
            if len(audio_clip) == 0:
                return None
            audio_bytes = VAD._audiosegment_to_wav_bytes(audio_clip)
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", start_time_str, "-i", video_path,
                "-i", "pipe:0",
                "-t", duration_str,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "pcm_s16le",
                "-shortest",
                "-avoid_negative_ts", "1",
                "-movflags", "+faststart",
                output_filename,
            ]
            try:
                subprocess.run(
                    cmd,
                    input=audio_bytes,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                return None
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
                return stderr

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_segment, s_ms, e_ms, start_str, duration_str, output_filename): (s_ms, e_ms)
                for (s_ms, e_ms, start_str, duration_str, output_filename) in tasks
            }
            for fut in as_completed(futures):
                s_ms, e_ms = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        errors.append((s_ms, e_ms, result))
                except Exception as e:
                    errors.append((s_ms, e_ms, str(e)))
        for s_ms, e_ms, msg in errors:
            logger.error("Error processing video segment %s-%s: %s", s_ms, e_ms, msg)

        logger.info("Finished processing video segments.")