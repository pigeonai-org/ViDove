from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from logging import getLogger

import requests

from src.SRT.srt import SrtScript, SrtSegment
from src.audio.VAD import VAD

logger = getLogger(__name__)


class APIPyannoteVAD(VAD):
    """VAD provider backed by the pyannote hosted API."""

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        min_segment_seconds: float = 1.0,
        *,
        model: str = "precision-2",
        api_token: str | None = None,
    ) -> None:
        super().__init__(src_lang, tgt_lang, min_segment_seconds)
        self.api_token = api_token or os.getenv("PYANNOTE_API_KEY")
        self.model = model
        if not self.api_token:
            raise ValueError("Set PYANNOTE_API_KEY or pass api_token to use API mode")

    def _auth_headers_json(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _is_url(path: str) -> bool:
        s = str(path)
        return (
            s.startswith("http://")
            or s.startswith("https://")
            or s.startswith("media://")
        )

    def upload_media(self, audio_path: str) -> str:
        """Upload a local file to pyannote temporary media storage and return the media:// URL.

        If ``audio_path`` is already an http(s) or media:// URL, return it unchanged.
        """
        if os.path.isfile(audio_path) and not self._is_url(audio_path):
            input_path = Path(audio_path)
            object_key = f"vidove{uuid.uuid4().hex}"
            media_input_url = f"media://{object_key}"

            presign_endpoint = "https://api.pyannote.ai/v1/media/input"
            body = {"url": media_input_url}
            resp = requests.post(
                presign_endpoint, json=body, headers=self._auth_headers_json()
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise Exception(
                    f"Failed to create pre-signed URL ({resp.status_code}): {resp.text}\nSent body: {body}"
                ) from exc
            presigned = resp.json().get("url")
            if not presigned:
                raise Exception("Pre-signed URL response missing 'url' field")

            with open(input_path, "rb") as file_handle:
                put_resp = requests.put(
                    presigned,
                    data=file_handle,
                    headers={"Content-Type": "application/octet-stream"},
                )
                try:
                    put_resp.raise_for_status()
                except requests.HTTPError as exc:
                    raise Exception(
                        f"Upload to pre-signed URL failed ({put_resp.status_code}): {put_resp.text}"
                    ) from exc
            return media_input_url
        return audio_path

    def create_diarization_job(
        self, media_url: str, webhook_url: str | None = None
    ) -> str:
        """Create a diarization job for the given media:// or http(s) URL and return the job id."""
        jobs_endpoint = "https://api.pyannote.ai/v1/diarize"
        job_body: dict[str, str] = {"url": media_url}
        if self.model:
            job_body["model"] = self.model
        if webhook_url:
            job_body["webhook"] = webhook_url
        response = requests.post(
            jobs_endpoint, json=job_body, headers=self._auth_headers_json()
        )
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("jobId") or job_data.get("id")
        if not job_id:
            raise Exception(f"Unexpected job creation response: {job_data}")
        return job_id

    def poll_diarization_results(
        self,
        job_id: str,
        *,
        initial_delay: float = 2.0,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
    ) -> SrtScript:
        """Poll a diarization job until completion and return an ``SrtScript`` with segments."""
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        jobs_endpoint = "https://api.pyannote.ai/v1/jobs"
        poll_endpoint = f"{jobs_endpoint}/{job_id}"
        deadline = time.time() + timeout

        if initial_delay > 0:
            time.sleep(initial_delay)

        while True:
            status_resp = requests.get(
                poll_endpoint,
                headers={"Authorization": f"Bearer {self.api_token}"},
            )

            if status_resp.status_code == 404:
                if time.time() >= deadline:
                    raise TimeoutError(f"Job not registered before timeout: {job_id}")
                time.sleep(min(2.0, poll_interval))
                continue

            if status_resp.status_code == 429:
                if time.time() >= deadline:
                    raise TimeoutError("Polling rate-limited until timeout expired")
                time.sleep(5.0)
                continue

            if status_resp.status_code != 200:
                raise Exception(
                    f"Failed to get job status: {status_resp.status_code} {status_resp.text}"
                )

            status_data = status_resp.json()
            status = status_data.get("status")
            if status == "succeeded":
                output = status_data.get("output") or {}
                diar = output.get("diarization") or []
                segments = diar if isinstance(diar, list) else []
                try:
                    segments.sort(key=lambda seg: float(seg.get("start") or 0.0))
                except Exception:
                    pass

                for segment in segments:
                    start_sec = float(segment.get("start") or 0.0)
                    end_sec = float(segment.get("end") or 0.0)
                    speaker = segment.get("speaker")

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

            if status == "failed":
                raise Exception(
                    f"Job failed: {status_data.get('error') or status_data}"
                )

            if time.time() >= deadline:
                raise TimeoutError("Diarization job polling timed out")
            time.sleep(poll_interval)

        self.srt = srt
        return srt

    def get_speaker_segments(
        self, audio_path: str, webhook_url: str | None = None
    ) -> SrtScript:
        logger.info("Processing audio file via API: %s", audio_path)
        media_url = self.upload_media(audio_path)
        job_id = self.create_diarization_job(media_url, webhook_url)
        return self.poll_diarization_results(job_id)


class LocalPyannoteVAD(VAD):
    """VAD provider backed by a locally hosted pyannote pipeline."""

    def __init__(
        self,
        model_name_or_path: str,
        src_lang: str,
        tgt_lang: str,
        min_segment_seconds: float = 1.0,
        *,
        hf_token: str | None = None,
        **pipeline_kwargs,
    ) -> None:
        super().__init__(src_lang, tgt_lang, min_segment_seconds)
        from pyannote.audio import Pipeline  # type: ignore

        self.pipeline = Pipeline.from_pretrained(
            model_name_or_path,
            use_auth_token=hf_token or os.getenv("HF_TOKEN"),
            **pipeline_kwargs,
        )

    def get_speaker_segments(
        self, audio_path: str, webhook_url: str | None = None
    ) -> SrtScript:  # noqa: ARG002
        logger.info("Processing audio file locally: %s", audio_path)
        diarization = self.pipeline(audio_path)
        srt = SrtScript(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (turn.end - turn.start) < self.min_segment_seconds:
                continue
            srt.segments.append(
                SrtSegment(
                    src_lang=self.src_lang,
                    tgt_lang=self.tgt_lang,
                    src_text="",
                    translation="",
                    speaker=speaker,
                    start_time=float(turn.start),
                    end_time=float(turn.end),
                    idx=len(srt.segments),
                )
            )
        self.srt = srt
        return srt


def create_vad(
    *,
    model_name_or_path: str,
    src_lang: str,
    tgt_lang: str,
    min_segment_seconds: float = 1.0,
    **provider_kwargs,
) -> VAD:
    """Factory helper returning the appropriate VAD provider implementation."""
    if model_name_or_path.strip().lower() == "api":
        return APIPyannoteVAD(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            min_segment_seconds=min_segment_seconds,
            **provider_kwargs,
        )

    return LocalPyannoteVAD(
        model_name_or_path=model_name_or_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        min_segment_seconds=min_segment_seconds,
        **provider_kwargs,
    )


__all__ = [
    "APIPyannoteVAD",
    "LocalPyannoteVAD",
    "create_vad",
]
