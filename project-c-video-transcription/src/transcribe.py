"""Extract audio with ffmpeg; transcribe via OpenAI, local faster-whisper, or Deepgram."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import (
    deepgram_api_key,
    deepgram_model,
    local_whisper_compute_type,
    local_whisper_device,
    local_whisper_model,
    openai_api_key,
    transcribe_segment_seconds,
    transcription_backend,
    whisper_model,
)

# OpenAI Whisper HTTP API file size limit is 25 MiB; stay under with margin.
_MAX_BYTES_PER_REQUEST = 20 * 1024 * 1024


class TranscriptionError(Exception):
    pass


def _ensure_speaker_fields(segments: list[dict]) -> list[dict]:
    """Whisper paths have no diarization; keep API shape consistent."""
    out: list[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        d = dict(seg)
        if "speaker" not in d:
            d["speaker"] = None
        if "speaker_label" not in d:
            d["speaker_label"] = None
        out.append(d)
    return out


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise TranscriptionError("ffmpeg and ffprobe must be installed and on PATH")


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()[:2000]
        raise TranscriptionError(f"Command failed ({p.returncode}): {' '.join(cmd[:4])}… — {err}")


def _ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise TranscriptionError(f"ffprobe failed: {(p.stderr or p.stdout)[:500]}")
    try:
        return float((p.stdout or "").strip())
    except ValueError as e:
        raise TranscriptionError("Could not parse media duration") from e


def _extract_audio_mp3(video_path: Path, out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(out_mp3),
    ]
    _run(cmd)


def _segment_audio_mp3(src: Path, out_dir: Path, segment_seconds: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        pattern,
    ]
    _run(cmd)
    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise TranscriptionError("Audio segmentation produced no chunks")
    return chunks


def _transcribe_file_openai(client: object, model: str, audio_path: Path) -> dict:
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
        )
    segments: list[dict] = []
    raw_segments = getattr(result, "segments", None)
    if raw_segments:
        for s in raw_segments:
            if hasattr(s, "model_dump"):
                segments.append(s.model_dump())
            elif isinstance(s, dict):
                segments.append(s)
    lang = getattr(result, "language", None)
    text = (getattr(result, "text", None) or "").strip()
    return {"text": text, "segments": segments, "language": lang}


def _normalize_local_segments(segments: list) -> list[dict]:
    out: list[dict] = []
    for i, s in enumerate(segments):
        if hasattr(s, "start") and hasattr(s, "end") and hasattr(s, "text"):
            out.append(
                {
                    "id": i,
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": (s.text or "").strip(),
                    "speaker": None,
                    "speaker_label": None,
                }
            )
        elif isinstance(s, dict):
            d = dict(s)
            d.setdefault("id", i)
            d.setdefault("speaker", None)
            d.setdefault("speaker_label", None)
            out.append(d)
    return out


def _transcribe_file_local(model: object, audio_path: Path) -> dict:
    """model: faster_whisper.WhisperModel instance."""
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )
    segs = list(segments_iter)
    text = "".join(getattr(s, "text", "") or "" for s in segs).strip()
    segments = _normalize_local_segments(segs)
    lang = getattr(info, "language", None)
    return {"text": text, "segments": segments, "language": lang}


def _load_faster_whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise TranscriptionError(
            "Local transcription needs faster-whisper. Run: pip install -r requirements-local.txt"
        ) from e
    return WhisperModel(
        local_whisper_model(),
        device=local_whisper_device(),
        compute_type=local_whisper_compute_type(),
    )


def _merge_chunk_results(
    chunk_results: list[dict],
    chunk_durations: list[float],
    model_label: str,
) -> dict:
    combined_text: list[str] = []
    combined_segments: list[dict] = []
    language: str | None = None
    cumulative_offset = 0.0
    next_id = 0

    for data, dur in zip(chunk_results, chunk_durations, strict=True):
        if language is None:
            language = data.get("language")
        piece = (data.get("text") or "").strip()
        if piece:
            combined_text.append(piece)
        for seg in data.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            adj = dict(seg)
            adj["id"] = next_id
            next_id += 1
            try:
                if "start" in adj:
                    adj["start"] = float(adj["start"]) + cumulative_offset
                if "end" in adj:
                    adj["end"] = float(adj["end"]) + cumulative_offset
            except (TypeError, ValueError):
                pass
            adj.setdefault("speaker", None)
            adj.setdefault("speaker_label", None)
            combined_segments.append(adj)
        cumulative_offset += dur

    return {
        "text": "\n".join(combined_text) if combined_text else "",
        "segments": combined_segments,
        "language": language,
        "model": model_label,
    }


def _transcribe_openai_path(video_path: Path) -> dict:
    from openai import OpenAI

    key = openai_api_key()
    if not key:
        raise TranscriptionError(
            "OPENAI_API_KEY is not set. Use TRANSCRIPTION_BACKEND=local or set the key."
        )

    client = OpenAI(api_key=key)
    model = whisper_model()
    seg_seconds = transcribe_segment_seconds()

    with tempfile.TemporaryDirectory(prefix="vtx_") as tmp:
        tmp_path = Path(tmp)
        full_mp3 = tmp_path / "full.mp3"
        _extract_audio_mp3(video_path, full_mp3)
        size = full_mp3.stat().st_size

        if size <= _MAX_BYTES_PER_REQUEST:
            data = _transcribe_file_openai(client, model, full_mp3)
            return {
                "text": (data.get("text") or "").strip(),
                "segments": _ensure_speaker_fields(data.get("segments") or []),
                "language": data.get("language"),
                "model": model,
            }

        chunks_dir = tmp_path / "chunks"
        chunk_results: list[dict] = []
        durations: list[float] = []
        for chunk_path in _segment_audio_mp3(full_mp3, chunks_dir, seg_seconds):
            durations.append(_ffprobe_duration_seconds(chunk_path))
            chunk_results.append(_transcribe_file_openai(client, model, chunk_path))

        return _merge_chunk_results(chunk_results, durations, model)


def _transcribe_local_path(video_path: Path) -> dict:
    fw_model = _load_faster_whisper()
    model_label = f"local:{local_whisper_model()}"
    seg_seconds = transcribe_segment_seconds()

    with tempfile.TemporaryDirectory(prefix="vtx_") as tmp:
        tmp_path = Path(tmp)
        full_mp3 = tmp_path / "full.mp3"
        _extract_audio_mp3(video_path, full_mp3)
        duration = _ffprobe_duration_seconds(full_mp3)
        size = full_mp3.stat().st_size

        # Long or heavy files: chunk to limit memory and match API-style pipeline.
        need_chunks = duration > float(seg_seconds) or size > _MAX_BYTES_PER_REQUEST

        if not need_chunks:
            data = _transcribe_file_local(fw_model, full_mp3)
            return {
                "text": data.get("text") or "",
                "segments": _ensure_speaker_fields(data.get("segments") or []),
                "language": data.get("language"),
                "model": model_label,
            }

        chunks_dir = tmp_path / "chunks"
        chunk_results: list[dict] = []
        durations: list[float] = []
        for chunk_path in _segment_audio_mp3(full_mp3, chunks_dir, seg_seconds):
            durations.append(_ffprobe_duration_seconds(chunk_path))
            chunk_results.append(_transcribe_file_local(fw_model, chunk_path))

        return _merge_chunk_results(chunk_results, durations, model_label)


def _parse_deepgram_response(payload: dict) -> dict:
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    language: str | None = None
    if channels and isinstance(channels[0], dict):
        language = channels[0].get("detected_language")

    utterances = results.get("utterances")
    model_label = f"deepgram:{deepgram_model()}"
    segments: list[dict] = []

    if utterances:
        i = 0
        for u in utterances:
            if not isinstance(u, dict):
                continue
            sp = u.get("speaker")
            txt = (u.get("transcript") or "").strip()
            try:
                start = float(u["start"])
                end = float(u["end"])
            except (KeyError, TypeError, ValueError):
                continue
            sp_int: int | None
            if sp is None:
                sp_int = None
            else:
                try:
                    sp_int = int(sp)
                except (TypeError, ValueError):
                    sp_int = None
            label = f"Speaker {sp_int + 1}" if sp_int is not None else None
            segments.append(
                {
                    "id": i,
                    "start": start,
                    "end": end,
                    "text": txt,
                    "speaker": sp_int,
                    "speaker_label": label,
                    "confidence": u.get("confidence"),
                }
            )
            i += 1

    if not segments and channels:
        alt0 = (channels[0].get("alternatives") or [{}])[0]
        if isinstance(alt0, dict):
            txt = (alt0.get("transcript") or "").strip()
            if txt:
                segments = [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 0.0,
                        "text": txt,
                        "speaker": None,
                        "speaker_label": None,
                    }
                ]

    texts = [s["text"] for s in segments if s.get("text")]
    full_text = " ".join(texts)

    return {
        "text": full_text,
        "segments": segments,
        "language": language,
        "model": model_label,
    }


def _deepgram_listen_mp3(mp3_path: Path) -> dict:
    key = deepgram_api_key()
    if not key:
        raise TranscriptionError(
            "DEEPGRAM_API_KEY is not set. Set it or use another TRANSCRIPTION_BACKEND."
        )
    params = {
        "model": deepgram_model(),
        "diarize": "true",
        "utterances": "true",
        "punctuate": "true",
        "smart_format": "true",
        "detect_language": "true",
    }
    url = "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)
    body = mp3_path.read_bytes()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": "audio/mpeg",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")[:4000]
        raise TranscriptionError(f"Deepgram HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise TranscriptionError(f"Deepgram request failed: {e}") from e

    return _parse_deepgram_response(payload)


def _transcribe_deepgram_path(video_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="vtx_") as tmp:
        full_mp3 = Path(tmp) / "full.mp3"
        _extract_audio_mp3(video_path, full_mp3)
        return _deepgram_listen_mp3(full_mp3)


def transcribe_video_path(video_path: Path) -> dict:
    """
    Return dict with keys: text, segments (list), language (optional), model.
    """
    _require_ffmpeg()

    if not video_path.is_file():
        raise TranscriptionError(f"Video file not found: {video_path}")

    backend = transcription_backend()
    if backend == "local":
        return _transcribe_local_path(video_path)
    if backend == "deepgram":
        return _transcribe_deepgram_path(video_path)
    return _transcribe_openai_path(video_path)
