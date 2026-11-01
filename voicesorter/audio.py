"""Audio discovery, dedupe and decoding (PyAV based, no external ffmpeg)."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable

import av
import numpy as np
import soundfile as sf

from .project import AUDIO_EXT


def list_audio(root: str, recursive: bool = True) -> list[str]:
    """Return audio file paths under *root*, sorted for stable ordering."""
    out: list[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in AUDIO_EXT:
                    out.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in AUDIO_EXT:
                out.append(p)
    out.sort()
    return out


def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan(root: str, recursive: bool = True, dedupe: bool = True) -> dict:
    """List audio files, optionally collapsing byte-identical duplicates.

    Returns ``{"files": [{path,name,size,md5}], "duplicates": [...], "errors": [...]}``.
    """
    files, duplicates, errors = [], [], []
    seen: dict[str, str] = {}
    for p in list_audio(root, recursive):
        try:
            size = os.path.getsize(p)
            digest = file_md5(p) if dedupe else ""
        except OSError as e:  # unreadable file
            errors.append({"path": p, "error": str(e)})
            continue
        item = {"path": p, "name": os.path.basename(p), "size": size, "md5": digest}
        if dedupe and digest in seen:
            duplicates.append({**item, "same_as": seen[digest]})
            continue
        if dedupe:
            seen[digest] = p
        files.append(item)
    return {"files": files, "duplicates": duplicates, "errors": errors}


def decode(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode any supported file to mono float32. Returns ``(samples, sample_rate)``."""
    container = av.open(path)
    try:
        stream = container.streams.audio[0]
        sr = target_sr or stream.rate or 16000
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray())
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray())
    finally:
        container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32), sr
    wav = np.concatenate(chunks, axis=-1).reshape(-1).astype(np.float32)
    return np.ascontiguousarray(wav), sr


def write_wav(path: str, wav: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr, subtype=subtype)


def duration(path: str) -> float:
    """Best-effort duration in seconds (decodes when the container lies)."""
    try:
        container = av.open(path)
        try:
            d = float(container.duration or 0) / 1_000_000
            if d > 0:
                return d
        finally:
            container.close()
    except Exception:
        pass
    wav, sr = decode(path, target_sr=16000)
    return len(wav) / sr if sr else 0.0


def peak_normalize(wav: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak <= ceiling or peak == 0.0:
        return wav
    return (wav * (ceiling / peak)).astype(np.float32)


def iter_in_batches(items: Iterable, n: int):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
