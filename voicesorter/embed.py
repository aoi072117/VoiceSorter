"""Speaker embeddings via sherpa-onnx (CAM++ 3D-Speaker model, 192-d)."""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np

from . import audio

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
MODEL_NAME = "campplus_zh_16k.onnx"
SAMPLE_RATE = 16000

_extractor = None


def model_path(models_dir: str) -> str:
    return os.path.join(models_dir, MODEL_NAME)


def ensure_model(models_dir: str, progress=None) -> str:
    """Download the CAM++ model once. Returns its path."""
    os.makedirs(models_dir, exist_ok=True)
    dst = model_path(models_dir)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        return dst
    tmp = dst + ".part"
    if progress:
        progress("downloading speaker model (~27 MB)")

    def hook(blocks, block_size, total):
        if progress and total > 0:
            done = min(blocks * block_size, total)
            progress(f"downloading model {done * 100 // total}%")

    urllib.request.urlretrieve(MODEL_URL, tmp, hook)
    os.replace(tmp, dst)
    return dst


def get_extractor(mp: str, threads: int = 4):
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=mp, num_threads=threads)
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def _init_worker(mp: str, threads: int) -> None:
    global _extractor
    _extractor = get_extractor(mp, threads)


def _entry(path: str, dur: float, error: str | None, peak: float = 0.0,
           rms: float = 0.0, sr: int = 0) -> dict:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return {"path": path, "name": os.path.basename(path), "size": size,
            "dur": round(dur, 3), "peak": round(peak, 4), "rms": round(rms, 4),
            "sr": int(sr),
            "error": error}


def _stats(wav: np.ndarray) -> tuple[float, float]:
    """Peak and RMS of a clip, used for the loudness filter."""
    if not len(wav):
        return 0.0, 0.0
    peak = float(np.abs(wav).max())
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    return peak, rms


def _work(path: str, min_duration: float = 0.0):
    """Worker: decode one file and return its L2-normalised embedding.

    Clips shorter than *min_duration* are rejected outright ("filtered") so
    they never reach clustering, review or export.
    """
    try:
        wav, _sr = audio.decode(path, target_sr=SAMPLE_RATE)
        dur = len(wav) / SAMPLE_RATE
        try:
            with av.open(path) as _c:
                native_sr = int(_c.streams.audio[0].rate or 0)
        except Exception:
            native_sr = 0
        peak, rms = _stats(wav)
        if min_duration and dur < min_duration:
            return {"path": path, "dur": dur, "emb": None, "error": None,
                    "filtered": True, "peak": peak, "rms": rms, "sr": native_sr}
        if dur < 0.05:
            return {"path": path, "dur": dur, "emb": None, "error": "too short",
                    "peak": peak, "rms": rms, "sr": native_sr}
        stream = _extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, wav)
        stream.input_finished()
        emb = np.asarray(_extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            return {"path": path, "dur": dur, "emb": None, "error": "empty embedding",
                    "peak": peak, "rms": rms, "sr": native_sr}
        return {"path": path, "dur": dur, "emb": emb / norm, "error": None,
                "peak": peak, "rms": rms, "sr": native_sr}
    except Exception as e:  # noqa: BLE001 - one bad file must not kill the run
        return {"path": path, "dur": 0.0, "emb": None, "error": f"{type(e).__name__}: {e}",
                "peak": 0.0, "rms": 0.0}


def _work_tuple(args):
    path, min_duration = args
    return _work(path, min_duration)


def extract(paths: list[str], mp: str, workers: int = 4, threads: int = 2,
            throttle_ms: int = 0, min_duration: float = 0.0,
            progress=None, on_file=None) -> dict:
    """Embed many files in parallel.

    ``throttle_ms`` sleeps between batches so laptops do not cook themselves.

    Returns ``{"embeddings": (N,192) float32, "items": [...], "failed": [...]}``.
    """
    import multiprocessing as mp_mod
    import time as _time

    embs, items, failed, filtered = [], [], [], []
    total = len(paths)
    t0 = _time.time()
    ctx = mp_mod.get_context("spawn")
    with ctx.Pool(processes=max(1, workers), initializer=_init_worker,
                  initargs=(mp, max(1, threads))) as pool:
        work = [(p, min_duration) for p in paths]
        for i, r in enumerate(pool.imap_unordered(_work_tuple, work, chunksize=16)):
            if r.get("filtered"):
                filtered.append({"path": r["path"], "dur": round(r["dur"], 3)})
            elif r["emb"] is not None:
                embs.append(r["emb"])
                items.append(_entry(r["path"], r["dur"], None, r.get("peak",0.0), r.get("rms",0.0), r.get("sr",0)))
            else:
                failed.append({"path": r["path"], "error": r["error"]})
                items.append(_entry(r["path"], r["dur"], r["error"], r.get("peak",0.0), r.get("rms",0.0), r.get("sr",0)))
            if throttle_ms and (i + 1) % 25 == 0:
                _time.sleep(throttle_ms / 1000.0)
            if progress and (i + 1) % 200 == 0:
                elapsed = _time.time() - t0
                rate = (i + 1) / elapsed if elapsed else 0
                eta = (total - i - 1) / rate if rate else 0
                progress(i + 1, total, round(eta, 1))
            if on_file:
                on_file(r)

    dim = len(embs[0]) if embs else 192
    arr = np.vstack(embs) if embs else np.zeros((0, dim), dtype=np.float32)
    return {"embeddings": arr, "items": items, "failed": failed, "filtered": filtered}


def embed_one(path: str, mp: str) -> np.ndarray | None:
    """Single-file embedding (used by the review UI for on-demand analysis)."""
    global _extractor
    if _extractor is None:
        _extractor = get_extractor(mp)
    r = _work(path)
    return r["emb"]


def save_cache(cache_dir: str, result: dict, items: list[dict]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "embeddings.npy"), result["embeddings"])
    with open(os.path.join(cache_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)


def load_cache(cache_dir: str):
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    meta_path = os.path.join(cache_dir, "items.json")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return None, None
    embs = np.load(emb_path)
    with open(meta_path, encoding="utf-8") as f:
        items = json.load(f)
    if len(items) != len(embs):
        return None, None
    return embs, items
