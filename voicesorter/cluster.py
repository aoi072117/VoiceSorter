"""Cold-start clustering: sample some clips, group them, let the user name groups."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from . import audio


def pick_sample(n_total: int, size: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    size = min(size, n_total)
    return sorted(rng.choice(n_total, size=size, replace=False).tolist())


def estimate_k(embs: np.ndarray, k_min: int = 3, k_max: int = 12) -> int:
    """Pick k by silhouette.

    When several k are within 2 silhouette points of the best, prefer the
    larger one: more (purer) clusters are friendlier to the naming step than
    one blob that has to be split by hand.
    """
    n = len(embs)
    if n < 8:
        return min(max(1, n // 2), 3)
    scores: dict[int, float] = {}
    for k in range(k_min, min(k_max, n // 10 + 1) + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embs)
            if len(set(labels)) < 2:
                continue
            scores[k] = silhouette_score(embs, labels, metric="cosine", sample_size=min(600, n))
        except Exception:
            continue
    if not scores:
        return min(3, n)
    best = max(scores.values())
    good = [k for k, s in scores.items() if s >= best - 0.02]
    return max(good) if good else max(scores, key=scores.get)


def run(embs: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(embs)))
    if k == 1:
        return np.zeros(len(embs), dtype=int)
    if k <= 12 and len(embs) > 20:
        return KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(embs)
    # many speakers: agglomerative on cosine distance
    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return model.fit_predict(embs)


def export_samples(project, items: list[dict], sample_idx: list[int],
                   labels: np.ndarray, per_cluster: int = 20) -> dict:
    """Write a few playable clips per cluster and return a cluster summary."""
    if os.path.isdir(project.samples_dir):
        shutil.rmtree(project.samples_dir)
    os.makedirs(project.samples_dir, exist_ok=True)

    clusters: dict[int, list[int]] = {}
    for pos, original in enumerate(sample_idx):
        clusters.setdefault(int(labels[pos]), []).append(original)

    summary = []
    rng = np.random.default_rng(1)
    for cid in sorted(clusters):
        members = clusters[cid]
        picks = rng.choice(members, size=min(per_cluster, len(members)), replace=False).tolist()
        folder = os.path.join(project.samples_dir, f"cluster_{cid:02d}")
        os.makedirs(folder, exist_ok=True)
        clips = []
        for i, original in enumerate(picks):
            src = items[original]["path"]
            try:
                wav, sr = audio.decode(src)
                name = f"{cid:02d}_{i:02d}_{os.path.splitext(os.path.basename(src))[0]}.wav"
                audio.write_wav(os.path.join(folder, name), wav, sr)
                clips.append({"name": name, "dur": round(len(wav) / sr, 2)})
            except Exception:
                continue
        summary.append({
            "id": cid,
            "size": len(members),
            "share": round(len(members) / max(1, len(sample_idx)), 4),
            "clips": clips,
            "member_indices": members[:200],
        })

    return {"clusters": summary}
