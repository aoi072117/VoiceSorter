"""Character library and whole-collection classification."""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

CHARACTERS_FILE = "characters.json"
MODEL_TAG = "campplus_zh_16k"
MODEL_DIM = 192


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def build_library(embs: np.ndarray, items: list[dict], clusters: dict,
                  assignment: dict[int, str | None]) -> dict:
    """Turn "cluster id -> character name" into centroids.

    ``assignment`` maps cluster id to a character name, or None to discard.
    Returns ``{"characters": [{name, size, centroid}]}``.
    """
    sample_idx = clusters["sample_indices"]
    labels = np.asarray(clusters["labels"])
    groups: dict[str, list[int]] = {}
    for pos, original in enumerate(sample_idx):
        cid = int(labels[pos])
        name = assignment.get(cid)
        if not name:
            continue
        groups.setdefault(name, []).append(original)

    out = []
    for name, idxs in groups.items():
        centroid = normalize(embs[idxs].mean(axis=0))
        out.append({
            "name": name,
            "size": len(idxs),
            "centroid": centroid.astype(np.float32).tolist(),
            "clusters": [cid for cid, n in assignment.items() if n == name],
        })
    out.sort(key=lambda c: -c["size"])
    return {"characters": out, "other_label": "其他",
            "model": "campplus_zh_16k", "dim": int(embs.shape[1]),
            "format": 1, "created_at": datetime.now().isoformat(timespec="seconds")}


def save_library(project, lib: dict) -> None:
    with open(project.characters_path, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=1)


def load_library(project) -> dict | None:
    if not os.path.exists(project.characters_path):
        return None
    with open(project.characters_path, encoding="utf-8-sig") as f:
        return json.load(f)


def classify(embs: np.ndarray, lib: dict, threshold: float, margin: float,
             blocked: set[int] | None = None) -> dict:
    """Assign every embedding to a character or to "other".

    A clip is accepted only when its best cosine similarity beats *threshold*
    and beats the runner-up by *margin*.
    """
    chars = lib["characters"]
    if not chars:
        return {"labels": ["其他"] * len(embs), "scores": np.zeros(len(embs)),
                "margins": np.zeros(len(embs)), "names": []}

    C = np.vstack([np.asarray(c["centroid"], dtype=np.float32) for c in chars])
    S = embs @ C.T
    best_i = S.argmax(axis=1)
    best = S[np.arange(len(embs)), best_i]
    if S.shape[1] > 1:
        srt = np.sort(S, axis=1)
        marg = srt[:, -1] - srt[:, -2]
    else:
        marg = np.ones(len(embs), dtype=np.float32)

    labels = []
    for i in range(len(embs)):
        ok = best[i] >= threshold and marg[i] >= margin
        if blocked and i in blocked:
            ok = False
        labels.append(chars[int(best_i[i])]["name"] if ok else "其他")

    return {
        "labels": labels,
        "scores": best.astype(np.float32),
        "margins": marg.astype(np.float32),
        "names": [c["name"] for c in chars],
    }


def suggest_threshold(embs: np.ndarray, lib: dict, overrides: dict[int, str],
                      kinds: dict[int, str] | None = None) -> dict:
    """Suggest threshold/margin from the clips the user already judged.

    Confirmed clips (assigned to a character) are positives; clips the user
    pushed to "other" are negatives. The suggestion sits between the two
    distributions.
    """
    chars = lib["characters"]
    if not chars or not overrides:
        return {"ok": False, "reason": "先在校对页标记一些对错，再让它学"}

    C = np.vstack([np.asarray(c["centroid"], dtype=np.float32) for c in chars])
    S = embs @ C.T
    best = S.max(axis=1)
    if S.shape[1] > 1:
        srt = np.sort(S, axis=1)
        marg = srt[:, -1] - srt[:, -2]
    else:
        marg = np.ones(len(embs), dtype=np.float32)

    names = {c["name"] for c in chars}
    kinds = kinds or {}
    pos, neg = [], []
    for i, lab in overrides.items():
        if not (0 <= i < len(embs)):
            continue
        kind = kinds.get(i)
        if kind == "confirm":
            pos.append(i)
        elif kind == "fix":
            neg.append(i)
        elif kind is None:          # legacy marks: label decides
            (pos if lab in names else neg).append(i)
    if len(pos) < 3 or len(neg) < 3:
        return {"ok": False, "reason": f"样本还不够：确认 {len(pos)} 条、否定 {len(neg)} 条（各需 ≥3）", "pos": len(pos), "neg": len(neg)}

    ps, ns = best[pos], best[neg]
    pm, nm = marg[pos], marg[neg]
    p_pos = float(np.percentile(ps, 10))
    p_neg = float(np.percentile(ns, 90))
    threshold = float(np.clip((p_pos + p_neg) / 2, 0.5, 0.9))
    margin = float(np.clip((float(np.percentile(pm, 10)) + float(np.percentile(nm, 90))) / 4, 0.01, 0.10))
    return {
        "ok": True,
        "threshold": round(threshold, 3),
        "margin": round(margin, 3),
        "pos": len(pos), "neg": len(neg),
        "pos_p10": round(p_pos, 3), "neg_p90": round(p_neg, 3),
        "separable": p_pos > p_neg,
    }
