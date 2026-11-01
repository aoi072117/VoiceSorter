"""Export sorted clips to per-character folders plus a manifest."""

from __future__ import annotations

import csv
import os
import shutil

from . import audio

MANIFEST = "manifest.csv"


def _target_path(project, label: str, name: str, ext: str = ".wav") -> str:
    base = os.path.splitext(os.path.basename(name))[0] + ext
    return os.path.join(project.output_dir, label, base)


def estimate_bytes(items: list[dict], labels: list[str], out_format: str = "wav") -> int:
    """Rough output size.

    WAV keeps the source sample rate, so the size depends entirely on the
    library being processed -- there is no fixed number.
    """
    total = 0
    for it, label in zip(items, labels):
        if label == "其他":
            continue
        if out_format == "copy":
            total += int(it.get("size") or 0)
        else:
            sr = int(it.get("sr") or 32000)
            total += int(float(it.get("dur") or 0) * sr * 2)
    return total


def run(project, items: list[dict], labels: list[str], scores, margins,
        progress=None) -> dict:
    params = project.params
    out_root = project.output_dir
    os.makedirs(out_root, exist_ok=True)

    todo = [(i, lab) for i, lab in enumerate(labels) if lab != "其他"]
    counts: dict[str, int] = {}
    rows = []
    failed = []

    for n, (i, label) in enumerate(todo):
        it = items[i]
        name = it.get("name") or os.path.basename(it["path"])
        copy_mode = params.export_format == "copy"
        ext = os.path.splitext(it["path"])[1] if copy_mode else ".wav"
        dst = _target_path(project, label, name, ext)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if copy_mode:
                shutil.copy2(it["path"], dst)
            else:
                wav, sr = audio.decode(it["path"])
                if params.normalize_peak:
                    wav = audio.peak_normalize(wav)
                audio.write_wav(dst, wav, sr)
            counts[label] = counts.get(label, 0) + 1
            rows.append([label, it["name"], round(float(scores[i]), 4),
                         round(float(margins[i]), 4), it.get("dur", 0), dst])
        except Exception as e:  # noqa: BLE001
            failed.append({"path": it["path"], "error": str(e)})
        step = max(25, len(todo) // 100)
        if progress and (n + 1) % step == 0:
            progress(n + 1, len(todo))

    rows.sort(key=lambda r: (r[0], r[1]))
    with open(os.path.join(out_root, MANIFEST), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["角色", "原文件名", "相似度", "与第二名差", "时长秒", "输出路径"])
        w.writerows(rows)

    others = sum(1 for lab in labels if lab == "其他")
    return {"counts": counts, "exported": sum(counts.values()),
            "others": others, "failed": failed, "manifest": os.path.join(out_root, MANIFEST)}


def remove_others(project) -> int:
    """Delete the '其他' folder from the output dir. Returns bytes freed."""
    folder = os.path.join(project.output_dir, "其他")
    if not os.path.isdir(folder):
        return 0
    size = 0
    for dirpath, _d, files in os.walk(folder):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    shutil.rmtree(folder, ignore_errors=True)
    return size


def clean_cache(project, keep_embeddings: bool = True) -> int:
    """Remove intermediate files. Returns bytes freed."""
    freed = 0
    targets = []
    if os.path.isdir(project.samples_dir):
        targets.append(project.samples_dir)
    if os.path.isdir(project.review_dir):
        targets.append(project.review_dir)
    if os.path.isdir(project.cache_dir):
        for f in os.listdir(project.cache_dir):
            if keep_embeddings and f.startswith("embeddings"):
                continue
            targets.append(os.path.join(project.cache_dir, f))
    for t in targets:
        if os.path.isdir(t):
            for dirpath, _d, files in os.walk(t):
                for f in files:
                    try:
                        freed += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            shutil.rmtree(t, ignore_errors=True)
        elif os.path.isfile(t):
            try:
                freed += os.path.getsize(t)
            except OSError:
                pass
            os.remove(t)
    return freed
