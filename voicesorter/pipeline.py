"""Pipeline stages: each one persists its result so a run can resume."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import audio, classify, cluster, embed, export
from .project import atomic_write_json, load_json

SCAN_FILE = "scan.json"
CLUSTERS_FILE = "clusters.json"
RESULT_FILE = "result.npz"
PUBLIC_RESULT = "result.json"
BLOCKLIST = "blocklist.json"


def _p(project, name: str) -> str:
    return os.path.join(project.cache_dir, name)


def stage_scan(project, progress=None, force: bool = False) -> dict:
    existing = None if force else load_json(_p(project, SCAN_FILE))
    if existing and project.is_at_least("embedded"):
        if progress:
            progress("复用已扫描结果")
        return existing
    if progress:
        progress("scanning files")
    res = audio.scan(project.input_dir, project.params.recursive, project.params.dedupe)
    atomic_write_json(_p(project, SCAN_FILE), res)
    project.stats["files"] = len(res["files"])
    project.stats["duplicates"] = len(res["duplicates"])
    project.stats["unreadable"] = len(res["errors"])
    project.save()
    return res


def stage_embed(project, progress=None, force: bool = False) -> dict:
    if not force and project.is_at_least("embedded"):
        embs, items = embed.load_cache(project.cache_dir)
        if embs is not None:
            if progress:
                progress(f"复用已有声纹 {len(embs)} 条")
            return {"embeddings": embs, "items": items,
                    "failed": project.stats.get("failed_files", [])}

    scan = load_json(_p(project, SCAN_FILE))
    if not scan:
        raise RuntimeError("还没有扫描结果：请先在第②步点「开始处理」")
    paths = [f["path"] for f in scan["files"]]

    def tick(done, total, eta):
        if progress:
            progress(done, total, f"剩余约 {int(eta)} 秒")

    mp = embed.ensure_model(os.path.join(os.path.dirname(__file__), "..", "models"),
                            progress=progress)
    res = embed.extract(paths, mp, workers=project.params.workers,
                        threads=project.params.worker_threads,
                        throttle_ms=project.params.throttle_ms,
                        min_duration=project.params.min_duration, progress=tick)
    embed.save_cache(project.cache_dir, res, res["items"])
    project.stats["embedded"] = len(res["embeddings"])
    project.stats["filtered"] = len(res.get("filtered", []))
    project.stats["failed_files"] = res["failed"]
    project.advance("embedded")
    return res


def load_embeddings(project):
    return embed.load_cache(project.cache_dir)


def stage_cluster(project, sample_size: int = 0, n_clusters: int = 0, seed: int = 0,
                  progress=None) -> dict:
    embs, items = load_embeddings(project)
    if embs is None:
        raise RuntimeError("还没有声纹数据：请先在第②步点「开始处理」")
    size = sample_size or project.params.sample_size
    idx = cluster.pick_sample(len(items), size, seed)
    sub = embs[idx]
    k = n_clusters or project.params.n_clusters or cluster.estimate_k(sub)
    labels = cluster.run(sub, k)
    summary = cluster.export_samples(project, items, idx, labels,
                                     per_cluster=project.params.samples_per_cluster or 20)
    data = {
        "sample_size": len(idx),
        "k": int(k),
        "sample_indices": idx,
        "labels": [int(x) for x in labels],
        **summary,
    }
    atomic_write_json(_p(project, CLUSTERS_FILE), data)
    project.advance("clustered")
    return data


def stage_build_library(project, assignment: dict[int, str | None]) -> dict:
    """``assignment`` maps cluster id -> character name (None/"" to discard)."""
    embs, items = load_embeddings(project)
    clusters = load_json(_p(project, CLUSTERS_FILE))
    if clusters is None:
        raise RuntimeError("还没有聚类结果：请先在第②步点「开始处理」")
    clean = {int(k): (v or None) for k, v in assignment.items()}
    lib = classify.build_library(embs, items, clusters, clean)
    classify.save_library(project, lib)
    project.stats["characters"] = [c["name"] for c in lib["characters"]]
    project.advance("named")
    return lib


def _apply_quality(project, items, labels) -> int:
    """Force low-quality clips to "other": too short, too long, too loud, clipped."""
    p = project.params
    changed = 0
    for i, it in enumerate(items):
        d = float(it.get("dur") or 0)
        peak = float(it.get("peak") or 0)
        rms = float(it.get("rms") or 0)
        drop = False
        if d and d < p.min_duration:
            drop = True
        elif p.max_duration and d > p.max_duration:
            drop = True
        elif p.max_rms and rms > p.max_rms:
            drop = True
        elif peak >= p.max_peak:
            drop = True
        if drop:
            it["hidden"] = True
            if labels[i] != "其他":
                labels[i] = "其他"
                changed += 1
    return changed


def stage_classify(project, progress=None) -> dict:
    embs, items = load_embeddings(project)
    lib = classify.load_library(project)
    if embs is None or lib is None:
        raise RuntimeError("还没有声纹或角色库：请先在第②步点「开始处理」，并建立或导入角色库")
    blocked = set(load_json(_p(project, BLOCKLIST), []) or [])
    res = classify.classify(embs, lib, project.params.threshold,
                            project.params.margin, blocked)
    _apply_quality(project, items, res["labels"])
    np.savez_compressed(_p(project, RESULT_FILE),
                        scores=res["scores"], margins=res["margins"])
    counts: dict[str, int] = {}
    for lab in res["labels"]:
        counts[lab] = counts.get(lab, 0) + 1
    atomic_write_json(_p(project, PUBLIC_RESULT),
                      {"counts": counts, "names": res["names"],
                       "threshold": project.params.threshold,
                       "margin": project.params.margin})
    project.stats["classified"] = counts
    project.stats["characters"] = res["names"]
    project.advance("classified")
    return {"labels": res["labels"], "counts": counts}


def get_result(project):
    """Rebuild labels from cache (cheap: one matmul)."""
    embs, items = load_embeddings(project)
    lib = classify.load_library(project)
    if embs is None or lib is None:
        return None, None, None
    blocked = set(load_json(_p(project, BLOCKLIST), []) or [])
    res = classify.classify(embs, lib, project.params.threshold,
                            project.params.margin, blocked)
    _apply_quality(project, items, res["labels"])
    return items, res, embs


def stage_export(project, progress=None) -> dict:
    items, res, _embs = get_result(project)
    if items is None:
        raise RuntimeError("还没有分类结果：请先在第③步点「开始分类」")
    stats = export.run(project, items, res["labels"], res["scores"], res["margins"],
                       progress=progress)
    project.stats["export"] = {"counts": stats["counts"], "others": stats["others"]}
    project.advance("exported")
    return stats


def add_blocklist(project, indices: list[int]) -> None:
    path = _p(project, BLOCKLIST)
    cur = set(load_json(path, []) or [])
    cur.update(int(i) for i in indices)
    atomic_write_json(path, sorted(cur))


def cascade_neighbours(project, index: int, label: str, threshold: float | None = None,
                       limit: int = 30) -> list[dict]:
    """Clips in the same predicted class that are very close to *index*."""
    items, res, embs = get_result(project)
    if items is None or index >= len(items):
        return []
    th = threshold if threshold is not None else project.params.cascade
    sims = embs @ embs[index]
    cands = []
    for j, lab in enumerate(res["labels"]):
        if j == index or lab != label:
            continue
        if sims[j] >= th:
            cands.append({"index": j, "name": items[j]["name"], "sim": round(float(sims[j]), 4),
                          "dur": items[j]["dur"], "path": items[j]["path"]})
    cands.sort(key=lambda c: -c["sim"])
    return cands[:limit]


def state(project) -> dict:
    return {
        "stage": project.stage,
        "stats": project.stats,
        "params": project.params.__dict__,
        "input_dir": project.input_dir,
        "output_dir": project.output_dir,
        "name": project.name,
    }
