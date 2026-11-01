"""Project state and configuration.

A project is a directory holding:

    project.json      config + pipeline stage
    characters.json   speaker library (name -> centroid + refs)
    cache/            embeddings and scan metadata
    samples/          cluster preview clips
    review/           working copies for the review stage
    out/              exported result
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

STAGES = ["created", "embedded", "clustered", "named", "classified", "reviewed", "exported"]

AUDIO_EXT = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".aiff", ".aif"}


@dataclass
class Params:
    sample_size: int = 500
    n_clusters: int = 0            # 0 = auto estimate
    threshold: float = 0.65
    margin: float = 0.03
    cascade: float = 0.78
    workers: int = 4               # parallel processes
    worker_threads: int = 2        # onnxruntime threads per worker
    throttle_ms: int = 0           # sleep between batches, for cooler CPUs
    samples_per_cluster: int = 20  # preview clips shown per cluster
    recursive: bool = True
    dedupe: bool = True
    export_format: str = "wav"     # wav | copy
    normalize_peak: bool = False
    min_duration: float = 1.0      # shorter clips are 语气词, useless for cloning
    max_duration: float = 0.0      # 0 = unlimited
    max_rms: float = 0.0           # 0 = off; e.g. 0.25 drops screams / shouting
    max_peak: float = 0.995        # clipping guard


@dataclass
class Project:
    path: str
    name: str = "untitled"
    input_dir: str = ""
    output_dir: str = ""
    stage: str = "created"
    params: Params = field(default_factory=Params)
    stats: dict[str, Any] = field(default_factory=dict)

    # ---------- persistence ----------

    @property
    def json_path(self) -> str:
        return os.path.join(self.path, "project.json")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.path, "cache")

    @property
    def samples_dir(self) -> str:
        return os.path.join(self.path, "samples")

    @property
    def review_dir(self) -> str:
        return os.path.join(self.path, "review")

    @property
    def characters_path(self) -> str:
        return os.path.join(self.path, "characters.json")

    def ensure_dirs(self) -> None:
        for d in (self.path, self.cache_dir, self.samples_dir, self.review_dir):
            os.makedirs(d, exist_ok=True)

    def save(self) -> None:
        self.ensure_dirs()
        data = asdict(self)
        atomic_write_json(self.json_path, data)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(os.path.join(path, "project.json"), encoding="utf-8") as f:
            data = json.load(f)
        params = Params(**data.pop("params", {}))
        p = cls(path=path, params=params, **{k: v for k, v in data.items() if k != "path"})
        return p

    @classmethod
    def create(cls, path: str, input_dir: str, output_dir: str, name: str = "") -> "Project":
        p = cls(path=os.path.abspath(path), input_dir=os.path.abspath(input_dir),
                output_dir=os.path.abspath(output_dir),
                name=name or os.path.basename(os.path.abspath(path)))
        p.ensure_dirs()
        p.save()
        return p

    # ---------- stage helpers ----------

    def stage_index(self) -> int:
        return STAGES.index(self.stage) if self.stage in STAGES else 0

    def advance(self, stage: str) -> None:
        self.stage = stage
        self.save()

    def is_at_least(self, stage: str) -> bool:
        return self.stage_index() >= STAGES.index(stage)


def atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
