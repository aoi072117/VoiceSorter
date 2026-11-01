"""Local web server: one page drives the whole pipeline."""

from __future__ import annotations

import json
import os
import socket
import threading
import traceback
import uuid

from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import audio, classify, embed, export, pipeline
from .project import Params, Project, atomic_write_json, load_json

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
USER_DIR = os.path.join(os.path.expanduser("~"), ".voicesorter")
RECENT_FILE = os.path.join(USER_DIR, "projects.json")


def read_recent() -> list[dict]:
    data = load_json(RECENT_FILE, {}) or {}
    items = []
    for p in data.get("projects", []):
        path = p.get("path", "")
        items.append({**p, "exists": os.path.isfile(os.path.join(path, "project.json"))})
    items.sort(key=lambda x: x.get("opened_at", ""), reverse=True)
    return items


def touch_recent(path: str) -> None:
    os.makedirs(USER_DIR, exist_ok=True)
    items = [p for p in read_recent() if p.get("path") != path]
    try:
        proj = Project.load(path)
        name = proj.name or os.path.basename(path)
        stage = proj.stage
    except Exception:
        name, stage = os.path.basename(path), ""
    items.insert(0, {"path": path, "name": name, "stage": stage,
                     "opened_at": datetime.now().isoformat(timespec="seconds")})
    atomic_write_json(RECENT_FILE, {"projects": items[:30]})


def forget_recent(path: str) -> None:
    items = [p for p in read_recent() if p.get("path") != path]
    os.makedirs(USER_DIR, exist_ok=True)
    atomic_write_json(RECENT_FILE, {"projects": items})


# Request models live at module scope: FastAPI resolves string annotations
# (``from __future__ import annotations``) against module globals, so classes
# defined inside create_app() would silently degrade to query parameters.
class NewProject(BaseModel):
    path: str
    input_dir: str
    output_dir: str
    name: str = ""


class ClusterReq(BaseModel):
    sample_size: int = 0
    k: int = 0
    seed: int = 0


class BuildReq(BaseModel):
    assignment: dict[str, str] = {}


class ReviewAction(BaseModel):
    index: int
    action: str = ""          # assign | other | reset | block
    name: str = ""


class CascadeReq(BaseModel):
    index: int = 0
    label: str = ""
    threshold: float = 0.0


class ThresholdReq(BaseModel):
    threshold: float | None = None
    margin: float | None = None


class CleanReq(BaseModel):
    scopes: list[str] = []


class PickReq(BaseModel):
    kind: str = "dir"
    path: str = ""


class ImportBody(BaseModel):
    content: str = ""


class OpenReq(BaseModel):
    path: str = ""


class ParamsReq(BaseModel):
    workers: int | None = None
    min_duration: float | None = None
    max_rms: float | None = None
    worker_threads: int | None = None
    throttle_ms: int | None = None
    samples_per_cluster: int | None = None
    sample_size: int | None = None


class App:
    def __init__(self, project: Project | None = None):
        self.project = project
        self.jobs: dict[str, dict] = {}
        self.overrides: dict[int, str] = {}
        self.kinds: dict[int, str] = {}   # "confirm" | "fix"
        self.failed: list[dict] = []

    # ---------- jobs ----------

    def start_job(self, name: str, fn) -> str:
        jid = uuid.uuid4().hex[:8]
        self.jobs[jid] = {"id": jid, "name": name, "status": "running",
                          "message": "", "percent": 0, "result": None, "error": None}

        def progress(*args):
            job = self.jobs[jid]
            if not args:
                return
            if len(args) >= 2 and isinstance(args[1], (int, float)) and args[1]:
                done, total = float(args[0]), float(args[1])
                note = f" · {args[2]}" if len(args) > 2 and isinstance(args[2], str) else ""
                job["message"] = f"{int(done)}/{int(total)}{note}"
                job["percent"] = round(done / total * 100)
            else:
                job["message"] = str(args[0])

        def runner():
            try:
                self.jobs[jid]["result"] = fn(progress)
                self.jobs[jid]["status"] = "done"
            except Exception as e:  # noqa: BLE001
                self.jobs[jid]["status"] = "error"
                self.jobs[jid]["error"] = f"{type(e).__name__}: {e}"
                self.jobs[jid]["trace"] = traceback.format_exc()

        threading.Thread(target=runner, daemon=True).start()
        return jid

    # ---------- review overrides ----------

    def override_path(self) -> str:
        return os.path.join(self.project.cache_dir, "overrides.json")

    def load_overrides(self) -> None:
        data = load_json(self.override_path(), {}) or {}
        if "labels" in data:      # current format
            self.overrides = {int(k): v for k, v in data.get("labels", {}).items()}
            self.kinds = {int(k): v for k, v in data.get("kinds", {}).items()}
        else:                     # legacy: plain index -> label
            self.overrides = {int(k): v for k, v in data.items()}
            self.kinds = {k: ("confirm" if v != "其他" else "fix") for k, v in self.overrides.items()}

    def save_overrides(self) -> None:
        atomic_write_json(self.override_path(), {
            "labels": {str(k): v for k, v in self.overrides.items()},
            "kinds": {str(k): v for k, v in self.kinds.items()},
        })

    def effective_labels(self):
        items, res, embs = pipeline.get_result(self.project)
        if items is None:
            return None, None, None, None
        labels = list(res["labels"])
        for idx, lab in self.overrides.items():
            if 0 <= idx < len(labels):
                labels[idx] = lab
        return items, res, embs, labels


def create_app(project: Project | None = None) -> FastAPI:
    app = FastAPI(title="VoiceSorter")
    # 不再自动打开上次的项目：用户不知道项目存在哪，自动打开只会造成困惑
    st = App(project)
    if project:
        st.load_overrides()

    # ---------- helpers ----------

    def need_project() -> Project:
        if st.project is None:
            raise HTTPException(409, "还没有打开项目：请先在第①步选择目录并创建项目")
        return st.project

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/api/state")
    def state():
        if st.project is None:
            return {"open": False}
        p = st.project
        return {"open": True, "path": p.path, "name": p.name,
                "input_dir": p.input_dir, "output_dir": p.output_dir,
                **pipeline.state(p)}

    @app.post("/api/project")
    def new_project(req: NewProject):
        p = Project.create(req.path, req.input_dir, req.output_dir, req.name)
        st.project = p
        st.load_overrides()
        touch_recent(p.path)
        return {"ok": True, "path": p.path}

    @app.post("/api/open")
    def open_project(req: OpenReq):
        path = (req.path or "").strip()
        if not path or not os.path.isfile(os.path.join(path, "project.json")):
            raise HTTPException(400, "这个文件夹里没有 project.json")
        p = Project.load(path)
        st.project = p
        st.load_overrides()
        touch_recent(p.path)
        return {"ok": True, "stage": p.stage, "path": p.path}

    @app.get("/api/recent")
    def recent():
        cur = st.project.path if st.project else ""
        return {"current": cur, "projects": read_recent()}

    @app.post("/api/recent/remove")
    def recent_remove(req: OpenReq):
        forget_recent(req.path)
        return {"ok": True}

    @app.post("/api/project/delete")
    def project_delete(req: OpenReq):
        path = os.path.abspath(req.path)
        if not os.path.isfile(os.path.join(path, "project.json")):
            raise HTTPException(400, "不是项目目录")
        try:
            import shutil as _sh

            _sh.rmtree(path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"删除失败: {e}")
        forget_recent(path)
        if st.project and os.path.abspath(st.project.path) == path:
            st.project = None
            st.overrides = {}
        return {"ok": True}

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        j = st.jobs.get(jid)
        if not j:
            raise HTTPException(404)
        return j

    # ---------- stages ----------

    @app.post("/api/params")
    def set_params(req: ParamsReq):
        p = need_project()
        for field, value in req.model_dump(exclude_none=True).items():
            if hasattr(p.params, field):
                setattr(p.params, field, value)
        p.save()
        return {"ok": True, "params": p.params.__dict__}

    @app.post("/api/scan")
    def scan():
        p = need_project()
        jid = st.start_job("scan", lambda pr: pipeline.stage_scan(p, pr))
        return {"job": jid}

    @app.post("/api/embed")
    def do_embed():
        p = need_project()
        jid = st.start_job("embed", lambda pr: {k: v for k, v in pipeline.stage_embed(p, pr).items()
                                                if k != "embeddings"})
        return {"job": jid}

    @app.post("/api/cluster")
    def do_cluster(req: ClusterReq):
        p = need_project()

        def run(pr):
            data = pipeline.stage_cluster(p, req.sample_size, req.k, req.seed, pr)
            data.pop("labels", None)
            data.pop("sample_indices", None)
            return data

        return {"job": st.start_job("cluster", run)}

    @app.get("/api/clusters")
    def clusters():
        p = need_project()
        data = load_json(os.path.join(p.cache_dir, "clusters.json"))
        if data is None:
            raise HTTPException(404, "no clustering yet")
        for c in data["clusters"]:
            c.pop("member_indices", None)
        return data

    @app.get("/api/cluster-sample/{cid}/{name}")
    def cluster_sample(cid: int, name: str):
        p = need_project()
        path = os.path.join(p.samples_dir, f"cluster_{cid:02d}", name)
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/build")
    def build(req: BuildReq):
        p = need_project()
        assignment = {int(k): (v or None) for k, v in req.assignment.items()}
        lib = pipeline.stage_build_library(p, assignment)
        return {"ok": True, "characters": [{"name": c["name"], "size": c["size"]}
                                           for c in lib["characters"]]}

    @app.get("/api/characters")
    def characters():
        p = need_project()
        lib = classify.load_library(p)
        return lib or {"characters": []}

    @app.get("/api/characters/export")
    def characters_export():
        p = need_project()
        lib = classify.load_library(p)
        if not lib:
            raise HTTPException(404, "no library yet")
        import json as _json

        from fastapi.responses import Response

        body = _json.dumps(lib, ensure_ascii=False, indent=1)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="characters.json"'})

    @app.post("/api/characters/import")
    def characters_import(req: ImportBody):
        p = need_project()
        text = (req.content or "").strip().lstrip("\ufeff")
        if not text:
            raise HTTPException(400, "文件是空的")
        try:
            lib = json.loads(text)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"不是有效的 JSON: {e}")
        if not isinstance(lib, dict) or "characters" not in lib:
            raise HTTPException(400, "缺少 characters 字段")
        expect = classify.MODEL_TAG
        got = lib.get("model")
        if got and got != expect:
            raise HTTPException(400, f"角色库来自模型 {got}，本程序使用 {expect}，声纹不通用")
        dim = lib.get("dim")
        if dim and dim != classify.MODEL_DIM:
            raise HTTPException(400, f"声纹维度不匹配（{dim} vs {classify.MODEL_DIM}）")
        lib.setdefault("model", expect)
        lib.setdefault("dim", classify.MODEL_DIM)
        classify.save_library(p, lib)
        names = [c["name"] for c in lib["characters"]]
        p.stats["characters"] = names
        p.save()
        return {"ok": True, "characters": names}

    @app.post("/api/calibrate")
    def calibrate():
        p = need_project()
        items, res, embs, _labels = st.effective_labels()
        lib = classify.load_library(p)
        if items is None or lib is None:
            raise HTTPException(409, "还没有分类结果：请先在第③步点「开始分类」")
        return classify.suggest_threshold(embs, lib, st.overrides, st.kinds)

    @app.post("/api/classify")
    def do_classify():
        p = need_project()
        return {"job": st.start_job("classify", lambda pr: pipeline.stage_classify(p, pr)["counts"])}

    # ---------- review ----------

    @app.get("/api/review")
    def review(label: str = "", offset: int = 0, limit: int = 60,
               order: str = "score", only_pending: bool = False):
        p = need_project()
        items, res, embs, labels = st.effective_labels()
        if items is None:
            raise HTTPException(409, "还没有分类结果：请先在第③步点「开始分类」")
        hidden = {i for i, it in enumerate(items) if it.get("hidden")}
        idxs = [i for i, lab in enumerate(labels)
                if i not in hidden and (not label or lab == label)]
        if order == "score":
            idxs.sort(key=lambda i: float(res["scores"][i]))
        elif order == "score_desc":
            idxs.sort(key=lambda i: -float(res["scores"][i]))
        if only_pending:
            idxs = [i for i in idxs if i not in st.overrides]
        page = idxs[offset:offset + limit]
        return {
            "total": len(idxs),
            "reviewed": len(st.overrides),
            "items": [{
                "index": i,
                "name": items[i]["name"],
                "dur": items[i]["dur"],
                "label": labels[i],
                "predicted": res["labels"][i],
                "score": round(float(res["scores"][i]), 4),
                "margin": round(float(res["margins"][i]), 4),
                "overridden": i in st.overrides,
            } for i in page],
        }

    @app.get("/api/audio/{index}")
    def clip(index: int):
        p = need_project()
        items, res, _embs, _labels = st.effective_labels()
        if items is None or not (0 <= index < len(items)):
            raise HTTPException(404)
        cache = os.path.join(p.cache_dir, "preview")
        os.makedirs(cache, exist_ok=True)
        dst = os.path.join(cache, f"{index}.wav")
        if not os.path.exists(dst):
            wav, sr = audio.decode(items[index]["path"])
            audio.write_wav(dst, wav, sr)
        return FileResponse(dst, media_type="audio/wav")

    @app.post("/api/review/action")
    def review_action(a: ReviewAction):
        p = need_project()
        if a.action == "reset":
            st.overrides.pop(a.index, None)
            st.kinds.pop(a.index, None)
        elif a.action == "confirm":
            st.overrides[a.index] = a.name or "其他"
            st.kinds[a.index] = "confirm"
        elif a.action in ("assign", "other", "block"):
            if a.action == "block":
                pipeline.add_blocklist(p, [a.index])
            st.overrides[a.index] = a.name if a.action == "assign" else "其他"
            st.kinds[a.index] = "fix"
        else:
            raise HTTPException(400, "unknown action")
        st.save_overrides()
        return {"ok": True, "override": st.overrides.get(a.index)}

    @app.post("/api/review/cascade")
    def cascade(req: CascadeReq):
        p = need_project()
        items, res, _embs, labels = st.effective_labels()
        if items is None:
            raise HTTPException(409, "还没有分类结果：请先在第③步点「开始分类」")
        label = req.label or labels[req.index]
        cands = pipeline.cascade_neighbours(p, req.index, label,
                                            req.threshold if req.threshold > 0 else None)
        return {"candidates": cands}

    # ---------- tuning / export / clean ----------

    @app.post("/api/threshold")
    def set_threshold(req: ThresholdReq):
        p = need_project()
        if req.threshold is not None:
            p.params.threshold = float(req.threshold)
        if req.margin is not None:
            p.params.margin = float(req.margin)
        p.save()
        out = pipeline.stage_classify(p)
        return {"ok": True, "counts": out["counts"]}

    @app.get("/api/estimate")
    def estimate():
        p = need_project()
        items, res, _embs, labels = st.effective_labels()
        if items is None:
            raise HTTPException(409)
        need = export.estimate_bytes(items, labels, "wav")
        need_copy = export.estimate_bytes(items, labels, "copy")
        selected = sum(1 for lab in labels if lab != "其他")
        free = 0
        try:
            import shutil as _sh
            free = _sh.disk_usage(os.path.dirname(p.output_dir) or ".").free
        except OSError:
            pass
        return {"needed": need, "needed_copy": need_copy,
                "selected": selected, "free": free}

    @app.post("/api/export")
    def do_export():
        p = need_project()

        def job(progress):
            items, res, _embs, labels = st.effective_labels()
            stats = export.run(p, items, labels, res["scores"], res["margins"], progress=progress)
            p.stats["export"] = {"counts": stats["counts"], "others": stats["others"]}
            p.advance("exported")
            return stats

        return {"job": st.start_job("export", job)}

    @app.get("/api/system")
    def system_info():
        """CPU info + a recommended worker count (12 was fastest on a 24-thread i7)."""
        logical = os.cpu_count() or 4
        recommended = max(1, min(12, logical // 2))
        return {"logical_cpus": logical, "recommended_workers": recommended,
                "min_workers": 1, "max_workers": max(1, min(16, logical))}

    @app.get("/api/browse")
    def browse(path: str = ""):
        """Server-side folder browser: no native dialogs, no blocking calls."""
        if not path:
            drives = []
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                root = f"{letter}:\\"
                if os.path.exists(root):
                    drives.append({"name": root, "path": root})
            return {"path": "", "parent": None, "dirs": drives, "audio": 0}

        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise HTTPException(400, "目录不存在")
        dirs = []
        try:
            with os.scandir(path) as it:
                for e in it:
                    try:
                        if not e.is_dir():
                            continue
                    except OSError:
                        continue
                    if e.name in ("System Volume Information", "$RECYCLE.BIN", "node_modules", ".git"):
                        continue
                    dirs.append({"name": e.name, "path": e.path})
        except PermissionError:
            raise HTTPException(403, "没有权限读取该目录")
        dirs.sort(key=lambda d: d["name"].lower())
        parent = os.path.dirname(path.rstrip("\\/")) or None
        try:
            audio_here = len(audio.list_audio(path, recursive=False))
        except Exception:
            audio_here = 0
        return {"path": path, "parent": parent, "dirs": dirs, "audio": audio_here}

    @app.post("/api/clean")
    def clean(req: CleanReq):
        p = need_project()
        freed = 0
        if "others" in req.scopes:
            freed += export.remove_others(p)
        if "cache" in req.scopes:
            freed += export.clean_cache(p, keep_embeddings=True)
        if "all_cache" in req.scopes:
            freed += export.clean_cache(p, keep_embeddings=False)
        return {"ok": True, "freed": freed}

    return app


def free_port(start: int = 8756, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start
