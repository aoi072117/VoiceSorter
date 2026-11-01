"""Command line entry point."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser

from .project import Params, Project


def _progress_printer(prefix: str):
    def progress(*args):
        if len(args) == 1:
            print(f"\r{prefix}: {args[0]}", end="", flush=True)
        else:
            print(f"\r{prefix}: {args[0]}/{args[1]}", end="", flush=True)
    return progress


def cmd_init(args) -> int:
    p = Project.create(args.project, args.input, args.output, args.name)
    print(f"project created at {p.path}")
    return 0


def cmd_scan(args) -> int:
    from . import pipeline
    p = Project.load(args.project)
    res = pipeline.stage_scan(p)
    print(f"files={len(res['files'])} duplicates={len(res['duplicates'])} unreadable={len(res['errors'])}")
    return 0


def cmd_embed(args) -> int:
    from . import pipeline
    p = Project.load(args.project)
    if args.workers:
        p.params.workers = args.workers
        p.save()
    pipeline.stage_embed(p, _progress_printer("embed"))
    print()
    return 0


def cmd_cluster(args) -> int:
    from . import pipeline
    p = Project.load(args.project)
    data = pipeline.stage_cluster(p, args.sample_size, args.k, args.seed,
                                  _progress_printer("cluster"))
    print(f"\n{len(data['clusters'])} clusters from {data['sample_size']} sampled clips")
    for c in data["clusters"]:
        print(f"  cluster {c['id']:02d}  {c['size']:5d} clips  {c['share']*100:5.1f}%")
    print(f"samples written to {p.samples_dir}")
    return 0


def cmd_build(args) -> int:
    from . import pipeline
    p = Project.load(args.project)
    assignment = {}
    for item in args.assign or []:
        cid, name = item.split("=", 1)
        assignment[int(cid)] = name or None
    lib = pipeline.stage_build_library(p, assignment)
    print("characters:", [(c["name"], c["size"]) for c in lib["characters"]])
    return 0


def cmd_classify(args) -> int:
    from . import pipeline
    p = Project.load(args.project)
    if args.threshold is not None:
        p.params.threshold = args.threshold
    if args.margin is not None:
        p.params.margin = args.margin
    p.save()
    out = pipeline.stage_classify(p)
    for name, n in out["counts"].items():
        print(f"  {name}: {n}")
    return 0


def cmd_export(args) -> int:
    from . import pipeline
    p = Project.load(args.project)
    stats = pipeline.stage_export(p, _progress_printer("export"))
    print()
    print("counts:", stats["counts"], "others:", stats["others"])
    print("manifest:", stats["manifest"])
    return 0


def cmd_clean(args) -> int:
    from . import export
    p = Project.load(args.project)
    freed = 0
    if args.others:
        freed += export.remove_others(p)
    if args.cache:
        freed += export.clean_cache(p, keep_embeddings=not args.drop_embeddings)
    print(f"freed {freed / 1024 / 1024:.1f} MB")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app, free_port

    project = Project.load(args.project) if args.project else None
    app = create_app(project)
    port = args.port or free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"VoiceSorter running at {url}")
    print("close this window to stop the server")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="voicesorter", description="Sort voice clips by speaker.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a project")
    p.add_argument("--project", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", default="")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("scan", help="list and dedupe input audio")
    p.add_argument("--project", required=True)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("embed", help="extract speaker embeddings")
    p.add_argument("--project", required=True)
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("cluster", help="cluster a sample of clips")
    p.add_argument("--project", required=True)
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("build", help="name clusters into a character library")
    p.add_argument("--project", required=True)
    p.add_argument("--assign", nargs="*", help="clusterId=Name pairs, e.g. 0=Alice 1=Bob")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("classify", help="classify the whole collection")
    p.add_argument("--project", required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--margin", type=float)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("export", help="write per-character folders")
    p.add_argument("--project", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("clean", help="remove intermediate files")
    p.add_argument("--project", required=True)
    p.add_argument("--others", action="store_true", help="delete the '其他' export folder")
    p.add_argument("--cache", action="store_true", help="delete previews and samples")
    p.add_argument("--drop-embeddings", action="store_true")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("serve", help="start the local web UI")
    p.add_argument("--project", default="")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
