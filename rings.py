#!/usr/bin/env python3
"""rings.py - nested radial "generalization rings" from LeRobot v3 datasets.

    python rings.py INNER [MIDDLE ...] OUTER [options]

Each positional argument is a local dataset root or a Hugging Face repo id,
ordered innermost -> outermost. Ring 0 is a pie whose bars hang from its
perimeter toward the center; ring k>0 is an annulus whose bars hang from its
outer perimeter inward. Every ring is split into even thirds: state (blue),
task (green), environment (red). Outer rings are darker.

POINTS, per ring and category
  state        every frame's observation.state, z-scored with the INNERMOST
               ring's mean/std (one global scale). Joints matched by name.
  task         one point per episode: its task string.
  environment  video frames as tiny RGB thumbnails (default 16x12), z-scored
               per image and channel (--env-raw to skip). Each ring is decoded
               densely (up to --ref-points frames) to serve as a reference;
               slices are scored on an even subset (--max-points).

SCORE of a point p in ring k against ring k-1 (the next inner ring)
  d = distance to p's nearest neighbour among ALL ring k-1 reference points
      (environment: same camera key when available).
  sigma = innermost ring's grain: median distance from a point to the nearest
      point from a DIFFERENT episode, times --sigma-scale.
  --score coverage (default): 1 if d <= tau*sigma else 0   (--tau, default 1)
  --score kernel:             K(d/sigma), gauss or cauchy  (--kernel)
  task: d = normalised word-level edit distance to the closest ring k-1 task;
      coverage: 1 if d <= --task-tau (default 0 = exact match), kernel: 1-d.
  Ring 0: 1 if the dataset has no "success" column (placeholder), else the
      episode's success.

SLICES
  --unit episode (default): one slice per episode; value = mean score of that
      episode's points ("how much of this episode the inner ring covers").
  --unit cell: up to --n slices from farthest-point sampling; each owns the
      points nearest to it; value = mean score over the cell.
  --width mass (default): slice angle proportional to frames represented.
      With coverage scoring, filled area = covered fraction of the data.
  --width equal: equal angles.
  --sort value (default): highest values first, so covered slices form one arc.
  --sort natural: episode order (episode unit) or principal-axis order (cell).

The report (--report) lists every slice plus covered_fraction per category:
sum(value * frames) / sum(frames).
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["state", "task", "environment"]
BASE_RGB = {"state": (0.0, 0.0, 1.0), "task": (0.0, 0.63, 0.0), "environment": (1.0, 0.0, 0.0)}
ISSUES = []


def issue(msg):
    if msg not in ISSUES:
        ISSUES.append(msg)
        print(f"[issue] {msg}", file=sys.stderr)


# ----------------------------------------------------------------- loading
@dataclass
class Dataset:
    name: str
    root: Path
    info: dict
    frames: pd.DataFrame
    tasks: dict
    episodes: pd.DataFrame
    ep_len: pd.Series
    success: pd.Series = None  # per episode, if any success field exists


def resolve(src, cache_dir, want_video):
    p = Path(src).expanduser()
    if p.exists():
        return p, p.name
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"'{src}' is not a local path; install huggingface_hub to fetch it from the Hub.")
    patterns = None if want_video else ["meta/*", "meta/**", "data/**"]
    print(f"downloading {src} from the Hub ...", file=sys.stderr)
    local = snapshot_download(repo_id=src, repo_type="dataset", cache_dir=cache_dir,
                              allow_patterns=patterns)
    return Path(local), src


def load(src, cache_dir, want_video=True):
    import pyarrow.parquet as pq
    root, name = resolve(src, cache_dir, want_video)
    info = json.loads((root / "meta" / "info.json").read_text())
    ver = str(info.get("codebase_version", ""))
    if not ver.startswith("v3"):
        issue(f"{name}: codebase_version={ver!r}, expected v3.x - layout may not match")

    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        sys.exit(f"{name}: no parquet files under data/")
    schema = pq.read_schema(files[0]).names
    succ = [c for c in schema if "success" in c.lower()]
    keep = [c for c in ["observation.state", "episode_index", "task_index", "index"] if c in schema]
    frames = pd.concat([pd.read_parquet(f, columns=keep + succ) for f in files], ignore_index=True)

    tasks = {}
    tp = root / "meta" / "tasks.parquet"
    if tp.exists():
        t = pd.read_parquet(tp)
        strings = t["task"] if "task" in t.columns else t.index.to_series()
        tasks = dict(zip(t["task_index"].astype(int), strings.astype(str)))
    elif (root / "meta" / "tasks.jsonl").exists():
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
            r = json.loads(line)
            tasks[int(r["task_index"])] = r["task"]
    else:
        issue(f"{name}: no meta/tasks.parquet")

    ep_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True) if ep_files else pd.DataFrame()
    eps = eps[[c for c in eps.columns if not c.startswith("stats/")]]

    success = None
    if succ:
        success = frames.groupby("episode_index")[succ[0]].max().astype(float)
    else:
        ep_succ = [c for c in eps.columns if "success" in c.lower()]
        if ep_succ:
            success = eps.set_index("episode_index")[ep_succ[0]].astype(float)

    ep_len = frames.groupby("episode_index").size()
    print(f"loaded {name}: {len(frames)} frames, {len(ep_len)} episodes, {len(tasks)} tasks",
          file=sys.stderr)
    return Dataset(name, root, info, frames, tasks, eps, ep_len, success)


# ----------------------------------------------------------------- points
@dataclass
class Points:
    X: object            # (N, D) float32 array, or list of task strings
    episode: np.ndarray  # episode id per point
    group: np.ndarray    # camera key per point (environment), else ""
    weight: np.ndarray   # frames each point stands for


@dataclass
class CatData:
    ev: Points   # scored points: slices of this ring are built from these
    ref: Points  # reference points: used when this ring is the next inner one


def even_idx(n, m):
    return np.unique(np.linspace(0, n - 1, min(n, m)).round().astype(int)) if n else np.array([], int)


def subset(p, idx):
    X = [p.X[i] for i in idx] if isinstance(p.X, list) else p.X[idx]
    return Points(X, p.episode[idx], p.group[idx], p.weight[idx])


def flat_names(names):
    if isinstance(names, dict):
        return [n for v in names.values() for n in v]
    return list(names) if names else None


def state_data(ds, common):
    if "observation.state" not in ds.frames:
        issue(f"{ds.name}: no observation.state column")
        return None
    X = np.stack(ds.frames["observation.state"].to_numpy()).astype(np.float32)
    names = flat_names(ds.info["features"]["observation.state"].get("names"))
    X = X[:, [names.index(c) for c in common]] if names else X[:, :len(common)]
    ep = ds.frames["episode_index"].to_numpy()
    p = Points(X, ep, np.array([""] * len(X)), np.ones(len(X)))
    return CatData(p, p)


def task_data(ds):
    if "task_index" not in ds.frames or not ds.tasks:
        issue(f"{ds.name}: no task information")
        return None
    ep_task = ds.frames.groupby("episode_index")["task_index"].agg(lambda s: s.mode().iloc[0])
    eps = ep_task.index.to_numpy()
    X = [ds.tasks.get(int(t), f"<task {t}>") for t in ep_task.to_numpy()]
    p = Points(X, eps, np.array([""] * len(X)), ds.ep_len.loc[eps].to_numpy().astype(float))
    return CatData(p, p)


def video_keys(ds):
    return sorted(k for k, v in ds.info["features"].items() if v.get("dtype") == "video")


def episode_lookup(ds, key, chunk, file_i):
    e = ds.episodes
    cc, fc, tc = f"videos/{key}/chunk_index", f"videos/{key}/file_index", f"videos/{key}/from_timestamp"
    if e.empty or cc not in e:
        return None
    m = e[(e[cc] == chunk) & (e[fc] == file_i)].sort_values(tc)
    return m[tc].to_numpy(), m["episode_index"].to_numpy()


def env_data(ds, ref_pts, eval_pts, thumb, keyframes_only, normalise):
    import av
    keys = video_keys(ds)
    if not keys:
        kind = "images stored in parquet (dtype=image) - not supported yet" \
            if any(v.get("dtype") == "image" for v in ds.info["features"].values()) else "no camera features"
        issue(f"{ds.name}: {kind}")
        return None
    per_cam = max(1, ref_pts // len(keys))
    X, ep, grp = [], [], []
    tw, th = thumb
    for key in keys:
        files = sorted((ds.root / "videos" / key).glob("**/*.mp4"))
        if not files:
            issue(f"{ds.name}: no video files for {key} (downloaded without videos?)")
            continue
        totals = []
        for f in files:
            with av.open(str(f)) as c:
                s = c.streams.video[0]
                totals.append(s.frames or int(float(s.duration * s.time_base) * float(s.average_rate)))
        grand = max(1, sum(totals))
        for f, tot in zip(files, totals):
            stride = max(1, tot // max(1, round(per_cam * tot / grand)))
            chunk = int(re.search(r"chunk-(\d+)", str(f)).group(1))
            file_i = int(re.search(r"file-(\d+)", f.name).group(1))
            lk = episode_lookup(ds, key, chunk, file_i)
            if lk is None:
                issue(f"{ds.name}: no episode/video timestamps; using 2 s windows as pseudo-episodes")
            with av.open(str(f)) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                if keyframes_only:
                    s.codec_context.skip_frame = "NONKEY"
                for j, fr in enumerate(c.decode(s)):
                    if not keyframes_only and j % stride:
                        continue
                    img = fr.reformat(width=tw, height=th, format="rgb24").to_ndarray()
                    img = img.reshape(-1, 3).astype(np.float32) / 255.0
                    if normalise:
                        img = (img - img.mean(0)) / (img.std(0) + 1e-3)
                    X.append(img.reshape(-1))
                    t = fr.time or 0.0
                    if lk is not None and len(lk[0]):
                        k = max(0, np.searchsorted(lk[0], t + 1e-6, side="right") - 1)
                        ep.append(int(lk[1][k]))
                    else:
                        ep.append(int(t // 2.0))
                    grp.append(key)
    if not X:
        return None
    ref = Points(np.array(X, np.float32), np.array(ep), np.array(grp), np.ones(len(X)))
    return CatData(subset(ref, even_idx(len(X), eval_pts)), ref)


# ----------------------------------------------------------------- distances
def word_edit(a, b):
    a, b = re.findall(r"[a-z0-9]+", a.lower()), re.findall(r"[a-z0-9]+", b.lower())
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1] / max(len(a), len(b), 1)


def dist_block(A, B):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def nn_min(P, Q):
    """Distance from each row of P to its nearest row of Q."""
    if P.shape[1] <= 32:
        from scipy.spatial import cKDTree
        return cKDTree(Q).query(P, k=1)[0]
    rows = max(1, int(2e7 // max(1, len(Q))))
    return np.concatenate([dist_block(P[s:s + rows], Q).min(1) for s in range(0, len(P), rows)])


def nn_dist(ev, ref, by_group):
    if not by_group:
        return nn_min(ev.X, ref.X)
    d = np.empty(len(ev.X))
    for g in np.unique(ev.group):
        mp, mq = ev.group == g, ref.group == g
        if not mq.any():
            issue(f"camera '{g}' has no counterpart in the next inner ring; comparing against all cameras")
            mq = np.ones(len(ref.X), bool)
        d[mp] = nn_min(ev.X[mp], ref.X[mq])
    return d


def grain(cd, n_eval, n_ref):
    """Median distance from a point to the nearest point of a different episode (same camera)."""
    P = subset(cd.ev, even_idx(len(cd.ev.X), n_eval))
    Q = subset(cd.ref, even_idx(len(cd.ref.X), n_ref))
    rows = max(1, int(2e7 // max(1, len(Q.X))))
    mins = []
    for s in range(0, len(P.X), rows):
        D = dist_block(P.X[s:s + rows], Q.X)
        D[P.episode[s:s + rows, None] == Q.episode[None]] = np.inf
        D[P.group[s:s + rows, None] != Q.group[None]] = np.inf
        mins.append(D.min(1))
    m = np.concatenate(mins)
    m = m[np.isfinite(m)]
    if not len(m):
        issue("innermost ring has a single episode; sigma from within-episode distances")
        D = dist_block(P.X, P.X)
        np.fill_diagonal(D, np.inf)
        m = D.min(1)
    return float(np.median(m))


# ----------------------------------------------------------------- scoring
def kernel(z, kind):
    return np.exp(-0.5 * z ** 2) if kind == "gauss" else 1.0 / (1.0 + z ** 2)


def score_points(cat, cd, inner, sigma, a):
    if cat == "task":
        uniq = sorted(set(inner.ref.X))
        cache = {t: min(word_edit(t, u) for u in uniq) for t in set(cd.ev.X)}
        d = np.array([cache[t] for t in cd.ev.X])
        return (d <= a.task_tau).astype(float) if a.score == "coverage" else 1 - d
    d = nn_dist(cd.ev, inner.ref, by_group=(cat == "environment"))
    z = d / sigma
    return (z <= a.tau).astype(float) if a.score == "coverage" else kernel(z, a.kernel)


def method_text(cat, k, a, sigma, has_success):
    if k == 0:
        return "episode success" if has_success else "PLACEHOLDER: no success field; value = 1"
    if cat == "task":
        return (f"coverage: word-edit distance to ring k-1 tasks <= {a.task_tau}" if a.score == "coverage"
                else "1 - min normalised word-edit distance to ring k-1 tasks")
    if a.score == "coverage":
        return f"coverage: d_nn to ring k-1 <= {a.tau} * sigma (sigma={sigma:.4g})"
    return f"kernel {a.kernel}: K(d_nn / sigma), sigma={sigma:.4g}"


# ----------------------------------------------------------------- slices
def farthest_points(D_from, n_pts, n, start):
    chosen, mind = [start], D_from(start)
    for _ in range(1, min(n, n_pts)):
        nxt = int(np.argmax(mind))
        if mind[nxt] <= 1e-12:
            break
        chosen.append(nxt)
        mind = np.minimum(mind, D_from(nxt))
    return chosen


def slices_by_episode(cat, ds, ev, scores):
    out, missing = [], 0
    for e in ds.ep_len.index:
        m = ev.episode == e
        if m.any():
            v = float(np.average(scores[m], weights=ev.weight[m]))
        else:
            v, missing = 0.0, missing + 1
        out.append({"label": f"ep {int(e)}", "value": v, "mass": float(ds.ep_len.loc[e]), "order": int(e)})
    if missing:
        issue(f"{ds.name} {cat}: {missing} episodes had no sampled points (value 0); raise --max-points")
    return out


def slices_by_cell(cat, ev, scores, n):
    if cat == "task":
        uniq = sorted(set(ev.X))
        D = np.array([[word_edit(x, y) for y in uniq] for x in uniq])
        w = np.array([ev.weight[[t == u for t in ev.X]].sum() for u in uniq])
        S = farthest_points(lambda i: D[i], len(uniq), n, int(np.argmax(w)))
        owner = {u: uniq[S[int(np.argmin(D[i, S]))]] for i, u in enumerate(uniq)}
        cell = np.array([owner[t] for t in ev.X])
        labels = {uniq[s]: (uniq[s], i) for i, s in enumerate(sorted(S, key=lambda s: uniq[s].lower()))}
    else:
        X = ev.X
        start = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
        S = farthest_points(lambda i: np.sqrt(((X - X[i]) ** 2).sum(1)), len(X), n, start)
        rows = max(1, int(2e7 // len(S)))
        cell = np.concatenate([np.array(S)[dist_block(X[s:s + rows], X[S]).argmin(1)]
                               for s in range(0, len(X), rows)])
        Xc = X - X.mean(0)
        pc1 = np.linalg.svd(Xc[even_idx(len(X), 5000)], full_matrices=False)[2][0]
        order = sorted(S, key=lambda i: (ev.group[i], float(Xc[i] @ pc1)))
        labels = {s: (f"{ev.group[s] or 'cell'} #{i}", i) for i, s in enumerate(order)}
    out = []
    for key, (lab, o) in labels.items():
        m = cell == key
        out.append({"label": lab, "value": float(np.average(scores[m], weights=ev.weight[m])),
                    "mass": float(ev.weight[m].sum()), "order": o})
    return out


# ----------------------------------------------------------------- render
def render(rings, path, title, width_mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge

    L = len(rings)
    R0, T = 0.5, 0.45
    fig, ax = plt.subplots(figsize=(8, 8.6))
    third = 120.0
    for k, ring in enumerate(rings):
        dark = 1 - 0.5 * k / max(1, L - 1)
        r_out = R0 + k * T
        depth_max = R0 if k == 0 else T
        for c_i, cat in enumerate(CATS):
            col = tuple(v * dark for v in BASE_RGB[cat])
            a0 = c_i * third
            entry = ring["categories"].get(cat)
            if not entry or not entry["slices"]:
                ax.add_patch(Wedge((0, 0), r_out, 90 - a0 - third, 90 - a0, width=depth_max,
                                   fc="none", ec="0.6", hatch="///", lw=0))
                continue
            sl = entry["slices"]
            mass = np.array([s["mass"] for s in sl]) if width_mode == "mass" else np.ones(len(sl))
            spans = third * mass / mass.sum()
            s0 = a0
            for s, span in zip(sl, spans):
                depth = depth_max * min(max(s["value"], 0.0), 1.0)
                if depth > 0:
                    ax.add_patch(Wedge((0, 0), r_out, 90 - s0 - span, 90 - s0, width=depth,
                                       fc=col, ec=col, lw=0.2))
                s0 += span
        ax.add_patch(Circle((0, 0), r_out, fill=False, ec="0.55", lw=0.8))
    rmax = R0 + (L - 1) * T
    for c_i, cat in enumerate(CATS):
        mid = np.deg2rad(90 - (c_i + 0.5) * third)
        ax.text(1.15 * rmax * np.cos(mid), 1.15 * rmax * np.sin(mid), cat,
                ha="center", va="center", fontsize=12, color=BASE_RGB[cat])
    lim = rmax * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    legend = "\n".join(
        f"ring {k} ({'innermost' if k == 0 else 'vs ring ' + str(k - 1)}): {r['name']}  covered "
        + "  ".join(f"{c[0]}={r['categories'][c]['covered_fraction']:.2f}" if r["categories"].get(c)
                    else f"{c[0]}=--" for c in CATS)
        for k, r in enumerate(rings))
    ax.set_title(title, fontsize=11)
    fig.text(0.02, 0.01, legend, fontsize=8, family="monospace", va="bottom")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="local roots or HF repo ids, innermost first")
    ap.add_argument("--unit", choices=["episode", "cell"], default="episode")
    ap.add_argument("--width", choices=["mass", "equal"], default="mass")
    ap.add_argument("--score", choices=["coverage", "kernel"], default="coverage")
    ap.add_argument("--tau", type=float, default=1.0, help="coverage threshold, in units of sigma")
    ap.add_argument("--task-tau", type=float, default=0.0, help="task coverage: max word-edit distance")
    ap.add_argument("--kernel", choices=["gauss", "cauchy"], default="gauss")
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--sort", choices=["value", "natural"], default="value")
    ap.add_argument("--n", type=int, default=24, help="max slices per category (cell unit)")
    ap.add_argument("--max-points", type=int, default=3000, help="scored environment frames per ring")
    ap.add_argument("--ref-points", type=int, default=20000, help="decoded reference frames per ring")
    ap.add_argument("--thumb", default="16x12", help="environment thumbnail WxH")
    ap.add_argument("--env-raw", action="store_true", help="don't normalise thumbnails per image")
    ap.add_argument("--env-keyframes", action="store_true", help="decode keyframes only (fast)")
    ap.add_argument("--no-env", action="store_true", help="skip video (and video download)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="rings.png")
    ap.add_argument("--report", default="rings.json")
    a = ap.parse_args(argv)
    thumb = tuple(int(v) for v in a.thumb.lower().split("x"))

    dss = [load(s, a.cache_dir, want_video=not a.no_env) for s in a.datasets]

    names = [flat_names(d.info["features"].get("observation.state", {}).get("names")) for d in dss]
    if all(names):
        common = [n for n in names[0] if all(n in m for m in names[1:])]
        dropped = sorted(set(sum(names, [])) - set(common))
        if dropped:
            issue(f"state dims not shared by every dataset were dropped: {dropped}")
    else:
        dims = [d.info["features"].get("observation.state", {}).get("shape", [0])[0] for d in dss]
        common = list(range(min(dims)))
        if len(set(dims)) > 1:
            issue(f"state dims differ {dims} and lack names; truncated to first {min(dims)}")

    data = []
    for d in dss:
        data.append({
            "state": state_data(d, common) if common else None,
            "task": task_data(d),
            "environment": None if a.no_env else env_data(d, a.ref_points, a.max_points, thumb,
                                                          a.env_keyframes, not a.env_raw)})

    st0 = data[0]["state"]
    if st0 is not None:
        mu, sd = st0.ref.X.mean(0), st0.ref.X.std(0)
        floor = 0.05 * (sd.mean() if sd.mean() > 0 else 1.0)
        if (sd < floor).any():
            issue(f"innermost state has near-constant dims (std floored at {floor:.3g})")
        sd = np.maximum(sd, floor)
        for dd in data:
            if dd["state"] is not None:
                dd["state"].ref.X = ((dd["state"].ref.X - mu) / sd).astype(np.float32)
                dd["state"].ev = dd["state"].ref

    sigmas = {cat: grain(data[0][cat], a.max_points, a.ref_points) * a.sigma_scale
              for cat in ["state", "environment"] if data[0][cat] is not None}

    rings = []
    for k, (ds, dd) in enumerate(zip(dss, data)):
        ring = {"name": ds.name, "categories": {}}
        for cat in CATS:
            cd = dd[cat]
            if cd is None:
                continue
            if k == 0:
                if ds.success is not None:
                    scores = np.array([ds.success.get(e, np.nan) for e in cd.ev.episode])
                    scores = np.nan_to_num(scores, nan=0.0)
                else:
                    scores = np.ones(len(cd.ev.episode))
            elif data[k - 1][cat] is None or (cat != "task" and cat not in sigmas):
                issue(f"ring {k} {cat}: no inner reference; values set to 0")
                scores = np.zeros(len(cd.ev.episode))
            else:
                scores = score_points(cat, cd, data[k - 1][cat], sigmas.get(cat), a)

            sl = slices_by_episode(cat, ds, cd.ev, scores) if a.unit == "episode" \
                else slices_by_cell(cat, cd.ev, scores, a.n)
            key = (lambda s: (-s["value"], s["order"])) if a.sort == "value" else (lambda s: s["order"])
            sl = sorted(sl, key=key)
            mass = np.array([s["mass"] for s in sl])
            vals = np.array([s["value"] for s in sl])
            ring["categories"][cat] = {
                "method": method_text(cat, k, a, sigmas.get(cat), ds.success is not None),
                "unit": a.unit,
                "covered_fraction": float((vals * mass).sum() / mass.sum()),
                "slices": [{kk: s[kk] for kk in ("label", "value", "mass")} for s in sl]}
        rings.append(ring)

    title = "generalization rings (inner -> outer): " + " | ".join(d.name for d in dss)
    render(rings, a.out, title, a.width)
    settings = {k: v for k, v in vars(a).items() if k not in ("datasets", "out", "report", "cache_dir")}
    Path(a.report).write_text(json.dumps(
        {"settings": settings, "sigmas": sigmas, "rings": rings, "issues": ISSUES}, indent=2))

    print(f"\nwrote {a.out} and {a.report}")
    for k, r in enumerate(rings):
        print(f"ring {k}: {r['name']}")
        for cat in CATS:
            e = r["categories"].get(cat)
            if e:
                print(f"  {cat:12s} slices={len(e['slices']):4d}  covered={e['covered_fraction']:.3f}")
            else:
                print(f"  {cat:12s} (missing)")
    if ISSUES:
        print("\nissues:")
        for m in ISSUES:
            print(" -", m)
    return rings


if __name__ == "__main__":
    main()
