#!/usr/bin/env python3
"""
Forward accuracy tracker: a live, honest track record.

  python track.py --snapshot   save the projections for the coming gameweek (history/gwN.json)
  python track.py --score      once a gameweek has finished, score its saved snapshot against what
                               players actually scored and update tracker.json

The benchmark is FPL's own expected points (ep_next, saved in the same snapshot), plus recent form.
Players counted: likely starters (chance of playing 75%+, averaging 45+ minutes a gameweek).
The snapshot is overwritten on every run until the deadline, so the last run before the deadline is
the one that gets scored. It cannot be back-filled: tracking starts from the first snapshot.
"""
import argparse
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

N_TOP = {"GK": 5, "DEF": 10, "MID": 10, "FWD": 10}


def snapshot(proj_path="projections.json", hist_dir="history", now=None):
    P = json.load(open(proj_path, encoding="utf-8"))
    gw = P["gws"][0]
    path = os.path.join(hist_dir, f"gw{gw}.json")
    now = now or dt.datetime.now(dt.timezone.utc)
    dl = P.get("deadline")
    if dl and os.path.exists(path):
        try:
            if now >= dt.datetime.fromisoformat(dl.replace("Z", "+00:00")):
                print(f"GW{gw} deadline has passed: keeping the saved snapshot.")
                return gw
        except ValueError:
            pass
    os.makedirs(hist_dir, exist_ok=True)
    snap = {"gw": gw, "saved": P["generated"], "deadline": dl, "players": {
        str(p["id"]): {"pos": p["p"], "ep": p["ep"][0], "fe": p.get("fe"), "fo": p.get("fo"), "a": p["a"], "m": p["m"]}
        for p in P["players"]}}
    json.dump(snap, open(path, "w"), separators=(",", ":"))
    print(f"Saved snapshot for GW{gw} ({len(snap['players'])} players) -> {path}")
    return gw


def metrics(df, col):
    e = df[col] - df["actual"]
    rho = df[col].rank().corr(df["actual"].rank())
    tops = [d.nlargest(N_TOP[p], col)["actual"].mean() for p, d in df.groupby("pos")]
    return {"mae": round(float(e.abs().mean()), 3), "bias": round(float(e.mean()), 3),
            "rho": round(float(rho), 3), "top": round(float(np.mean(tops)), 3)}


def evaluate(snap, live):
    gw = snap["gw"]
    rows = []
    for pid, p in snap["players"].items():
        if p["a"] < 0.75 or p["m"] < 45 * (gw - 1) or p.get("fe") is None:
            continue
        rows.append({"pos": p["pos"], "model": p["ep"], "fpl": p["fe"], "form": p.get("fo") or 0.0,
                     "actual": float(live.get(int(pid), 0))})
    if len(rows) < 30:
        return None
    df = pd.DataFrame(rows)
    return {"gw": gw, "n": int(len(df)), "model": metrics(df, "model"), "fpl": metrics(df, "fpl"), "form": metrics(df, "form")}


def aggregate(gws):
    n = sum(g["n"] for g in gws)
    out = {"n": n}
    for k in ("model", "fpl", "form"):
        out[k] = {m: round(sum(g[k][m] * g["n"] for g in gws) / n, 3) for m in ("mae", "bias", "rho", "top")}
    return out


def score(live_fn, finished, hist_dir="history", out="tracker.json"):
    tr = {"gws": []}
    if os.path.exists(out):
        try:
            tr = json.load(open(out))
        except Exception:
            tr = {"gws": []}
    done = {g["gw"] for g in tr["gws"]}
    snaps = sorted(glob.glob(os.path.join(hist_dir, "gw*.json")))
    for path in snaps:
        snap = json.load(open(path))
        gw = snap["gw"]
        if gw in done or gw not in finished:
            continue
        res = evaluate(snap, live_fn(gw))
        if res:
            tr["gws"].append(res)
            print(f"Scored GW{gw}: model error {res['model']['mae']:.2f} vs FPL's own {res['fpl']['mae']:.2f} ({res['n']} players)")
    tr["gws"].sort(key=lambda g: g["gw"])
    firsts = [json.load(open(p))["gw"] for p in snaps]
    tr["since"] = min(firsts) if firsts else None
    if tr["gws"]:
        tr["total"] = aggregate(tr["gws"])
    tr["updated"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    json.dump(tr, open(out, "w"), indent=1)
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if not (args.snapshot or args.score):
        ap.error("choose --snapshot and/or --score")
    if args.score:
        import fpl_forecast as f
        bs = f.get("bootstrap-static/")
        finished = {e["id"] for e in bs["events"] if e["finished"]}
        live = lambda gw: {el["id"]: el["stats"]["total_points"] for el in f.get(f"event/{gw}/live/")["elements"]}
        score(live, finished)
    if args.snapshot:
        snapshot()


if __name__ == "__main__":
    main()
