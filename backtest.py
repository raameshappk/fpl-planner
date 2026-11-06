#!/usr/bin/env python3
"""
Backtest for the FPL forecaster.

For each finished gameweek g (from --start), rebuild what the model could have known BEFORE g
(player xG/xA/minutes, team results and xG up to g-1), project GW g, and compare with the points
players actually scored. Reports error, bias and ranking quality against two simple baselines
(last-5 average and season average), writes:

  calibration.json  per-position scale factors (fitted on past gameweeks only when evaluated)
  accuracy.json     summary numbers for the website

Limits you should know about:
  * Injury/suspension news from the past is not available, so the model sees everyone as fit.
    Surprise absences therefore count as misses (this is realistic, but it adds noise).
  * Penalty/corner order and the player price use the values at the time where possible, otherwise today's.
  * Early in a season the sample is small. Treat the numbers as indicative.

Usage:
  python backtest.py                 # fetches player histories (about 450 requests, then cached)
  python backtest.py --start 4 --min-minutes 90
"""
import argparse
import copy
import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import fpl_forecast as f

N_TOP = {"GK": 5, "DEF": 10, "MID": 10, "FWD": 10}
CLAMP = (0.5, 1.3)


def fflt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- data
def fetch_histories(bootstrap, min_minutes=90, cache_path="backtest_cache.json"):
    """{player_id: [gameweek rows from element-summary]} for players with some minutes. Cached per finished-GW count."""
    finished = sum(1 for e in bootstrap["events"] if e["finished"])
    stamp = f"fin{finished}"
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    if cache.get("_stamp") != stamp:
        cache = {"_stamp": stamp}
    ids = [e["id"] for e in bootstrap["elements"] if fflt(e.get("minutes")) >= min_minutes]
    need = [i for i in ids if str(i) not in cache]

    def one(i):
        try:
            return i, f.get(f"element-summary/{i}/")["history"]
        except Exception:
            return i, None

    if need:
        print(f"Fetching histories for {len(need)} players (cached after the first run)...")
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, h in ex.map(one, need):
                if h is not None:
                    cache[str(i)] = h
        try:
            json.dump(cache, open(cache_path, "w"))
        except Exception:
            pass
    return {int(k): v for k, v in cache.items() if k != "_stamp"}


# --------------------------------------------------------------------------- point-in-time inputs
def pseudo_bootstrap(bs, hist, g):
    """The bootstrap as it would have looked just before gameweek g."""
    out = {"teams": bs["teams"], "chips": bs.get("chips", []),
           "events": [{"id": e["id"], "finished": e["id"] < g, "is_next": e["id"] == g,
                       "is_current": e["id"] == g - 1} for e in bs["events"]]}
    els = []
    for e in bs["elements"]:
        rows = sorted((r for r in hist.get(e["id"], []) if r["round"] < g), key=lambda r: r["round"])
        e2 = dict(e)
        mins = sum(fflt(r.get("minutes")) for r in rows)
        n90 = mins / 90.0
        xg = sum(fflt(r.get("expected_goals")) for r in rows)
        xa = sum(fflt(r.get("expected_assists")) for r in rows)
        e2.update({
            "minutes": mins,
            "expected_goals": str(xg),
            "expected_goals_per_90": str(xg / n90) if n90 > 0 else "0",
            "expected_assists_per_90": str(xa / n90) if n90 > 0 else "0",
            "defensive_contribution_per_90": str(sum(fflt(r.get("defensive_contribution")) for r in rows) / n90) if n90 > 0 else "0",
            "expected_goals_conceded": str(sum(fflt(r.get("expected_goals_conceded")) for r in rows)) if e["element_type"] == 1 else "0",
            "form": str(np.mean([fflt(r.get("total_points")) for r in rows[-5:]]) if rows else 0.0),
            "chance_of_playing_next_round": None, "status": "a", "news": "",
        })
        if rows and rows[-1].get("value"):
            e2["now_cost"] = rows[-1]["value"]
        els.append(e2)
    out["elements"] = els
    return out


def pseudo_fixtures(fixtures, g):
    out = []
    for fx in fixtures:
        x = dict(fx)
        if x["event"] is not None and x["event"] >= g:
            x.update(team_h_score=None, team_a_score=None, finished=False, finished_provisional=False)
        out.append(x)
    return out


# --------------------------------------------------------------------------- one gameweek
def evaluate_gw(bs, fixtures, hist, g, min_avg_minutes=45):
    pbs, pfx = pseudo_bootstrap(bs, hist, g), pseudo_fixtures(fixtures, g)
    S = f.team_strength(pbs, pfx)
    recent = {pid: [fflt(r.get("minutes")) for r in sorted(hist[pid], key=lambda r: r["round"]) if r["round"] < g][-5:]
              for pid in hist}
    p, gws, _ = f.forecast(pbs, pfx, 1, S, {}, recent, 0.10)
    nfx = {}
    for fx in fixtures:
        if fx["event"] == g:
            nfx[fx["team_h"]] = nfx.get(fx["team_h"], 0) + 1
            nfx[fx["team_a"]] = nfx.get(fx["team_a"], 0) + 1
    rows = []
    for pid, r in p.iterrows():
        h = [x for x in hist.get(pid, []) if x["round"] < g]
        if len(h) < 3 or nfx.get(int(r["team"]), 0) == 0:
            continue
        h.sort(key=lambda x: x["round"])
        last5 = [fflt(x.get("minutes")) for x in h[-5:]]
        if np.mean(last5) < min_avg_minutes:
            continue
        actual = sum(fflt(x.get("total_points")) for x in hist[pid] if x["round"] == g)
        k = nfx[int(r["team"])]
        form5 = np.mean([fflt(x.get("total_points")) for x in h[-5:]])
        season = np.mean([fflt(x.get("total_points")) for x in h])
        rows.append({"gw": g, "id": pid, "pos": r["pos"], "pred": float(r[f"gw{g}"]),
                     "b_form": form5 * k, "b_season": season * k, "actual": actual})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- metrics
def metrics(df, col):
    e = df[col] - df["actual"]
    # Spearman rank correlation per gameweek = Pearson on ranks (no scipy needed), then averaged
    rho = np.nanmean([d[col].rank().corr(d["actual"].rank()) for _, d in df.groupby("gw")])
    tops = []
    for (g, pos), d in df.groupby(["gw", "pos"]):
        tops.append(d.nlargest(N_TOP[pos], col)["actual"].mean())
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "rho": float(rho), "top": float(np.mean(tops))}


def fit_scale(df):
    s = {}
    for pos in N_TOP:
        d = df[df["pos"] == pos]
        s[pos] = float(np.clip(d["actual"].sum() / d["pred"].sum(), *CLAMP)) if len(d) >= 30 and d["pred"].sum() > 0 else 1.0
    return s


def run(bs, fixtures, hist, start, end):
    frames = []
    for g in range(start, end + 1):
        d = evaluate_gw(bs, fixtures, hist, g)
        print(f"  GW{g}: {len(d)} player-games")
        frames.append(d)
    df = pd.concat([d for d in frames if len(d)], ignore_index=True)

    # walk-forward calibration: scale for GW g is fitted only on earlier gameweeks
    df["pred_cal"] = df["pred"]
    for g in sorted(df["gw"].unique()):
        past = df[df["gw"] < g]
        if past["gw"].nunique() >= 2:
            sc = fit_scale(past)
            m = df["gw"] == g
            df.loc[m, "pred_cal"] = df.loc[m, "pred"] * df.loc[m, "pos"].map(sc)
    scored = df[df["gw"].isin([g for g in df["gw"].unique() if df[df["gw"] < g]["gw"].nunique() >= 2])]
    return df, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=4, help="first gameweek to test (needs a few results before it)")
    ap.add_argument("--min-minutes", type=int, default=90, help="only fetch players with at least this many season minutes")
    ap.add_argument("--out-dir", default=".", help="where to write calibration.json and accuracy.json")
    args = ap.parse_args()

    bs, fixtures = f.get("bootstrap-static/"), f.get("fixtures/")
    last = max((e["id"] for e in bs["events"] if e["finished"]), default=0)
    if last < args.start + 2:
        raise SystemExit(f"Only {last} gameweeks finished: need at least {args.start + 2} for a meaningful backtest.")
    hist = fetch_histories(bs, args.min_minutes)
    print(f"Backtesting GW{args.start}-GW{last}...")
    df, scored = run(bs, fixtures, hist, args.start, last)
    report(df, scored, args.out_dir, args.start, last)


def report(df, scored, out_dir, start, last):
    m_model, m_cal = metrics(scored, "pred"), metrics(scored, "pred_cal")
    m_form, m_season = metrics(scored, "b_form"), metrics(scored, "b_season")
    print(f"\n=== Results: {scored['gw'].min()}-{scored['gw'].max()}, {len(scored)} player-games ===")
    print(f"{'':<22}{'MAE':>7}{'RMSE':>7}{'Bias':>7}{'Rank corr':>11}{'Top-10 actual':>15}")
    for name, m in (("Model", m_model), ("Model (calibrated)", m_cal), ("Last-5 average", m_form), ("Season average", m_season)):
        print(f"{name:<22}{m['mae']:>7.2f}{m['rmse']:>7.2f}{m['bias']:>+7.2f}{m['rho']:>11.2f}{m['top']:>15.2f}")
    print("\nBias by position (projected minus actual, per player-game):")
    bias = {}
    for pos in N_TOP:
        d = scored[scored["pos"] == pos]
        bias[pos] = float((d["pred"] - d["actual"]).mean())
        print(f"  {pos:<4} {bias[pos]:+.2f}  (n={len(d)})")
    scale = fit_scale(df)
    print("\nSuggested scale per position (actual / projected):", {k: round(v, 2) for k, v in scale.items()})
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"scale": scale, "gws": [start, last], "n": int(len(df))}, open(os.path.join(out_dir, "calibration.json"), "w"), indent=1)
    acc = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "gws": [int(scored["gw"].min()), int(scored["gw"].max())],
           "n": int(len(scored)), "model": m_model, "calibrated": m_cal, "form": m_form, "season": m_season, "bias": bias}
    json.dump(acc, open(os.path.join(out_dir, "accuracy.json"), "w"), indent=1)
    print(f"Wrote calibration.json and accuracy.json to {out_dir}")


if __name__ == "__main__":
    main()
