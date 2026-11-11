#!/usr/bin/env python3
"""
Machine-learning layer for the FPL forecaster (LightGBM).

It learns "points in a gameweek" from past seasons using only information that was known before that
gameweek: recent points, minutes, xG/xA per 90, defensive actions, price, home/away, number of fixtures,
and how strong the team and the opponent have recently been (goals and xG for/against).

It is trained on past seasons (vaastav/Fantasy-Premier-League data, downloaded once and cached) plus the
current season so far, and is then BLENDED with the hand-built projection:

    final = w * ML + (1 - w) * physics        (w is chosen by the backtest, per position)

Commands
  python ml_model.py validate              train on older seasons, test on the latest finished one (offline check)
  python ml_model.py backtest              compare ML, physics and the blend on the current season (needs FPL API)
  python ml_model.py train                 fit on everything and save ml_model.txt + ml_blend.json

Everything here is leak-free: features for gameweek g use only gameweeks before g.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/gws/merged_gw.csv"
POS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
FEATURES = [
    "pos", "price", "nfix", "home",
    "ppf_l3", "ppf_l5", "ppf_l10", "ppf_all", "min_l1", "min_l3", "min_l5", "min_all", "n_prior",
    "xg90_l5", "xg90_all", "xa90_l5", "xa90_all", "xgc90_l5", "dc90_l5", "dc90_all",
    "team_gf", "team_ga", "team_xgf", "team_xga", "opp_gf", "opp_ga", "opp_xgf", "opp_xga",
]
N_TOP = {1: 5, 2: 10, 3: 10, 4: 10}
PARAMS = dict(objective="regression", learning_rate=0.03, n_estimators=450, num_leaves=15, min_child_samples=60,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


# --------------------------------------------------------------------------- data adapters
def load_season(season, cache_dir="ml_data"):
    """One past season in the common schema (one row per player per fixture)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{season}.csv")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(RAW.format(s=season), path)
    d = pd.read_csv(path, low_memory=False)
    # team id <-> name from the fixtures themselves: rows of team A carry the id of team B as opponent_team
    names = d.groupby("fixture")["team"].agg(lambda s: sorted(set(s)))
    idmap = {}
    for fx, grp in d.groupby("fixture"):
        t = names[fx]
        if len(t) != 2:
            continue
        for a, b in ((t[0], t[1]), (t[1], t[0])):
            ids = grp.loc[grp["team"] == a, "opponent_team"]
            if len(ids):
                idmap.setdefault(int(ids.iloc[0]), b)
    out = pd.DataFrame({
        "pid": season + ":" + d["element"].astype(str), "season": season, "pos": d["position"].map(POS),
        "gw": d["GW"].astype(int), "fixture": season + ":" + d["fixture"].astype(str),
        "team": d["team"], "opp": d["opponent_team"].map(idmap), "home": d["was_home"].astype(float),
        "minutes": d["minutes"].astype(float), "points": d["total_points"].astype(float),
        "xg": d["expected_goals"].astype(float), "xa": d["expected_assists"].astype(float),
        "xgc": d["expected_goals_conceded"].astype(float),
        "dc": d["defensive_contribution"].astype(float) if "defensive_contribution" in d else np.nan,
        "price": d["value"].astype(float),
        "gf": np.where(d["was_home"], d["team_h_score"], d["team_a_score"]).astype(float),
        "ga": np.where(d["was_home"], d["team_a_score"], d["team_h_score"]).astype(float),
    })
    return out.dropna(subset=["pos", "opp"])


def from_fpl_history(bootstrap, histories, season="cur"):
    """The current season, from element-summary histories (the same rows backtest.py caches)."""
    tname = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    rows = []
    for e in bootstrap["elements"]:
        for r in histories.get(e["id"], []):
            home = bool(r.get("was_home"))
            hs, as_ = r.get("team_h_score"), r.get("team_a_score")
            if hs is None or as_ is None:
                continue
            rows.append({
                "pid": f"{season}:{e['id']}", "season": season, "pos": int(e["element_type"]), "gw": int(r["round"]),
                "fixture": f"{season}:{r['fixture']}", "team": tname[e["team"]], "opp": tname[r["opponent_team"]],
                "home": float(home), "minutes": _f(r.get("minutes")), "points": _f(r.get("total_points")),
                "xg": _f(r.get("expected_goals")), "xa": _f(r.get("expected_assists")),
                "xgc": _f(r.get("expected_goals_conceded")),
                "dc": _f(r["defensive_contribution"]) if r.get("defensive_contribution") is not None else np.nan,
                "price": _f(r.get("value")), "gf": float(hs if home else as_), "ga": float(as_ if home else hs)})
    return pd.DataFrame(rows)


def upcoming_rows(bootstrap, fixtures, gw, season="cur"):
    """Placeholder rows (no outcomes yet) for the fixtures of gameweek gw, one per player per fixture."""
    tname = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    rows = []
    for fx in fixtures:
        if fx["event"] != gw:
            continue
        for team, opp, home in ((fx["team_h"], fx["team_a"], 1.0), (fx["team_a"], fx["team_h"], 0.0)):
            for e in bootstrap["elements"]:
                if e["team"] == team:
                    rows.append({"pid": f"{season}:{e['id']}", "season": season, "pos": int(e["element_type"]), "gw": gw,
                                 "fixture": f"{season}:f{fx['id']}", "team": tname[team], "opp": tname[opp], "home": home,
                                 "minutes": np.nan, "points": np.nan, "xg": np.nan, "xa": np.nan, "xgc": np.nan,
                                 "dc": np.nan, "price": float(e["now_cost"]), "gf": np.nan, "ga": np.nan})
    return pd.DataFrame(rows)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- features
def _team_table(df):
    """Rolling team form (last 6 matches) as known BEFORE each gameweek: goals and xG for/against per match."""
    d = df.dropna(subset=["gf"])
    tf = d.groupby(["season", "fixture", "team", "opp", "gw"]).agg(gf=("gf", "first"), ga=("ga", "first"), xgf=("xg", "sum")).reset_index()
    opp_x = tf[["season", "fixture", "team", "xgf"]].rename(columns={"team": "opp", "xgf": "xga"})
    tf = tf.merge(opp_x, on=["season", "fixture", "opp"], how="left")
    g = tf.groupby(["season", "team", "gw"]).agg(gf=("gf", "sum"), ga=("ga", "sum"), xgf=("xgf", "sum"), xga=("xga", "sum"),
                                                  n=("gf", "size")).reset_index().sort_values(["season", "team", "gw"])
    for c in ("gf", "ga", "xgf", "xga", "n"):
        g[c] = g.groupby(["season", "team"])[c].transform(lambda s: s.rolling(6, min_periods=1).sum())
    for c in ("gf", "ga", "xgf", "xga"):
        g[c] = g[c] / g["n"]
    return g[["season", "team", "gw", "gf", "ga", "xgf", "xga"]]


def _lookup(rows, tt, key, prefix):
    """Attach the latest team stats from a gameweek strictly before the row's gameweek."""
    left = rows[["season", "gw", key]].reset_index().rename(columns={key: "team"}).sort_values("gw")
    right = tt.sort_values("gw")
    m = pd.merge_asof(left, right, on="gw", by=["season", "team"], allow_exact_matches=False)
    m = m.set_index("index")
    return m[["gf", "ga", "xgf", "xga"]].add_prefix(prefix)


def build_features(df):
    """df: common-schema rows (past results plus optional placeholder rows). Returns one row per player per gameweek."""
    df = df.copy()
    tt = _team_table(df)
    df = df.reset_index(drop=True)
    df = df.join(_lookup(df, tt, "team", "team_")).join(_lookup(df, tt, "opp", "opp_"))
    agg = df.groupby(["pid", "season", "gw"]).agg(
        pos=("pos", "first"), price=("price", "last"), nfix=("fixture", "size"), home=("home", "mean"),
        minutes=("minutes", "sum"), points=("points", "sum"), xg=("xg", "sum"), xa=("xa", "sum"), xgc=("xgc", "sum"),
        dc=("dc", "sum"), team_gf=("team_gf", "mean"), team_ga=("team_ga", "mean"), team_xgf=("team_xgf", "mean"),
        team_xga=("team_xga", "mean"), opp_gf=("opp_gf", "mean"), opp_ga=("opp_ga", "mean"), opp_xgf=("opp_xgf", "mean"),
        opp_xga=("opp_xga", "mean"), pending=("points", lambda s: s.isna().all())).reset_index()
    agg = agg.sort_values(["pid", "gw"]).reset_index(drop=True)
    done = ~agg["pending"]
    for c in ("minutes", "points", "xg", "xa", "xgc", "dc"):
        agg.loc[~done, c] = np.nan
    agg["ppf"] = agg["points"] / agg["nfix"]
    agg["mpf"] = agg["minutes"] / agg["nfix"]
    g = agg.groupby("pid")

    def prev(col, n=None):
        # statistics over earlier rows only (shift by one), as sum over the last n rows or all
        s = g[col].shift(1)
        if n is None:
            return s.groupby(agg["pid"]).transform(lambda x: x.expanding().sum())
        return s.groupby(agg["pid"]).transform(lambda x: x.rolling(n, min_periods=1).sum())

    def mean_prev(col, n=None):
        s = g[col].shift(1)
        t = s.groupby(agg["pid"])
        return t.transform(lambda x: x.expanding().mean()) if n is None else t.transform(lambda x: x.rolling(n, min_periods=1).mean())

    agg["n_prior"] = g.cumcount()
    for n in (3, 5, 10):
        agg[f"ppf_l{n}"] = mean_prev("ppf", n)
    agg["ppf_all"] = mean_prev("ppf")
    agg["min_l1"] = mean_prev("mpf", 1)
    agg["min_l3"], agg["min_l5"], agg["min_all"] = mean_prev("mpf", 3), mean_prev("mpf", 5), mean_prev("mpf")
    m5, mall = prev("minutes", 5), prev("minutes")
    for name, col in (("xg90", "xg"), ("xa90", "xa")):
        agg[f"{name}_l5"] = prev(col, 5) / m5.where(m5 > 0) * 90
        agg[f"{name}_all"] = prev(col) / mall.where(mall > 0) * 90
    agg["xgc90_l5"] = prev("xgc", 5) / m5.where(m5 > 0) * 90
    agg["dc90_l5"] = prev("dc", 5) / m5.where(m5 > 0) * 90
    agg["dc90_all"] = prev("dc") / mall.where(mall > 0) * 90
    return agg


def regulars(f, min_l5=45.0, min_prior=3):
    """The rows the model is trained and judged on: players who have been playing."""
    return f[(f["n_prior"] >= min_prior) & (f["min_l5"] >= min_l5)]


# --------------------------------------------------------------------------- model
def fit(train):
    import lightgbm as lgb
    t = train[train["points"].notna()]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(t[FEATURES], t["points"])
    return m


def metrics(d, col):
    e = d[col] - d["points"]
    rho = np.nanmean([x[col].rank().corr(x["points"].rank()) for _, x in d.groupby("gw")])
    top = np.mean([x.nlargest(N_TOP[p], col)["points"].mean() for (_, p), x in d.groupby(["gw", "pos"])])
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "rho": float(rho), "top": float(top)}


def show(rows):
    print(f"{'':<24}{'MAE':>7}{'RMSE':>7}{'Bias':>7}{'RankCorr':>10}{'Top-10':>9}")
    for name, m in rows:
        print(f"{name:<24}{m['mae']:>7.2f}{m['rmse']:>7.2f}{m['bias']:>+7.2f}{m['rho']:>10.3f}{m['top']:>9.2f}")


def validate(train_seasons, test_season, start=6):
    frames = {s: load_season(s) for s in train_seasons + [test_season]}
    feats = {s: build_features(frames[s]) for s in frames}
    tr = regulars(pd.concat([feats[s] for s in train_seasons], ignore_index=True))
    te = regulars(feats[test_season])
    te = te[te["gw"] >= start].copy()
    model = fit(tr)
    te["ml"] = model.predict(te[FEATURES])
    te["b_form"] = te["ppf_l5"] * te["nfix"]
    te["b_season"] = te["ppf_all"] * te["nfix"]
    print(f"Train {train_seasons} ({len(tr)} rows) -> test {test_season} GW{start}+, {len(te)} player-gameweeks")
    show([("LightGBM", metrics(te, "ml")), ("Last-5 average", metrics(te, "b_form")), ("Season average", metrics(te, "b_season"))])
    for lo, hi in ((start, 9), (10, 19), (20, 38)):
        d = te[(te["gw"] >= lo) & (te["gw"] <= hi)]
        if len(d):
            print(f"\nGW{lo}-{hi} ({len(d)} rows)")
            show([("LightGBM", metrics(d, "ml")), ("Last-5 average", metrics(d, "b_form")), ("Season average", metrics(d, "b_season"))])
    print("\nBias by position (predicted - actual):", {p: round(float((d["ml"] - d["points"]).mean()), 2) for p, d in te.groupby("pos")})
    imp = sorted(zip(model.feature_importances_, FEATURES), reverse=True)[:8]
    print("Top features:", ", ".join(f"{n}" for _, n in imp))
    return te, model


# --------------------------------------------------------------------------- current season + blending
def fetch_histories(bootstrap, get, min_minutes=90, cache_path="backtest_cache.json"):
    """{player_id: element-summary history rows}; same cache file as backtest.py (stamped by finished gameweeks)."""
    from concurrent.futures import ThreadPoolExecutor
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
    need = [e["id"] for e in bootstrap["elements"] if _f(e.get("minutes")) >= min_minutes and str(e["id"]) not in cache]

    def one(i):
        try:
            return i, get(f"element-summary/{i}/")["history"]
        except Exception:
            return i, None
    if need:
        print(f"Fetching histories for {len(need)} players...")
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, h in ex.map(one, need):
                if h is not None:
                    cache[str(i)] = h
        try:
            json.dump(cache, open(cache_path, "w"))
        except Exception:
            pass
    return {int(k): v for k, v in cache.items() if k != "_stamp"}


def past_features(seasons=SEASONS, cache_dir="ml_data"):
    return pd.concat([regulars(build_features(load_season(s, cache_dir))) for s in seasons], ignore_index=True)


def fit_final(bootstrap, hist, past=None):
    """Model trained on past seasons and every finished gameweek of this one."""
    past = past_features() if past is None else past
    cur = build_features(from_fpl_history(bootstrap, hist))
    return fit(pd.concat([past, regulars(cur)], ignore_index=True)), cur


def predict_upcoming(bootstrap, fixtures, hist, gws, model):
    """{(player_id, gw): ML points} for players who qualify as regulars."""
    base = from_fpl_history(bootstrap, hist)
    extra = [upcoming_rows(bootstrap, fixtures, g) for g in gws]
    f = build_features(pd.concat([base] + [e for e in extra if len(e)], ignore_index=True))
    f = f[f["pending"] & f["gw"].isin(gws)]
    f = regulars(f)
    if f.empty:
        return {}
    pred = model.predict(f[FEATURES])
    return {(int(pid.split(":")[1]), int(g)): float(v) for pid, g, v in zip(f["pid"], f["gw"], pred)}


def blend_projection(p, gws, ml, w, avail_fn):
    """Replace gw columns by w*ML*availability + (1-w)*physics for players the ML model covers."""
    p = p.copy()
    for i, g in enumerate(gws):
        vals, used = [], []
        for pid, r in p.iterrows():
            m = ml.get((int(pid), g))
            phys = float(r[f"gw{g}"])
            if m is None or w <= 0:
                vals.append(phys); used.append(False)
            else:
                vals.append(w * max(0.0, m) * avail_fn(r, i) + (1 - w) * phys); used.append(True)
        p[f"gw{g}"] = vals
        p[f"ml{g}"] = used
    weights = {g: 0.92 ** i for i, g in enumerate(gws)}
    p["ep_total"] = sum(p[f"gw{g}"] * wt for g, wt in weights.items())
    p["ep_next"] = p[f"gw{gws[0]}"]
    return p


def backtest(out_dir=".", start=4):
    """ML vs physics vs blend on the current season, walk-forward, then write ml_blend.json."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import backtest as bt
    import fpl_forecast as ff
    bs, fixtures = ff.get("bootstrap-static/"), ff.get("fixtures/")
    last = max((e["id"] for e in bs["events"] if e["finished"]), default=0)
    if last < start + 3:
        raise SystemExit(f"Only {last} gameweeks finished: need at least {start + 3}.")
    hist = bt.fetch_histories(bs, 90)
    print(f"Physics backtest GW{start}-GW{last}...")
    df, scored = bt.run(bs, fixtures, hist, start, last)
    print("Training ML on past seasons + this season so far (re-fitted each gameweek)...")
    past = past_features()
    cur = build_features(from_fpl_history(bs, hist))
    parts = []
    for g in sorted(scored["gw"].unique()):
        model = fit(pd.concat([past, regulars(cur[cur["gw"] < g])], ignore_index=True))
        d = regulars(cur[cur["gw"] == g]).copy()
        d["ml"] = model.predict(d[FEATURES])
        d["id"] = d["pid"].str.split(":").str[1].astype(int)
        parts.append(d[["gw", "id", "ml"]])
    ml = pd.concat(parts, ignore_index=True)
    sc = scored.merge(ml, on=["gw", "id"], how="inner").rename(columns={"actual": "points"})
    sc["pos"] = sc["pos"].map(POS)
    print(f"\n=== GW{sc['gw'].min()}-{sc['gw'].max()}, {len(sc)} player-games ===")
    grid = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
    res = {}
    for w in grid:
        sc[f"b{w}"] = w * sc["ml"] + (1 - w) * sc["pred_cal"]
        res[w] = metrics(sc, f"b{w}")
    show([("Physics (calibrated)", metrics(sc, "pred_cal")), ("ML only", metrics(sc, "ml")), ("Last-5 average", metrics(sc, "b_form")),
          ("Season average", metrics(sc, "b_season"))])
    print("\nBlend weight on ML -> MAE / rank corr / top-10")
    for w in grid:
        print(f"  w={w:.1f}  {res[w]['mae']:.3f}  {res[w]['rho']:.3f}  {res[w]['top']:.2f}")
    # pick the weight with the lowest MAE; prefer a smaller weight when the gain is negligible (<0.005)
    best = min(grid, key=lambda w: (round(res[w]["mae"] / 0.005), w))
    print(f"\nChosen ML weight: {best}")
    out = {"w": best, "gws": [int(sc["gw"].min()), int(sc["gw"].max())], "n": int(len(sc)),
           "physics": metrics(sc, "pred_cal"), "ml": metrics(sc, "ml"), "blend": res[best],
           "form": metrics(sc, "b_form"), "season": metrics(sc, "b_season")}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "ml_blend.json"), "w"), indent=1)
    print("Wrote ml_blend.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "backtest"])
    ap.add_argument("--test", default="2025-26")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    if a.cmd == "validate":
        validate([s for s in SEASONS if s < a.test], a.test)
    else:
        backtest(a.out_dir)
