#!/usr/bin/env python3
"""
FPL points forecaster + squad optimiser (v2).

What's new in v2
  * Team attack/defence strength is learned from THIS season's results
    (recency-weighted, opponent-adjusted, blended with team xG/xGC, shrunk
    towards FPL's official ratings early in the season).
  * Official FPL FDR is used as a light nudge (--fdr-weight), not the driver.
  * Rotation: last-5-match minutes (from element-summary) blended with season minutes.
  * Set pieces: penalty / corner order from the API gives a small boost.
  * News: API injury flags + your own news_overrides.json (press conferences,
    suspensions, rotation, long-term injuries).
  * Fixture ticker printed so you can see what the model thinks of each run.

Usage examples:
  python fpl_forecast.py --team-id 1234567
  python fpl_forecast.py --team-id 1234567 --free-transfers 2 --horizon 6
  python fpl_forecast.py --team-id 1234567 --bench-boost
  python fpl_forecast.py --team-id 1234567 --wildcard
  python fpl_forecast.py --squad-file my_squad.json
  python fpl_forecast.py --team-id 1234567 --no-deep        # skip per-player history (faster)
"""
import argparse
import json
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pulp
import requests

BASE = "https://fantasy.premierleague.com/api"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
DC_THRESHOLD = {1: 99, 2: 10, 3: 12, 4: 12}  # defensive-contribution thresholds (2 pts)
POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
CALIB = {}  # per-position scale factors from backtest.py (calibration.json); empty = no adjustment
PEN_ADD = {1: 0.04, 2: 0.01}      # extra expected goals per match for pen taker #1 / #2 (kept small:
                                  # per-90 xG already contains some penalty history)
CORNER_ADD = {1: 0.02, 2: 0.01}   # extra expected assists per match for corner taker #1 / #2


# --------------------------------------------------------------------------- data
def get(path):
    r = SESSION.get(f"{BASE}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def num(df, col, default=0.0):
    if col not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


# --------------------------------------------------------------------------- team strength
def team_strength(bootstrap, fixtures, prior_k=4.0, half_life=6.0, xg_blend=0.5):
    """
    Poisson attack/defence model fitted on finished matches.
      expected goals(team t vs o, venue) = base_venue * att[t] * dfn[o]
    att/dfn are relative to league average (1.0). Matches are recency weighted,
    team goals are blended with team xG / xGC, and the estimates are shrunk
    towards FPL's official strength ratings with `prior_k` pseudo-matches.
    """
    teams = pd.DataFrame(bootstrap["teams"]).set_index("id")
    ids = list(teams.index)

    def official(cols):
        v = pd.concat([pd.to_numeric(teams[c], errors="coerce") for c in cols], axis=1).mean(axis=1)
        return v.where(v > 0)

    off_att = official(["strength_attack_home", "strength_attack_away"])
    off_def = official(["strength_defence_home", "strength_defence_away"])
    att_prior = (off_att / off_att.mean()).fillna(1.0).to_dict() if off_att.notna().any() else {t: 1.0 for t in ids}
    def_prior = (off_def.mean() / off_def).fillna(1.0).to_dict() if off_def.notna().any() else {t: 1.0 for t in ids}

    played = [f for f in fixtures
              if f.get("team_h_score") is not None and f.get("team_a_score") is not None
              and (f.get("finished") or f.get("finished_provisional"))]
    if not played:
        return {"att": att_prior, "dfn": def_prior, "base_home": 1.5, "base_away": 1.2, "avg": 1.35,
                "n_games": {t: 0 for t in ids}, "ids": ids}

    base_home = max(0.5, float(np.mean([f["team_h_score"] for f in played])))
    base_away = max(0.5, float(np.mean([f["team_a_score"] for f in played])))
    avg = (base_home + base_away) / 2
    last_gw = max(f["event"] for f in played)

    # team xG / xGC vs actual goals -> blend factor (clamped, so tiny samples can't explode)
    el = pd.DataFrame(bootstrap["elements"])
    xg_tot = num(el, "expected_goals").groupby(el["team"]).sum()
    gks = el[el["element_type"] == 1]
    xgc_tot = num(gks, "expected_goals_conceded").groupby(gks["team"]).sum()

    M = {t: [] for t in ids}  # matches per team: (gf, ga, opp, is_home, weight)
    for f in played:
        w = 0.5 ** ((last_gw - f["event"]) / half_life)
        M[f["team_h"]].append((f["team_h_score"], f["team_a_score"], f["team_a"], True, w))
        M[f["team_a"]].append((f["team_a_score"], f["team_h_score"], f["team_h"], False, w))

    n_games = {t: len(M[t]) for t in ids}
    f_att, f_def = {}, {}
    for t in ids:
        gf = sum(m[0] for m in M[t])
        ga = sum(m[1] for m in M[t])
        ra = np.clip(xg_tot.get(t, 0) / gf, 0.6, 1.6) if gf > 0 and xg_tot.get(t, 0) > 0 else 1.0
        rd = np.clip(xgc_tot.get(t, 0) / ga, 0.6, 1.6) if ga > 0 and xgc_tot.get(t, 0) > 0 else 1.0
        f_att[t] = (1 - xg_blend) + xg_blend * ra
        f_def[t] = (1 - xg_blend) + xg_blend * rd

    att, dfn = dict(att_prior), dict(def_prior)
    for _ in range(30):
        new_att, new_dfn = {}, {}
        for t in ids:
            na, da = prior_k * att_prior[t] * avg, prior_k * avg
            nd, dd = prior_k * def_prior[t] * avg, prior_k * avg
            for gf, ga, o, home, w in M[t]:
                b_for = base_home if home else base_away
                b_opp = base_away if home else base_home
                na += w * gf * f_att[t]
                da += w * b_for * dfn[o]
                nd += w * ga * f_def[t]
                dd += w * b_opp * att[o]
            new_att[t], new_dfn[t] = na / da, nd / dd
        ma, md = np.mean(list(new_att.values())), np.mean(list(new_dfn.values()))
        att = {t: v / ma for t, v in new_att.items()}
        dfn = {t: v / md for t, v in new_dfn.items()}
    return {"att": att, "dfn": dfn, "base_home": base_home, "base_away": base_away, "avg": avg,
            "n_games": n_games, "ids": ids}


def team_fixtures(fixtures, gws, ids):
    """{team_id: {gw: [(opp_id, is_home, official_fdr), ...]}} - handles doubles and blanks."""
    out = {t: {g: [] for g in gws} for t in ids}
    for f in fixtures:
        g = f.get("event")
        if g in gws:
            out[f["team_h"]][g].append((f["team_a"], True, f.get("team_h_difficulty")))
            out[f["team_a"]][g].append((f["team_h"], False, f.get("team_a_difficulty")))
    return out


def fixture_factors(t, opp, home, S, fdr, fdr_w):
    """(attack multiplier for the team's attackers, expected goals the team concedes)."""
    base_for = S["base_home"] if home else S["base_away"]
    base_opp = S["base_away"] if home else S["base_home"]
    att_mult = (base_for / S["avg"]) * S["dfn"][opp]
    lam_against = base_opp * S["att"][opp] * S["dfn"][t]
    if fdr_w and fdr:
        adj = 1 + fdr_w * (3 - float(fdr)) / 3  # official FDR 1..5 -> small +/- nudge
        att_mult *= adj
        lam_against /= adj
    return att_mult, lam_against


# --------------------------------------------------------------------------- players
def load_overrides(path):
    if not path or not os.path.exists(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _norm(s):
    """lower-case, accent-free: 'Ekitiké' -> 'ekitike', 'E.Le Fée' -> 'e.le fee'."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def find_player(p, name, club=None):
    """Return (player_id, None) or (None, reason). Tries exact web_name, then the part after
    initials ('E.Le Fée' -> 'Le Fée'), then surname; accents and case are ignored."""
    target = _norm(name)
    web = p["web_name"].map(_norm)
    tiers = [web == target, web.str.split(".").str[-1].str.strip() == target]
    if "second_name" in p:
        tiers.append(p["second_name"].map(_norm) == target)
    tiers.append(web.str.contains(target, regex=False))  # last resort, e.g. 'Gibbs' in 'Gibbs-White'
    for mask in tiers:
        m = p[mask]
        if club:
            m = m[m["team_short"] == club.strip().upper()]
        if len(m) == 1:
            return m.index[0], None
        if len(m) > 1:
            who = ", ".join(f"{r.web_name} ({r.team_short})" for r in m.head(4).itertuples())
            return None, f"{len(m)} players match ({who}) - use 'Name:CLUB'"
    return None, "no player found - check the spelling against projections.csv"


_WARNED = set()


def apply_overrides(p, overrides, short):
    p["minutes_mult"] = 1.0
    p["out_gws"] = 0
    p["avail_override"] = False
    p["note"] = ""
    p["team_short"] = p["team"].map(short)
    for key, val in overrides.items():
        name, _, club = key.partition(":")
        i, why = find_player(p, name, club or None)
        if i is None:
            if key not in _WARNED:
                print(f"  [news] '{key}' skipped: {why}")
                _WARNED.add(key)
            continue
        if "avail" in val:
            p.loc[i, "avail"] = float(val["avail"])
            p.loc[i, "avail_override"] = True
        if "out_gws" in val:
            p.loc[i, "out_gws"] = int(val["out_gws"])
        if "minutes_mult" in val:
            p.loc[i, "minutes_mult"] = float(val["minutes_mult"])
        if val.get("pen_taker"):
            p.loc[i, "pen_order"] = 1
        if "note" in val:
            p.loc[i, "note"] = str(val["note"])
    return p


def player_table(bootstrap, S, overrides, recent):
    p = pd.DataFrame(bootstrap["elements"]).set_index("id")
    short = pd.DataFrame(bootstrap["teams"]).set_index("id")["short_name"].to_dict()
    p["price"] = num(p, "now_cost") / 10
    p["xg90"] = num(p, "expected_goals_per_90")
    p["xa90"] = num(p, "expected_assists_per_90")
    p["dc90"] = num(p, "defensive_contribution_per_90")
    p["form"] = num(p, "form")
    p["pen_order"] = num(p, "penalties_order")
    p["corner_order"] = num(p, "corners_and_indirect_freekicks_order")
    minutes = num(p, "minutes")

    games = p["team"].map(S["n_games"]).fillna(0)
    season = (minutes / (games.clip(lower=1) * 90)).clip(0, 1)
    season = season.where(games > 0, 0.8)
    rec = pd.Series({i: (np.mean(m) / 90 if len(m) >= 3 else np.nan) for i, m in recent.items()}, dtype=float)
    rec = rec.reindex(p.index)
    p["season_share"] = season
    p["recent_share"] = rec
    p["share"] = np.where(rec.notna(), 0.4 * season + 0.6 * rec.fillna(0), season).clip(0, 1)

    chance = pd.to_numeric(p["chance_of_playing_next_round"], errors="coerce") if "chance_of_playing_next_round" in p else pd.Series(np.nan, index=p.index)
    avail = chance.fillna(100) / 100
    avail[(chance.isna()) & (p["status"].isin(["i", "s", "u", "n"]))] = 0.0
    avail[(chance.isna()) & (p["status"] == "d")] = 0.5
    p["avail"] = avail
    p["news"] = p["news"].fillna("") if "news" in p else ""
    return apply_overrides(p, overrides, short)


def avail_for_gw(r, i):
    """Availability in the i-th gameweek of the horizon (0 = next GW)."""
    if r.out_gws > 0:
        return 0.0 if i < r.out_gws else 0.9
    base = r.avail
    if base >= 1:
        return 1.0
    if not r.avail_override and r.status == "s":
        return 0.0 if i == 0 else 1.0
    if not r.avail_override and r.status in ("i", "u", "n") and base == 0:
        return min(0.5, 0.15 * i)  # unknown-length injury: slow, partial recovery
    return base + (1 - base) * min(1.0, i / 3)  # genuine doubt: recovers quickly


def fixture_points(r, att_mult, lam_against, avail):
    pos = r.element_type
    share = min(1.0, r.share * r.minutes_mult)
    pplay = avail * min(1.0, share * 1.2)
    p60 = pplay * min(1.0, share / 0.8)
    mins = p60 + (pplay - p60) * 0.3  # fraction of 90 played

    xg = r.xg90 * att_mult * mins + PEN_ADD.get(int(r.pen_order), 0) * pplay
    xa = r.xa90 * att_mult * mins + CORNER_ADD.get(int(r.corner_order), 0) * pplay
    cs_prob = np.exp(-lam_against) * p60

    pts = pplay + p60  # appearance points (1 for <60, 2 for 60+)
    pts += xg * GOAL_PTS[pos] + xa * 3
    pts += cs_prob * CS_PTS[pos]
    if pos <= 2:
        pts -= (lam_against / 2) * p60  # -1 per 2 goals conceded
    if pos == 1:
        pts += 1.0 * p60  # rough save points
    pts += 0.8 * (xg + xa)  # bonus from attacking returns
    if pos <= 2:
        pts += 0.3 * cs_prob  # bonus from clean sheets
    p_dc = min(1.0, r.dc90 / DC_THRESHOLD[pos]) * 0.7 if pos != 1 else 0.0
    pts += 2 * p_dc * p60  # defensive contribution points (heuristic)

    return 0.8 * pts + 0.2 * r.form * pplay  # blend with recent form


def build_ticker(tf, S, short, gws):
    """One row per team: opponents with official FDR / model FDR, sorted easiest first."""
    nets = {}
    for t, d in tf.items():
        for g in gws:
            for opp, home, fdr in d[g]:
                bf = S["base_home"] if home else S["base_away"]
                bo = S["base_away"] if home else S["base_home"]
                lam_for = bf * S["att"][t] * S["dfn"][opp]
                lam_ag = bo * S["att"][opp] * S["dfn"][t]
                nets[(t, g, opp)] = np.log(lam_for / lam_ag)
    if not nets:
        return []
    cuts = np.quantile(list(nets.values()), [0.2, 0.4, 0.6, 0.8])  # low net = hard
    rows = []
    for t, d in tf.items():
        cells, tot = [], 0.0
        for g in gws:
            if not d[g]:
                cells.append("BLANK")
                continue
            parts = []
            for opp, home, fdr in d[g]:
                n = nets[(t, g, opp)]
                mf = 5 - int(np.searchsorted(cuts, n))  # 1 easiest .. 5 hardest
                tot += n
                parts.append(f"{short[opp]}({'H' if home else 'A'}){fdr if fdr else '?'}/{mf}")
            cells.append("+".join(parts))
        rows.append((tot, short[t], cells))
    rows.sort(reverse=True)
    return rows


def forecast(bootstrap, fixtures, horizon, S, overrides, recent, fdr_w):
    nxt = next((e["id"] for e in bootstrap["events"] if e.get("is_next")), None)
    if nxt is None:
        nxt = next(e["id"] for e in bootstrap["events"] if not e["finished"])
    gws = [g for g in range(nxt, nxt + horizon) if g <= 38]

    tf = team_fixtures(fixtures, gws, S["ids"])
    p = player_table(bootstrap, S, overrides, recent)
    short = pd.DataFrame(bootstrap["teams"]).set_index("id")["short_name"].to_dict()

    for i, g in enumerate(gws):
        vals, fx_txt = [], []
        for pid, r in p.iterrows():
            a = avail_for_gw(r, i)
            total, txt = 0.0, []
            for opp, home, fdr in tf[r.team][g]:
                am, lam = fixture_factors(r.team, opp, home, S, fdr, fdr_w)
                total += fixture_points(r, am, lam, a)
                txt.append(f"{short[opp]}({'H' if home else 'A'})")
            vals.append(total * CALIB.get(POS_NAME[int(r.element_type)], 1.0))
            fx_txt.append("+".join(txt) if txt else "BLANK")
        p[f"gw{g}"] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        p[f"fx{g}"] = fx_txt

    weights = {g: 0.92 ** i for i, g in enumerate(gws)}
    p["ep_total"] = sum(p[f"gw{g}"] * w for g, w in weights.items())
    p["ep_next"] = p[f"gw{gws[0]}"]
    p["pos"] = p["element_type"].map(POS_NAME)
    p["avail_now"] = [avail_for_gw(r, 0) for _, r in p.iterrows()]  # what the model assumes for the next GW
    return p, gws, build_ticker(tf, S, short, gws)


def fetch_recent(ids, stamp, n=5, cache_path="history_cache.json"):
    """Last n gameweek minutes per player (rotation signal). Cached per finished-GW count."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    if cache.get("_stamp") != stamp:
        cache = {}
    need = [i for i in ids if str(i) not in cache]

    def one(i):
        try:
            return i, [x["minutes"] for x in get(f"element-summary/{i}/")["history"]][-n:]
        except Exception:
            return i, None

    if need:
        print(f"Fetching recent minutes for {len(need)} players...")
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, m in ex.map(one, need):
                if m is not None:
                    cache[str(i)] = m
    cache["_stamp"] = stamp
    try:
        json.dump(cache, open(cache_path, "w"))
    except Exception:
        pass
    return {int(k): v for k, v in cache.items() if k != "_stamp"}




# --------------------------------------------------------------------------- squad / chips / free transfers
CHIP_LABEL = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
# Rough points you give up by spending a chip now instead of saving it for a better gameweek
# (e.g. a double or blank). Scale them with --chip-cost-scale (0 = treat chips as free).
CHIP_COST = {"wildcard": 12.0, "freehit": 15.0, "bboost": 12.0, "3xc": 8.0}


def load_entry(team_id, bootstrap):
    """Current squad, bank, chips already used, history, and the latest gameweek with data."""
    hist = get(f"entry/{team_id}/history/")
    cur_ev = next((e["id"] for e in bootstrap["events"] if e.get("is_current")), None)
    if cur_ev is None:
        fin = [e["id"] for e in bootstrap["events"] if e["finished"]]
        cur_ev = max(fin) if fin else None
    if cur_ev is None:
        sys.exit("No gameweek has been played yet - use --squad-file instead.")
    chips_used = hist.get("chips", [])
    used_ev = cur_ev
    if cur_ev > 1 and any(c["name"] == "freehit" and c["event"] == cur_ev for c in chips_used):
        used_ev = cur_ev - 1  # a Free Hit squad reverts, so plan from the squad before it
    picks = get(f"entry/{team_id}/event/{used_ev}/picks/")
    ids = [x["element"] for x in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10
    return ids, bank, chips_used, hist, cur_ev


def estimate_free_transfers(hist, upto_ev, max_ft=5):
    """Free transfers available for the gameweek after `upto_ev`, simulated from your transfer history.
    The FPL API does not publish this number, so it is an estimate - check it in the app."""
    chips = {c["event"]: c["name"] for c in hist.get("chips", [])}
    cur = {e["event"]: e for e in hist.get("current", [])}
    ft = 1
    for g in range(2, upto_ev + 1):
        made = cur.get(g, {}).get("event_transfers", 0)
        if chips.get(g) in ("wildcard", "freehit"):
            ft = min(ft + 1, max_ft)  # chip weeks neither use nor lose banked transfers
        else:
            ft = min(max(ft - made, 0) + 1, max_ft)
    return ft


def chip_windows(bootstrap):
    out = [{"name": c["name"], "start": c.get("start_event", 1), "stop": c.get("stop_event", 38)}
           for c in bootstrap.get("chips", []) if c.get("name") in CHIP_LABEL]
    return out or [{"name": n, "start": 1, "stop": 38} for n in CHIP_LABEL]


def parse_chip_windows(arg, bootstrap, chips_used):
    """Which chips can still be played, and in which gameweek windows."""
    arg = (arg or "auto").strip().lower()
    if arg == "none":
        return []
    wins = chip_windows(bootstrap)
    if arg == "all" or (arg == "auto" and chips_used is None):
        return wins
    if arg == "auto":
        return [w for w in wins if not any(h["name"] == w["name"] and w["start"] <= h["event"] <= w["stop"]
                                           for h in chips_used)]
    wanted = [a.strip() for a in arg.split(",") if a.strip()]
    bad = [a for a in wanted if a not in CHIP_LABEL]
    if bad:
        sys.exit(f"Unknown chip(s) {bad}. Use: {', '.join(CHIP_LABEL)}, or none / all / auto.")
    return [{"name": n, "start": 1, "stop": 38} for n in wanted]


def squad_from_file(path, players):
    data = json.load(open(path, encoding="utf-8"))
    ids = []
    for name in data["players"]:
        nm, _, club = name.partition(":")
        i, why = find_player(players, nm, club or None)
        if i is None:
            sys.exit(f"Could not match '{name}': {why}")
        ids.append(i)
    return ids, float(data.get("bank", 0.0))


# --------------------------------------------------------------------------- planner
def candidate_pool(p, gws, cur_ids):
    """Shortlist of players the planner may pick (keeps the optimisation fast)."""
    d = p.copy()
    d["plan_sum"] = sum(d[f"gw{g}"] for g in gws)
    keep = set(cur_ids or [])
    cheap = {1: 4.5, 2: 4.5, 3: 5.0, 4: 5.0}
    for pos, n in {1: 8, 2: 30, 3: 34, 4: 20}.items():
        sub = d[d["element_type"] == pos]
        keep |= set(sub.nlargest(n, "plan_sum").index)
        keep |= set(sub[sub["price"] <= cheap[pos]].nlargest(6, "plan_sum").index)  # budget-friendly bench options
        for g in gws:
            keep |= set(sub.nlargest(max(3, n // 4), f"gw{g}").index)
    return d.loc[sorted(keep)]


def plan(p, gws, cur_ids, bank, ft0, windows, bench_w=0.1, disc=0.95, ft_value=1.0,
         max_ft=5, force_chip=None, time_limit=60, chip_cost=None):
    """
    Multi-gameweek planner (one MILP): squad, XI, captain, transfers, banked free transfers,
    hits, and chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain).
    Chips are only used if present in `windows` (list of {name,start,stop}).
    """
    pool = candidate_pool(p, gws, cur_ids)
    ids = list(pool.index)
    T = len(gws)
    R = range(T)
    cur_set = set(cur_ids or [])
    fresh = not cur_set
    budget = 100.0 if fresh else float(p.loc[list(cur_set), "price"].sum() + bank)
    ep = {(t, i): float(pool.at[i, f"gw{gws[t]}"]) for t in R for i in ids}
    price, pos, team = pool["price"].to_dict(), pool["element_type"].to_dict(), pool["team"].to_dict()
    by_pos = {k: [i for i in ids if pos[i] == k] for k in (1, 2, 3, 4)}
    by_team = {}
    for i in ids:
        by_team.setdefault(team[i], []).append(i)
    names = list(CHIP_LABEL)
    chip_cost = chip_cost or {}

    def bv(*k):
        return pulp.LpVariable("_".join(map(str, k)), cat="Binary")

    def cv(*k):
        return pulp.LpVariable("_".join(map(str, k)), lowBound=0, upBound=1)

    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    x = {(t, i): bv("x", t, i) for t in R for i in ids}      # in squad
    s = {(t, i): bv("s", t, i) for t in R for i in ids}      # in starting XI
    c = {(t, i): bv("c", t, i) for t in R for i in ids}      # captain
    tin = {(t, i): cv("tin", t, i) for t in R for i in ids}  # transfer in
    z = {(t, i): cv("z", t, i) for t in R for i in ids}      # bench player counted (Bench Boost)
    ctc = {(t, i): cv("ctc", t, i) for t in R for i in ids}  # captain tripled (Triple Captain)
    hits = {t: pulp.LpVariable(f"hits_{t}", lowBound=0, upBound=15, cat="Integer") for t in R}
    ft = {0: ft0}
    for t in range(1, T + 1):
        ft[t] = pulp.LpVariable(f"ft_{t}", lowBound=1, upBound=max_ft, cat="Integer")
    ch = {(n, t): bv("ch", n, t) for n in names for t in R}

    covers = lambda w, t: w["start"] <= gws[t] <= w["stop"]
    for n in names:
        for t in R:
            if not any(w["name"] == n and covers(w, t) for w in windows):
                prob += ch[(n, t)] == 0
    for w in windows:
        ts = [t for t in R if covers(w, t)]
        if ts:
            prob += pulp.lpSum(ch[(w["name"], t)] for t in ts) <= 1  # each chip at most once per window
    if force_chip:
        prob += ch[(force_chip, 0)] == 1

    fh_possible = any(w["name"] == "freehit" and any(covers(w, t) for t in R) for w in windows)
    # b = the squad you carry into the next week (differs from x only in a Free Hit week)
    b = {(t, i): bv("b", t, i) for t in R for i in ids} if fh_possible else x

    def base_prev(t, i):
        return (1 if i in cur_set else 0) if t == 0 else b[(t - 1, i)]

    for t in R:
        cw = ch[("wildcard", t)] + ch[("freehit", t)]
        prob += pulp.lpSum(ch[(n, t)] for n in names) <= 1
        prob += pulp.lpSum(x[(t, i)] for i in ids) == 15
        prob += pulp.lpSum(s[(t, i)] for i in ids) == 11
        prob += pulp.lpSum(c[(t, i)] for i in ids) == 1
        prob += pulp.lpSum(price[i] * x[(t, i)] for i in ids) <= budget
        for i in ids:
            prob += s[(t, i)] <= x[(t, i)]
            prob += c[(t, i)] <= s[(t, i)]
            prob += z[(t, i)] <= x[(t, i)] - s[(t, i)]
            prob += z[(t, i)] <= ch[("bboost", t)]
            prob += ctc[(t, i)] <= c[(t, i)]
            prob += ctc[(t, i)] <= ch[("3xc", t)]
            prob += tin[(t, i)] >= x[(t, i)] - base_prev(t, i)
        for k, n in {1: 2, 2: 5, 3: 5, 4: 3}.items():
            prob += pulp.lpSum(x[(t, i)] for i in by_pos[k]) == n
        for k, (lo, hi) in {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}.items():
            prob += pulp.lpSum(s[(t, i)] for i in by_pos[k]) >= lo
            prob += pulp.lpSum(s[(t, i)] for i in by_pos[k]) <= hi
        for members in by_team.values():
            prob += pulp.lpSum(x[(t, i)] for i in members) <= 3
        n_tr = pulp.lpSum(tin[(t, i)] for i in ids)
        free_all = 15 * cw + (15 if (fresh and t == 0) else 0)
        prob += hits[t] >= n_tr - ft[t] - free_all
        prob += ft[t + 1] <= ft[t] - n_tr + hits[t] + 1 + free_all
        prob += ft[t + 1] <= ft[t] + 1
        if fh_possible:
            fh = ch[("freehit", t)]
            for i in ids:
                prob += b[(t, i)] <= x[(t, i)] + fh
                prob += b[(t, i)] >= x[(t, i)] - fh
                prob += b[(t, i)] <= base_prev(t, i) + 1 - fh
                prob += b[(t, i)] >= base_prev(t, i) - 1 + fh

    obj = []
    for t in R:
        w_t = disc ** t
        for i in ids:
            obj.append(w_t * ep[(t, i)] * (s[(t, i)] + bench_w * (x[(t, i)] - s[(t, i)]) + c[(t, i)]
                                           + ctc[(t, i)] + (1 - bench_w) * z[(t, i)]))
        obj.append(-4 * w_t * hits[t])
    obj.append(ft_value * ft[T])  # a banked transfer has some value beyond the horizon
    for (n, t), v in ch.items():  # spending a chip has an opportunity cost
        obj.append(-chip_cost.get(n, 0.0) * disc ** t * v)
    prob += pulp.lpSum(obj)

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.002))
    if x[(0, ids[0])].value() is None:
        sys.exit(f"Planner found no solution ({pulp.LpStatus[prob.status]}).")

    # ---- read the plan back and recompute points / hits with the real FPL rules
    weeks, base, ft_now = [], set(cur_set), ft0
    for t in R:
        squad = [i for i in ids if x[(t, i)].value() > 0.5]
        xi = [i for i in ids if s[(t, i)].value() > 0.5]
        cap = next(i for i in ids if c[(t, i)].value() > 0.5)
        chip = next((n for n in names if ch[(n, t)].value() > 0.5), None)
        ins = [i for i in squad if i not in base]
        outs = [i for i in base if i not in squad]
        free_week = chip in ("wildcard", "freehit") or (fresh and t == 0)
        n_tr = 0 if (fresh and t == 0) else len(ins)
        hit = 0 if free_week else max(0, n_tr - ft_now)
        bench = [i for i in squad if i not in xi]
        pts = sum(ep[(t, i)] for i in xi) + ep[(t, cap)] * (2 if chip == "3xc" else 1)
        if chip == "bboost":
            pts += sum(ep[(t, i)] for i in bench)
        weeks.append({"gw": gws[t], "chip": chip, "ft": ft_now, "n_tr": n_tr, "hits": hit, "squad": squad,
                      "xi": xi, "bench": bench, "cap": cap, "ins": ins, "outs": outs,
                      "pts": pts, "net": pts - 4 * hit, "hits_milp": int(round(hits[t].value()))})
        ft_now = min(ft_now + 1, max_ft) if free_week else min(max(ft_now - n_tr, 0) + 1, max_ft)
        if chip != "freehit":
            base = set(squad)
    spent = sum(chip_cost.get(w["chip"], 0.0) for w in weeks if w["chip"])
    return {"weeks": weeks, "total_net": sum(w["net"] for w in weeks), "ft_after": ft_now, "chip_cost": spent}


# --------------------------------------------------------------------------- output
POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _xi_table(p, ids, col, cap):
    d = p.loc[ids].copy()
    d["player"] = [n + (" (C)" if i == cap else "") for i, n in zip(d.index, d["web_name"])]
    d["pts"] = d[col]
    d["flag"] = [(str(n)[:32] if n else "") or str(note)[:32] for n, note in zip(d["news"], d["note"])]
    d["_o"] = d["pos"].map(POS_ORDER)
    d = d.sort_values(["_o", "pts"], ascending=[True, False])
    return d[["player", "pos", "team_short", "price", "pts", "flag"]].round(2).to_string(index=False)


def print_plan(p, res, title):
    print(f"\n{'=' * 6} {title} {'=' * 6}")
    for w in res["weeks"]:
        chip = CHIP_LABEL.get(w["chip"], "none")
        cost = f"hits {w['hits']} (-{4 * w['hits']})"
        print(f"\n--- GW{w['gw']} | chip: {chip} | free transfers: {w['ft']} | transfers made: {w['n_tr']} | {cost}"
              f" | projected {w['pts']:.1f}, net {w['net']:.1f}")
        if w["ins"] and w["n_tr"]:
            key = lambda i: p.loc[i, "element_type"]
            for o, n in zip(sorted(w["outs"], key=key), sorted(w["ins"], key=key)):
                print(f"    OUT {p.loc[o, 'web_name']:<14} -> IN {p.loc[n, 'web_name']:<14}")
        col = f"gw{w['gw']}"
        print(_xi_table(p, w["xi"], col, w["cap"]))
        print("  Bench:")
        print(_xi_table(p, w["bench"], col, None))
    print(f"\nTotal projected net over {len(res['weeks'])} GWs: {res['total_net']:.1f}")


def show_ticker(rows, gws):
    print("\n=== Fixture ticker (easiest run first) ===")
    print("cell = OPP(venue) official-FDR / model-FDR   (1 = easiest, 5 = hardest)")
    head = f"{'TEAM':<5}" + "".join(f"{'GW' + str(g):<17}" for g in gws)
    print(head)
    for _, team, cells in rows:
        print(f"{team:<5}" + "".join(f"{c:<17}" for c in cells))


def load_calibration(path):
    """Read calibration.json written by backtest.py. 'none' or a missing file means no adjustment."""
    CALIB.clear()
    if not path or str(path).lower() == "none" or not os.path.exists(path):
        return {}
    try:
        CALIB.update({k: float(v) for k, v in json.load(open(path))["scale"].items()})
    except Exception as e:
        print(f"  [calibration] could not read {path}: {e}")
    return dict(CALIB)


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v else 0.0  # NaN -> 0


def export_site(p, gws, ticker, bootstrap, out_dir, accuracy_path="accuracy.json"):
    """Write projections.json for the website: every player's projected points per gameweek, the fixture
    ticker (for the heatmap) and the chip windows. The page does the team-specific maths in the browser."""
    import datetime as dt
    os.makedirs(out_dir, exist_ok=True)
    short = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    players = []  # everyone is exported, so a visitor's injured or benched players are always found
    for pid, r in p.iterrows():
        players.append({
            "id": int(pid), "n": str(r["web_name"]), "t": short[int(r["team"])], "p": str(r["pos"]),
            "c": round(float(r["price"]), 1), "a": round(float(r["avail_now"]), 2),
            "s": str(r["status"]), "w": str(r.get("news", "") or "")[:90],
            "ep": [round(float(r[f"gw{g}"]), 2) for g in gws],
            "o": round(_f(r.get("selected_by_percent")), 1),
            "ti": int(_f(r.get("transfers_in_event"))), "to": int(_f(r.get("transfers_out_event"))),
            "m": int(_f(r.get("minutes"))),
            "fe": round(_f(r.get("ep_next")), 2), "fo": round(_f(r.get("form")), 1),
            "gi": int(_f(r.get("goals_scored")) + _f(r.get("assists"))),
            "xgi": round(_f(r.get("expected_goals")) + _f(r.get("expected_assists")), 1),
        })
    nxt = next((e for e in bootstrap["events"] if e["id"] == gws[0]), {})
    data = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gws": gws,
        "deadline": nxt.get("deadline_time"),
        "chips": chip_windows(bootstrap),
        "ticker": [{"t": t, "c": cells} for _, t, cells in ticker],
        "tp": int(bootstrap.get("total_players") or 0),
        "players": players,
    }
    for key, name in (("accuracy", accuracy_path), ("tracker", "tracker.json")):
        for pth in (name, os.path.join(out_dir, os.path.basename(name))):
            if pth and os.path.exists(pth):
                try:
                    data[key] = json.load(open(pth))
                    break
                except Exception:
                    pass
    if CALIB:
        data["calibrated"] = True
    path = os.path.join(out_dir, "projections.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"Exported {len(players)} players, GW{gws[0]}-GW{gws[-1]} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-id", type=int, help="your FPL team id (the number in your FPL URL)")
    ap.add_argument("--squad-file", help="JSON file: {\"players\": [...web_names...], \"bank\": 1.0}")
    ap.add_argument("--plan-gws", type=int, default=3, help="gameweeks to plan together (default 3)")
    ap.add_argument("--horizon", type=int, default=5, help="gameweeks to project (>= plan-gws)")
    ap.add_argument("--free-transfers", default="auto", help="'auto' (estimate from history) or a number")
    ap.add_argument("--chips", default="auto",
                    help="auto | none | all | comma list of wildcard,freehit,bboost,3xc that you still have")
    ap.add_argument("--max-ft", type=int, default=5, help="max banked free transfers (default 5)")
    ap.add_argument("--ft-value", type=float, default=1.0, help="points value of a banked transfer after the plan")
    ap.add_argument("--chip-cost-scale", type=float, default=1.0,
                    help="opportunity cost of spending a chip now (1 = default costs, 0 = chips are free)")
    ap.add_argument("--disc", type=float, default=0.95, help="weekly discount on later gameweeks")
    ap.add_argument("--time-limit", type=int, default=60, help="seconds per optimisation")
    ap.add_argument("--news", default="news_overrides.json", help="your team-news overrides file")
    ap.add_argument("--fdr-weight", type=float, default=0.10, help="nudge from official FDR (0 = ignore)")
    ap.add_argument("--prior-k", type=float, default=4.0, help="pseudo-matches of official-rating prior")
    ap.add_argument("--half-life", type=float, default=6.0, help="recency half-life in gameweeks")
    ap.add_argument("--no-deep", action="store_true", help="skip per-player recent-minutes download")
    ap.add_argument("--deep-n", type=int, default=150, help="how many top players get a rotation check")
    ap.add_argument("--no-lists", action="store_true", help="skip the top-10-per-position lists")
    ap.add_argument("--calibration", default="calibration.json",
                    help="per-position scale factors from backtest.py ('none' to switch off)")
    ap.add_argument("--export-dir", help="write projections.json for the website into this folder")
    ap.add_argument("--export-only", action="store_true", help="with --export-dir: export and stop (no planning)")
    args = ap.parse_args()

    print("Fetching FPL data...")
    bootstrap, fixtures = get("bootstrap-static/"), get("fixtures/")
    overrides = load_overrides(args.news)
    if overrides:
        print(f"Loaded {len(overrides)} news overrides from {args.news}")
    S = team_strength(bootstrap, fixtures, args.prior_k, args.half_life)
    cal = load_calibration(args.calibration)
    if cal:
        print("Calibration from backtest applied: " + ", ".join(f"{k} x{v:.2f}" for k, v in cal.items()))

    cur, bank, chips_used, hist, cur_ev = None, 0.0, None, None, None
    if args.team_id:
        cur, bank, chips_used, hist, cur_ev = load_entry(args.team_id, bootstrap)

    horizon = max(args.horizon, args.plan_gws)
    p, gws, ticker = forecast(bootstrap, fixtures, horizon, S, overrides, {}, args.fdr_weight)
    if args.squad_file:
        cur, bank = squad_from_file(args.squad_file, p)

    if not args.no_deep:
        top = list(p.nlargest(args.deep_n, "ep_total").index)
        ids = sorted(set(top) | set(cur or []))
        finished = sum(1 for e in bootstrap["events"] if e["finished"])
        recent = fetch_recent(ids, stamp=f"fin{finished}")
        p, gws, ticker = forecast(bootstrap, fixtures, horizon, S, overrides, recent, args.fdr_weight)

    p.drop(columns=[c for c in p.columns if c.startswith("fx")]).round(3).to_csv("projections.csv")
    print(f"Projected GW{gws[0]}-GW{gws[-1]}. Saved projections.csv")
    if args.export_dir:
        export_site(p, gws, ticker, bootstrap, args.export_dir)
        if args.export_only:
            return

    # ---- assumptions
    if str(args.free_transfers).lower() == "auto":
        if hist:
            ft0 = estimate_free_transfers(hist, cur_ev, args.max_ft)
            ft_note = "estimated from your transfer history - confirm in the FPL app, override with --free-transfers N"
        else:
            ft0, ft_note = 1, "default (no team id) - set --free-transfers N"
    else:
        ft0, ft_note = int(args.free_transfers), "as given"
    windows = parse_chip_windows(args.chips, bootstrap, chips_used)
    plan_gws = gws[:args.plan_gws]
    costs = {n: v * args.chip_cost_scale for n, v in CHIP_COST.items()}
    kw = dict(disc=args.disc, ft_value=args.ft_value, max_ft=args.max_ft, time_limit=args.time_limit,
              chip_cost=costs)

    print("\n=== Assumptions ===")
    if cur:
        print(f"Squad: {len(cur)} players, bank £{bank:.1f}m (the squad from GW{cur_ev}; transfers already made for "
              f"GW{gws[0]} are not seen - use --squad-file if you have made some)")
    else:
        print("No squad given: building the best fresh 15-man squad (Wildcard-style).")
    print(f"Free transfers for GW{gws[0]}: {ft0} ({ft_note})")
    if chips_used:
        print("Chips already used: " + ", ".join(f"{CHIP_LABEL.get(c['name'], c['name'])} (GW{c['event']})"
                                                for c in sorted(chips_used, key=lambda c: c["event"])))
    elif chips_used is not None:
        print("Chips already used: none")
    if windows:
        print("Chips still available (and the gameweeks each can be played in):")
        for w in sorted(windows, key=lambda w: (w["start"], w["name"])):
            print(f"  {CHIP_LABEL[w['name']]:<15} GW{w['start']}-GW{w['stop']}")
    else:
        print("Chips available: none - only free transfers (and paid -4 hits) can improve the team")
    usable = [w for w in windows if any(w["start"] <= g <= w["stop"] for g in plan_gws)]
    if windows and not usable:
        print(f"None of your remaining chips can be played in GW{plan_gws[0]}-GW{plan_gws[-1]}, "
              "so this plan uses free transfers only. If that looks wrong, check the Chips page in the FPL app "
              "and override with --chips.")

    # ---- A) this week only
    g0 = plan_gws[0]
    print(f"\n########## A) BEST XI FOR GW{g0} (this week only) ##########")
    covered = [n for n in CHIP_LABEL if any(w["name"] == n and w["start"] <= g0 <= w["stop"] for w in windows)]
    options = []
    for opt in [None] + covered:
        ws = [w for w in windows if w["name"] == opt] if opt else []
        options.append((opt, plan(p, [g0], cur, bank, ft0, ws, force_chip=opt, **kw)))
    if not covered:
        print(f"No chip can be played in GW{g0}" + (" (your remaining chips unlock later)." if windows else
                                                      " (you have none left)."))
    base_net = options[0][1]["total_net"]
    print(f"\n{'Option':<16}{'Projected':>10}{'Hits':>6}{'Net':>8}{'Gain':>8}{'Chip cost':>11}{'Worth it?':>11}")
    for opt, r in options:
        w = r["weeks"][0]
        gain = w["net"] - base_net
        cost = costs.get(opt, 0.0)
        verdict = "-" if opt is None else ("yes" if gain > cost else "no")
        print(f"{CHIP_LABEL.get(opt, 'No chip'):<16}{w['pts']:>10.1f}{w['hits']:>6}{w['net']:>8.1f}{gain:>+8.1f}{cost:>11.1f}{verdict:>11}")
    best = max(options, key=lambda o: o[1]["total_net"] - costs.get(o[0], 0.0))
    print("\nGain = extra projected points this week vs no chip. Chip cost = rough points lost by spending the chip "
          "now instead of saving it (override with --chip-cost-scale). Wildcard / Free Hit look better in a one-week "
          "view than they are; Section B weighs timing across the next gameweeks.")
    print_plan(p, options[0][1], f"GW{g0} XI - no chip (free transfers only)")
    if best[0] and (best[1]["total_net"] - base_net) > costs.get(best[0], 0.0):
        print_plan(p, best[1], f"GW{g0} XI - best with a chip: {CHIP_LABEL[best[0]]}")

    # ---- B) multi-week plan
    print(f"\n########## B) PLAN FOR GW{plan_gws[0]}-GW{plan_gws[-1]} ##########")
    no_chip = plan(p, plan_gws, cur, bank, ft0, [], **kw)
    print_plan(p, no_chip, "Plan with free transfers only (no chips)" if usable else
               "Plan (free transfers only - no chip is playable in these gameweeks)")
    if usable:
        with_chips = plan(p, plan_gws, cur, bank, ft0, usable, **kw)
        print_plan(p, with_chips, "Plan with the best chip timing")
        used = [(w["gw"], CHIP_LABEL[w["chip"]]) for w in with_chips["weeks"] if w["chip"]]
        gain = with_chips["total_net"] - no_chip["total_net"]
        print("\n=== Chip verdict ===")
        print("Chips used: " + (", ".join(f"GW{g} {n}" for g, n in used) if used else "none worth playing in this window"))
        print(f"Net points: {no_chip['total_net']:.1f} without chips vs {with_chips['total_net']:.1f} with "
              f"({gain:+.1f}; chip opportunity cost counted: {with_chips['chip_cost']:.1f}).")
        if not used:
            print("No chip beats the value of saving it - keep them for a double or blank gameweek.")

    # ---- context
    if cur:
        flagged = p.loc[cur]
        flagged = flagged[(flagged["avail_now"] < 1) | (flagged["news"] != "") | (flagged["note"] != "")]
        if len(flagged):
            print("\nYour current squad - news flags:")
            for i, r in flagged.iterrows():
                print(f"  {r['web_name']:<14} {r['team_short']}  avail {r['avail_now']:.2f}  {r['news'] or r['note']}")
    show_ticker(ticker, gws)
    if not args.no_lists:
        print("\nTop 10 by projected points per position:")
        for pos in ("GK", "DEF", "MID", "FWD"):
            top = p[p["pos"] == pos].nlargest(10 if pos != "GK" else 5, "ep_total")
            print(f"\n{pos}")
            print(top[["web_name", "team_short", "price", "ep_total", f"fx{gws[0]}"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
