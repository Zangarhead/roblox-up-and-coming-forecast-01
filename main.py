"""
Roblox Up-and-Coming Chart Forecast Model.
Implements a recency-weighted rank mean with a 10-hour half-life, player-growth and recency-tilted drift,
horizon-growing diffusion, survival hazard with 28-day game age expiry, and 30% unnamed game share.
"""

import json
import math
import os
from datetime import datetime, timezone

HL = 10.0
S0 = 2.0
S1 = 0.7
UNNAMED = 0.30
LAM = 0.012
KP = 25.0
SAMPLES = 12

def parse_iso(ts_str):
    if not ts_str:
        return None
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str)

def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def load_data():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    task_path = os.path.join(input_dir, "task.json")
    history_path = os.path.join(input_dir, "history.ndjson")
    
    with open(task_path, "r", encoding="utf-8") as f:
        task = json.load(f)
        
    rows = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                
    return task, rows

def build(rows, until, window_from, window_hours=72.0):
    until_dt = parse_iso(until)
    parsed_reads = []
    for r in rows:
        at_dt = parse_iso(r.get("at"))
        if not at_dt:
            continue
        chart_list = r.get("up_and_coming", [])
        chart_dict = {}
        for g in chart_list:
            uid = g.get("universe_id")
            if uid is not None:
                chart_dict[uid] = g
        parsed_reads.append((at_dt, chart_dict, r))
    
    if not parsed_reads:
        return {"probabilities": [], "top10": []}
    
    parsed_reads.sort(key=lambda x: x[0])
    t0 = parsed_reads[0][0]
    
    reads_processed = []
    for at_dt, chart_dict, r in parsed_reads:
        th = (at_dt - t0).total_seconds() / 3600.0
        reads_processed.append((th, chart_dict, r))
        
    T0 = (until_dt - t0).total_seconds() / 3600.0
    
    game_obs = {}
    game_first_seen = {}
    game_creations = {}
    
    for idx, (th, chart_dict, r) in enumerate(reads_processed):
        details = r.get("details") or {}
        for uid, g in chart_dict.items():
            if uid not in game_obs:
                game_obs[uid] = []
                game_first_seen[uid] = idx
            rank = g.get("rank")
            playing = g.get("playing")
            game_obs[uid].append((th, rank, playing))
            
            if isinstance(details, dict):
                det = details.get(str(uid)) or details.get(uid)
                if det and isinstance(det, dict):
                    created_str = det.get("created")
                    if created_str:
                        cdt = parse_iso(created_str)
                        if cdt:
                            game_creations[uid] = cdt

    total_w = sum(0.5 ** ((T0 - th) / HL) for th, _, _ in reads_processed)
    if total_w <= 0:
        total_w = 1e-9
        
    hs = [window_hours * (i + 0.5) / SAMPLES for i in range(SAMPLES)]
    
    dist = {k: {} for k in range(1, 11)}
    mass = {}
    
    sorted_games = sorted(game_obs.keys(), key=lambda uid: game_first_seen[uid])
    last_read_th = reads_processed[-1][0] if reads_processed else 0.0
    
    window_from_dt = parse_iso(window_from)
    
    for uid in sorted_games:
        obs = game_obs[uid]
        wsum = 0.0
        rsum = 0.0
        for th, rank, _ in obs:
            w = 0.5 ** ((T0 - th) / HL)
            wsum += w
            rsum += w * rank
            
        mu0 = rsum / wsum if wsum > 0 else 50.0
        presence = wsum / total_w
        
        last_th, last_rank = obs[-1][0], obs[-1][1]
        if last_th < last_read_th - 0.01:
            mu0 = max(mu0, last_rank + 20.0)
            presence *= 0.45
            
        # Drift regression
        recent_obs = [o for o in obs if T0 - o[0] <= 96.0]
        drift = 0.0
        if len(recent_obs) >= 3:
            sw = sum(0.5 ** ((T0 - o[0]) / HL) for o in recent_obs)
            sx = sum((0.5 ** ((T0 - o[0]) / HL)) * o[0] for o in recent_obs)
            sy = sum((0.5 ** ((T0 - o[0]) / HL)) * o[1] for o in recent_obs)
            sxx = sum((0.5 ** ((T0 - o[0]) / HL)) * o[0] * o[0] for o in recent_obs)
            sxy = sum((0.5 ** ((T0 - o[0]) / HL)) * o[0] * o[1] for o in recent_obs)
            den = sw * sxx - sx * sx
            if abs(den) > 1e-9:
                raw_slope = (sw * sxy - sx * sy) / den
                drift = raw_slope * (sw / (sw + 3.0))
                
        # Player growth
        playing_obs = [o for o in obs if o[2]]
        sp = 0.0
        if len(playing_obs) >= 3 and playing_obs[-1][0] > playing_obs[0][0]:
            dt_hours = playing_obs[-1][0] - playing_obs[0][0]
            p_start = playing_obs[0][2]
            p_end = playing_obs[-1][2]
            if p_start > 0 and p_end > 0 and dt_hours > 0:
                sp = (math.log(p_end) - math.log(p_start)) / dt_hours
                
        drift -= max(-0.15, min(0.15, KP * sp / 100.0))
        drift = max(-0.22, min(0.22, drift))
        # Hazard
        lam = LAM * (2.0 if presence < 0.5 else 1.0) * (1.0 + min(2.0, max(0.0, -sp) * 30.0))
        
        # On-chart probability at horizon 0
        p_on0 = min(1.0, presence * 1.2)
        if last_th >= last_read_th - 0.01:
            p_on0 = max(p_on0, 0.92)
            
        # Expiry
        expiry_h = None
        if uid in game_creations:
            cdt = game_creations[uid]
            # 28 days = 28 * 24 hours = 672 hours
            expiry_dt = cdt.timestamp() + 28 * 86400
            expiry_h = (expiry_dt - window_from_dt.timestamp()) / 3600.0
            if expiry_h <= 0:
                continue
                
        acc = {k: 0.0 for k in range(1, 11)}
        valid_samples = 0
        
        for h in hs:
            if expiry_h is not None and h >= expiry_h:
                continue
            pon = p_on0 * math.exp(-lam * h)
            if pon < 0.01:
                continue
            mu = max(1.0, min(120.0, mu0 + drift * h))
            sig = S0 + S1 * math.sqrt(h)
            for k in range(1, 11):
                prob_bin = phi((k + 0.5 - mu) / sig) - phi((k - 0.5 - mu) / sig)
                acc[k] += pon * prob_bin
            valid_samples += 1
            
        if valid_samples > 0:
            total_acc_sum = 0.0
            for k in range(1, 11):
                v = acc[k] / len(hs)
                if v > 0.0004:
                    dist[k][uid] = v
                    total_acc_sum += v
            if total_acc_sum > 0:
                mass[uid] = total_acc_sum
                
    probabilities = []
    named_set = set()
    
    for k in range(1, 11):
        col = dist[k]
        s = sum(col.values())
        if s <= 0:
            continue
        scale = (1.0 - UNNAMED) / s
        sorted_col = sorted(col.items(), key=lambda x: x[1], reverse=True)
        for uid, v in sorted_col[:100]:
            p = v * scale
            if p >= 0.0008:
                probabilities.append({
                    "universe_id": uid,
                    "place": k,
                    "p": round(p, 6)
                })
                named_set.add(uid)
                
    last_chart_dict = reads_processed[-1][1] if reads_processed else {}
    last_chart_sorted_games = sorted(last_chart_dict.values(), key=lambda g: g.get("rank", 999))
    last_chart_ids = [g.get("universe_id") for g in last_chart_sorted_games if g.get("universe_id") is not None]
    
    top10 = []
    # 1. Last read's chart ids sorted by rank that are in named_set
    for uid in last_chart_ids:
        if uid in named_set and uid not in top10:
            top10.append(uid)
            
    # 2. Remaining named games sorted by mass descending
    remaining_named = [uid for uid in mass.keys() if uid in named_set and uid not in top10]
    remaining_named.sort(key=lambda uid: mass[uid], reverse=True)
    for uid in remaining_named:
        if len(top10) < 10:
            top10.append(uid)
            
    # 3. Fill from last read's chart if still fewer than 10
    if len(top10) < 10:
        for uid in last_chart_ids:
            if uid not in top10:
                top10.append(uid)
                if len(top10) == 10:
                    break
                    
    top10 = top10[:10]
    return {"probabilities": probabilities, "top10": top10}

def main():
    task, rows = load_data()
    until = task.get("history_until")
    window = task.get("window", {})
    window_from = window.get("from")
    window_until = window.get("until")
    
    from_dt = parse_iso(window_from)
    until_dt_win = parse_iso(window_until)
    window_hours = 72.0
    if from_dt and until_dt_win:
        window_hours = (until_dt_win - from_dt).total_seconds() / 3600.0
        
    result = build(rows, until, window_from, window_hours=window_hours)
    
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
