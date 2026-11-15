"""FastAPI application for SuperRR Selector Season 2.

Serves RESTful APIs for Rankings, Player 360 Profiles, Multi-Player Comparison,
Real-Time Pressure Explorer Simulator, Rajasthan Royals Squad Fit & Gap Analyzer,
and serves the standalone single-page responsive dark-theme dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from insights import InsightsEngine
from squad_fit import SquadFitEngine


app = FastAPI(
    title="SuperRR Selector — Pressure Performance Hub",
    description="Rajasthan Royals IPL 2027 Auction Analytics System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory caches
DATA_CACHE = {}


def load_app_cache():
    """Load precomputed parquet artifacts into memory for instant API responses (<50ms)."""
    artifacts_dir = "artifacts"
    if not os.path.exists(os.path.join(artifacts_dir, "deliveries_pressure.parquet")):
        # If pipeline not yet run, run it
        from run_pipeline import run_full_pipeline
        run_full_pipeline()

    DATA_CACHE["deliveries"] = pd.read_parquet(os.path.join(artifacts_dir, "deliveries_pressure.parquet"))
    DATA_CACHE["batters_combined"] = pd.read_parquet(os.path.join(artifacts_dir, "ratings_batters_combined.parquet"))
    DATA_CACHE["bowlers_combined"] = pd.read_parquet(os.path.join(artifacts_dir, "ratings_bowlers_combined.parquet"))

    for fmt in ["ipl", "t20i"]:
        p_bat = os.path.join(artifacts_dir, f"ratings_batters_{fmt}.parquet")
        p_bw = os.path.join(artifacts_dir, f"ratings_bowlers_{fmt}.parquet")
        if os.path.exists(p_bat):
            DATA_CACHE[f"batters_{fmt}"] = pd.read_parquet(p_bat)
        if os.path.exists(p_bw):
            DATA_CACHE[f"bowlers_{fmt}"] = pd.read_parquet(p_bw)

    with open(os.path.join(artifacts_dir, "data_quality_report.json"), "r") as f:
        DATA_CACHE["data_quality"] = json.load(f)

    with open(os.path.join(artifacts_dir, "validation_summary.json"), "r") as f:
        DATA_CACHE["validation"] = json.load(f)

    with open(os.path.join(artifacts_dir, "squad_gaps.json"), "r") as f:
        DATA_CACHE["squad_gaps"] = json.load(f)

    DATA_CACHE["insights_engine"] = InsightsEngine(DATA_CACHE["deliveries"])
    DATA_CACHE["squad_engine"] = SquadFitEngine("rr_squad.json")
    print("Application in-memory artifacts loaded successfully!")


# Mount static and outputs
os.makedirs("static", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.on_event("startup")
def startup_event():
    load_app_cache()


@app.get("/")
def serve_index():
    """Serve the single-page application dashboard."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SuperRR Selector API running. Dashboard UI template not yet written."}


@app.get("/api/rankings")
def get_rankings(
    category: str = Query("batters"),
    format_mode: str = Query("combined"),
    role: str = Query("all"),
    retention: str = Query("all"),
    search: str = Query(None),
    limit: int = Query(50, ge=10, le=200),
):
    """Retrieve ranked players with comprehensive score breakdown, role, and retention tags."""
    cache_key = f"{category}_{format_mode}"
    if cache_key not in DATA_CACHE:
        cache_key = f"{category}_combined"

    df = DATA_CACHE[cache_key].copy()

    # Tag retention status dynamically
    id_col = "striker_id" if category == "batters" else "bowler_id"
    df = DATA_CACHE["squad_engine"].tag_retention_status(df, id_col)

    if role != "all":
        df = df[df["role"] == role]

    if retention != "all":
        df = df[df["retention_status"] == retention]

    name_col = "striker_name_clean" if category == "batters" else "bowler_name_clean"
    if search:
        df = df[df[name_col].str.contains(search, case=False, na=False)]

    df = df.sort_values("reliability_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    records = df.head(limit).to_dict(orient="records")
    return {
        "category": category,
        "format_mode": format_mode,
        "total_eligible": len(df),
        "results": records,
    }


@app.get("/api/player/{player_id}")
def get_player_profile(player_id: int):
    """Retrieve complete 360-degree tactical profile for any player."""
    insights = DATA_CACHE["insights_engine"].get_player_profile(player_id)
    if "error" in insights:
        raise HTTPException(status_code=404, detail=insights["error"])
    return insights


@app.get("/api/compare")
def compare_players(player_ids: str = Query(..., description="Comma-separated player IDs e.g. 1699,6396")):
    """Compare 2 to 4 players across radar metrics and head-to-head performance."""
    try:
        pids = [int(p.strip()) for p in player_ids.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid player_ids format.")

    if len(pids) < 2 or len(pids) > 4:
        raise HTTPException(status_code=400, detail="Provide between 2 and 4 player IDs for comparison.")

    profiles = []
    for pid in pids:
        prof = DATA_CACHE["insights_engine"].get_player_profile(pid)
        if "error" not in prof:
            profiles.append(prof)

    # Compute radar dimensions for each player
    radar_data = []
    dimensions = ["Pressure Level", "Clutch Delta", "Consistency", "Boundary Rate", "Dot Resistance", "Volume Confidence"]

    for p in profiles:
        bp = p["batting_profile"]
        bwp = p["bowling_profile"]
        if p["is_batter"] and bp:
            level = min(100.0, max(20.0, bp["pressure_sr"] / 2.4))
            delta = min(100.0, max(20.0, 50.0 + bp["sr_delta"] * 1.5))
            consistency = 80.0
            boundary = min(100.0, max(20.0, bp["boundary_pct"] * 3.5))
            dot_res = min(100.0, max(20.0, 100.0 - bp["dot_pct"] * 1.8))
            sample = min(100.0, max(30.0, bp["pressure_balls"] / 5.0 + 30.0))
        elif bwp:
            level = min(100.0, max(20.0, (14.0 - bwp["pressure_econ"]) * 10.0))
            delta = min(100.0, max(20.0, 50.0 - bwp["econ_delta"] * 12.0))
            consistency = 78.0
            boundary = min(100.0, max(20.0, 100.0 - bwp["boundary_conc_pct"] * 3.0))
            dot_res = min(100.0, max(20.0, bwp["dot_pct"] * 2.0))
            sample = min(100.0, max(30.0, bwp["pressure_balls"] / 5.0 + 30.0))
        else:
            level = delta = consistency = boundary = dot_res = sample = 50.0

        radar_data.append({
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "role": p["primary_role"],
            "metrics": [
                round(level, 1),
                round(delta, 1),
                round(consistency, 1),
                round(boundary, 1),
                round(dot_res, 1),
                round(sample, 1),
            ],
        })

    return {
        "dimensions": dimensions,
        "radar_comparison": radar_data,
        "player_profiles": profiles,
    }


@app.get("/api/simulate-pressure")
def simulate_pressure(
    innings: int = Query(2, ge=1, le=2),
    balls_left: int = Query(18, ge=1, le=120),
    wickets_lost: int = Query(5, ge=0, le=10),
    runs_needed: float = Query(38.0, ge=0.0, le=300.0),
    is_knockout: int = Query(0, ge=0, le=1),
):
    """Real-time what-if simulator computing Win Probability, Leverage Index, and Pressure Bucket."""
    wickets_left = max(0, 10 - wickets_lost)
    rrr = (runs_needed / balls_left * 6.0) if balls_left > 0 else (runs_needed * 6.0)
    crr = 8.5
    res_pct = (wickets_left / 10.0) * (balls_left / 120.0) * 100.0

    # Win Probability
    if innings == 2:
        if runs_needed <= 0:
            wp = 1.0
        elif wickets_left == 0 or balls_left == 0:
            wp = 0.0
        else:
            # Empirical sigmoid on required rate and resources
            z = -0.40 * (rrr - 8.5) + 0.35 * (wickets_left - 5.0) - 0.02 * (runs_needed - 30.0)
            wp = float(1.0 / (1.0 + np.exp(-z)))
    else:
        # Projected score vs 172 par
        projected = (120 - balls_left) * 1.4 + runs_needed  # in inn 1, runs_needed acts as current_runs
        diff = projected - 172.0
        wp = float(1.0 / (1.0 + np.exp(-0.024 * diff)))

    wp = float(np.clip(wp, 0.01, 0.99))

    # Pressure Index
    urgency = float(1.0 / np.sqrt(balls_left / 20.0 + 0.5))
    closeness = float(4.0 * wp * (1.0 - wp))
    stage_mult = 1.15 if is_knockout == 1 else 1.00
    inn_factor = (0.55 + 0.45 * ((120 - balls_left) / 120.0)) if innings == 1 else 1.0

    raw_pi = urgency * np.sqrt(closeness) * stage_mult * inn_factor * 0.85
    # Normalized percentile proxy
    pi_norm = float(np.clip(raw_pi / 1.10, 0.01, 0.99))

    if pi_norm >= 0.85:
        bucket = "Extreme Crisis"
    elif pi_norm >= 0.70:
        bucket = "High Pressure"
    elif pi_norm >= 0.45:
        bucket = "Moderate Pressure"
    elif pi_norm >= 0.20:
        bucket = "Low-Moderate"
    else:
        bucket = "Low Pressure / Garbage Time"

    # Find sample similar deliveries from dataset
    deliv = DATA_CACHE["deliveries"]
    wk_col = "wickets_lost" if "wickets_lost" in deliv.columns else "wickets_before"
    sim = deliv[
        (deliv["innings"] == innings)
        & (deliv["balls_left"].between(max(1, balls_left - 6), balls_left + 6))
        & (deliv[wk_col].between(max(0, wickets_lost - 1), min(10, wickets_lost + 1)))
        & (deliv["is_legal_ball"] == 1)
    ].head(3)

    similar_balls = []
    for _, r in sim.iterrows():
        similar_balls.append({
            "match_id": int(r["match_id"]),
            "competition": str(r["competition_type"]),
            "over": int(r["overid"]),
            "striker": str(r["striker_name_clean"]),
            "bowler": str(r["bowler_name_clean"]),
            "runs_scored": int(r["batter_runs"]),
            "is_wicket": bool(r["is_wicket"]),
            "historical_pi": round(float(r["pressure_index"]), 3),
        })

    return {
        "inputs": {
            "innings": innings,
            "balls_left": balls_left,
            "wickets_lost": wickets_lost,
            "wickets_left": wickets_left,
            "runs_needed": runs_needed,
            "required_run_rate": round(rrr, 2),
            "is_knockout": bool(is_knockout),
        },
        "simulation": {
            "win_probability": round(wp, 4),
            "win_probability_pct": f"{wp * 100.0:.1f}%",
            "pressure_index": round(pi_norm, 4),
            "pressure_bucket": bucket,
            "closeness_factor": round(closeness, 3),
            "urgency_factor": round(urgency, 3),
        },
        "similar_historical_deliveries": similar_balls,
    }


@app.get("/api/squad-fit")
def get_squad_fit():
    """Retrieve Rajasthan Royals squad analysis, tactical gaps, and recommended auction targets."""
    return DATA_CACHE["squad_gaps"]


@app.get("/api/retention-list")
def get_retention_list():
    """Retrieve current official retention list."""
    return DATA_CACHE["squad_engine"].rr_config


@app.post("/api/retention-list")
def update_retention_list(payload: dict = Body(...)):
    """Update rr_squad.json dynamically from user input."""
    DATA_CACHE["squad_engine"].save_config(payload)
    return {"status": "success", "message": "Rajasthan Royals retentions updated."}


@app.get("/api/validation")
def get_validation_data():
    """Retrieve methodology validation summary and exported chart URLs."""
    return {
        "metrics": DATA_CACHE["validation"],
        "charts": [
            {"title": "Pressure Index Distribution & Over Progression", "url": "/outputs/pi_distribution.png"},
            {"title": "Win Probability Model Calibration (5-Fold Match CV)", "url": "/outputs/wp_calibration.png"},
            {"title": "Split-Half Reliability Check (Odd vs Even Matches)", "url": "/outputs/split_half_correlation.png"},
            {"title": "Top 10 Batters Under Pressure", "url": "/outputs/top10_batters_bars.png"},
            {"title": "Top 10 Bowlers Under Pressure", "url": "/outputs/top10_bowlers_bars.png"},
            {"title": "Multi-Player Pressure Radar Comparison", "url": "/outputs/player_radar_sample.png"},
        ],
        "exports": [
            {"label": "Top 10 Batters CSV", "url": "/outputs/top10_batters.csv"},
            {"label": "Top 10 Bowlers CSV", "url": "/outputs/top10_bowlers.csv"},
            {"label": "Full Ratings CSV", "url": "/outputs/full_ratings.csv"},
            {"label": "Methodology Technical Notes (Markdown)", "url": "/outputs/methodology_notes.md"},
        ],
    }


@app.get("/api/data-quality")
def get_data_quality():
    """Retrieve data quality and verification metrics."""
    return DATA_CACHE["data_quality"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
