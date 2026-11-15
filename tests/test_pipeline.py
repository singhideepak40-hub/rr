"""Comprehensive Sanity and Integrity Tests for SuperRR Selector.

Verifies data ingestion, feature calculations, win probability boundaries,
pressure index normalization, player ratings, squad gap analysis, and API endpoints.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
from data_loader import load_raw_data, clean_and_preprocess_data
from features import compute_match_state_features
from win_probability import WinProbabilityEngine
from pressure_index import PressureIndexEngine
from ratings import RatingsEngine
from squad_fit import SquadFitEngine
from main import (
    load_app_cache,
    get_rankings,
    get_player_profile,
    simulate_pressure,
    get_squad_fit,
    get_validation_data,
)


def test_data_loader_integrity():
    """Verify raw data loading, super over filtering, extras separation, and ID coverage."""
    raw = load_raw_data("Data Pack")
    assert "bbb" in raw and "matches" in raw and "competitions" in raw
    
    deliv, matches, rep = clean_and_preprocess_data(raw)
    assert rep["super_over_deliveries_excluded"] == 38
    assert rep["runs_includes_extras"] is False
    assert rep["legal_deliveries"] > 190000
    assert rep["matches_with_deliveries"] == 867
    assert deliv["innings"].isin([1, 2]).all()
    # No NaNs in critical identifiers
    assert deliv["striker_id"].notna().all()
    assert deliv["bowler_id"].notna().all()


def test_match_state_features_no_leakage():
    """Ensure state features are computed strictly prior to current ball outcome."""
    raw = load_raw_data("Data Pack")
    deliv, matches, _ = clean_and_preprocess_data(raw)
    sample = deliv.head(500).copy()
    feat = compute_match_state_features(sample)

    assert "runs_before" in feat.columns
    assert "balls_left" in feat.columns
    assert "exp_runs" in feat.columns
    # Check that runs_before on ball 1 is 0
    assert feat.iloc[0]["runs_before"] == 0
    assert (feat["balls_left"] >= 0).all()
    assert (feat["exp_runs"] > 0).all()


def test_win_probability_and_wpa():
    """Verify WP boundaries [0, 1] and zero-sum WPA."""
    raw = load_raw_data("Data Pack")
    deliv, matches, _ = clean_and_preprocess_data(raw)
    feat = compute_match_state_features(deliv.head(2000))

    engine = WinProbabilityEngine(random_state=42)
    engine.fit_inn1_par_model(feat)
    engine.fit_inn2_model(feat)
    wpa_df = engine.compute_wpa_for_dataframe(feat)

    assert (wpa_df["wp_before"] >= 0.0).all() and (wpa_df["wp_before"] <= 1.0).all()
    assert (wpa_df["wp_after"] >= 0.0).all() and (wpa_df["wp_after"] <= 1.0).all()
    # Batter WPA + Bowler WPA should exactly cancel out
    diff = wpa_df["wpa_batter"] + wpa_df["wpa_bowler"]
    np.testing.assert_allclose(diff.values, 0.0, atol=1e-7)


def test_pressure_index_normalization():
    """Verify PI is bounded [0, 1], flags top 25%, and buckets are valid."""
    raw = load_raw_data("Data Pack")
    deliv, matches, _ = clean_and_preprocess_data(raw)
    feat = compute_match_state_features(deliv.head(5000))

    wp_engine = WinProbabilityEngine(random_state=42)
    wp_engine.fit_inn1_par_model(feat)
    wp_engine.fit_inn2_model(feat)
    wpa_df = wp_engine.compute_wpa_for_dataframe(feat)

    pi_engine = PressureIndexEngine(wp_engine=wp_engine, default_threshold_pct=25.0)
    pi_df = pi_engine.compute_pressure_index(wpa_df, threshold_pct=25.0)

    assert (pi_df["pressure_index"] >= 0.0).all() and (pi_df["pressure_index"] <= 1.0).all()
    assert set(pi_df["is_high_pressure"].unique()).issubset({0, 1})
    valid_buckets = {"Extreme Crisis", "High Pressure", "Moderate Pressure", "Low-Moderate", "Low Pressure / Garbage Time"}
    assert set(pi_df["pressure_bucket"].unique()).issubset(valid_buckets)


def test_top10_rankings_integrity():
    """Verify Top 10 rankings have no duplicates, scores in [0, 100], and sequential ranks 1-10."""
    # Read precomputed outputs
    top_bat = pd.read_csv("outputs/top10_batters.csv")
    top_bowl = pd.read_csv("outputs/top10_bowlers.csv")

    assert len(top_bat) == 10
    assert len(top_bowl) == 10
    assert top_bat["rank"].tolist() == list(range(1, 11))
    assert top_bowl["rank"].tolist() == list(range(1, 11))
    assert top_bat["player_id"].nunique() == 10
    assert top_bowl["player_id"].nunique() == 10
    assert (top_bat["reliability_score"] >= 0).all() and (top_bat["reliability_score"] <= 100).all()
    assert (top_bowl["reliability_score"] >= 0).all() and (top_bowl["reliability_score"] <= 100).all()


def test_squad_fit_engine():
    """Verify Rajasthan Royals squad retention tagging and gap analyzer output."""
    squad_engine = SquadFitEngine("rr_squad.json")
    retained = squad_engine.get_retained_ids()
    assert 2258 in retained  # Sanju Samson
    assert 6819 in retained  # Yashasvi Jaiswal

    # Check gap analysis artifact
    with open("artifacts/squad_gaps.json", "r") as f:
        gaps = json.load(f)
    assert len(gaps["tactical_gaps"]) == 5
    for g in gaps["tactical_gaps"]:
        assert len(g["recommended_targets"]) > 0


def test_fastapi_endpoints():
    """Verify FastAPI API endpoint functions return valid data structures and codes."""
    load_app_cache()

    # 1. Rankings
    data_rank = get_rankings(category="batters", format_mode="combined", role="all", retention="all", search=None, limit=10)
    assert len(data_rank["results"]) == 10
    top_p = data_rank["results"][0]
    assert "striker_name_clean" in top_p
    assert "reliability_score" in top_p

    # 2. Player Profile
    top_pid = top_p["striker_id"]
    prof = get_player_profile(top_pid)
    assert prof["player_id"] == top_pid
    assert "batting_profile" in prof

    # 3. Simulate Pressure
    sim = simulate_pressure(innings=2, balls_left=18, wickets_lost=5, runs_needed=38.0, is_knockout=0)
    assert "simulation" in sim
    assert sim["simulation"]["pressure_index"] > 0
    assert sim["simulation"]["pressure_bucket"] in {"Extreme Crisis", "High Pressure", "Moderate Pressure"}

    # 4. Squad Fit
    squad_data = get_squad_fit()
    assert "tactical_gaps" in squad_data
    assert len(squad_data["tactical_gaps"]) == 5

    # 5. Validation
    val_data = get_validation_data()
    assert "metrics" in val_data
    assert len(val_data["charts"]) >= 5


if __name__ == "__main__":
    tests = [
        ("test_data_loader_integrity", test_data_loader_integrity),
        ("test_match_state_features_no_leakage", test_match_state_features_no_leakage),
        ("test_win_probability_and_wpa", test_win_probability_and_wpa),
        ("test_pressure_index_normalization", test_pressure_index_normalization),
        ("test_top10_rankings_integrity", test_top10_rankings_integrity),
        ("test_squad_fit_engine", test_squad_fit_engine),
        ("test_fastapi_endpoints", test_fastapi_endpoints),
    ]
    print("=" * 60)
    print("RUNNING UNIT & INTEGRITY TEST SUITE")
    print("=" * 60)
    for name, fn in tests:
        print(f"Running {name}...", end=" ")
        fn()
        print("✓ PASSED")
    print("=" * 60)
    print("ALL 7 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)
