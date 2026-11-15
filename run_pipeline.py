"""End-to-End Pipeline Runner for SuperRR Selector Season 2.

Executes all analytical modules in sequence, precomputes artifacts, exports final
Top 10 CSVs, saves publication charts, and verifies execution reproducibility.
"""

import os
import json
import time
import pandas as pd
from data_loader import load_raw_data, clean_and_preprocess_data, print_data_quality_report
from features import compute_match_state_features
from win_probability import WinProbabilityEngine
from pressure_index import PressureIndexEngine
from ratings import RatingsEngine
from validation import ValidationEngine
from squad_fit import SquadFitEngine


def run_full_pipeline(
    data_dir: str = "Data Pack",
    artifacts_dir: str = "artifacts",
    outputs_dir: str = "outputs",
    threshold_pct: float = 25.0,
):
    """Run full sports data science pipeline from raw CSVs to exported rankings and app caches."""
    start_time = time.time()
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND CLEANING RAW DATA PACK")
    print("=" * 70)
    raw_data = load_raw_data(data_dir)
    deliveries, matches, quality_report = clean_and_preprocess_data(raw_data)
    print_data_quality_report(quality_report)

    # Save quality report artifact
    with open(os.path.join(artifacts_dir, "data_quality_report.json"), "w") as f:
        json.dump(quality_report, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 2: COMPUTING MATCH-STATE FEATURES & CONTEXT BASELINES")
    print("=" * 70)
    feat_df = compute_match_state_features(deliveries)
    print(f"Features engineered for {len(feat_df):,} deliveries. Columns: {len(feat_df.columns)}")

    print("\n" + "=" * 70)
    print("STEP 3: TRAINING WIN PROBABILITY MODELS & COMPUTING WPA")
    print("=" * 70)
    wp_engine = WinProbabilityEngine(random_state=42)
    wp_engine.fit_inn1_par_model(feat_df)
    wp_engine.fit_inn2_model(feat_df)
    print("Win Probability Validation Metrics (5-Fold Match CV):", wp_engine.metrics)
    wpa_df = wp_engine.compute_wpa_for_dataframe(feat_df)

    print("\n" + "=" * 70)
    print(f"STEP 4: COMPUTING PRESSURE / LEVERAGE INDEX (PI) (Top {threshold_pct}% Threshold)")
    print("=" * 70)
    pi_engine = PressureIndexEngine(wp_engine=wp_engine, default_threshold_pct=threshold_pct)
    pi_df = pi_engine.compute_pressure_index(wpa_df, threshold_pct=threshold_pct)
    high_press_count = pi_df["is_high_pressure"].sum()
    print(f"PI computed. High-pressure balls: {high_press_count:,} ({high_press_count / len(pi_df) * 100:.1f}%)")

    # Cache deliveries with PI
    cache_cols = [
        "match_id", "innings", "overid", "innings_ball_num", "over_ball_num",
        "competition_type", "season", "is_knockout", "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name_clean", "bowler_id", "bowler_name_clean",
        "batter_runs", "total_runs_on_ball", "is_legal_ball", "is_four", "is_six",
        "is_wicket", "is_bowler_wicket", "is_batter_out", "dismissal_type", "phase",
        "pace_or_spin_clean", "stroke", "on_or_off", "batting_type", "bowling_type",
        "runs_before", "wickets_before", "balls_left", "runs_needed", "rrr",
        "wp_before", "wp_after", "delta_wp", "wpa_batter", "wpa_bowler",
        "pressure_index", "is_high_pressure", "pressure_bucket", "runs_above_expected"
    ]
    available_cols = [c for c in cache_cols if c in pi_df.columns]
    pi_df[available_cols].to_parquet(os.path.join(artifacts_dir, "deliveries_pressure.parquet"), index=False)
    print(f"Cached {len(pi_df):,} delivery records to artifacts/deliveries_pressure.parquet")

    print("\n" + "=" * 70)
    print("STEP 5: COMPUTING BATTER AND BOWLER RELIABILITY RATINGS")
    print("=" * 70)
    ratings_engine = RatingsEngine()
    squad_engine = SquadFitEngine("rr_squad.json")

    # Combined (Default)
    batters_combined = ratings_engine.compute_batting_ratings(pi_df, format_mode="combined")
    bowlers_combined = ratings_engine.compute_bowling_ratings(pi_df, format_mode="combined")
    batters_combined = squad_engine.tag_retention_status(batters_combined, "striker_id")
    bowlers_combined = squad_engine.tag_retention_status(bowlers_combined, "bowler_id")

    # IPL Only
    batters_ipl = ratings_engine.compute_batting_ratings(pi_df, format_mode="ipl")
    bowlers_ipl = ratings_engine.compute_bowling_ratings(pi_df, format_mode="ipl")

    # T20I Only
    batters_t20i = ratings_engine.compute_batting_ratings(pi_df, format_mode="t20i")
    bowlers_t20i = ratings_engine.compute_bowling_ratings(pi_df, format_mode="t20i")

    # Save artifact tables
    batters_combined.to_parquet(os.path.join(artifacts_dir, "ratings_batters_combined.parquet"), index=False)
    bowlers_combined.to_parquet(os.path.join(artifacts_dir, "ratings_bowlers_combined.parquet"), index=False)
    if not batters_ipl.empty:
        batters_ipl.to_parquet(os.path.join(artifacts_dir, "ratings_batters_ipl.parquet"), index=False)
    if not bowlers_ipl.empty:
        bowlers_ipl.to_parquet(os.path.join(artifacts_dir, "ratings_bowlers_ipl.parquet"), index=False)
    if not batters_t20i.empty:
        batters_t20i.to_parquet(os.path.join(artifacts_dir, "ratings_batters_t20i.parquet"), index=False)
    if not bowlers_t20i.empty:
        bowlers_t20i.to_parquet(os.path.join(artifacts_dir, "ratings_bowlers_t20i.parquet"), index=False)

    print(f"Eligible Batters: {len(batters_combined):,} | Eligible Bowlers: {len(bowlers_combined):,}")

    print("\n" + "=" * 70)
    print("STEP 6: STATISTICAL VALIDATION & STABILITY CHECKS")
    print("=" * 70)
    val_engine = ValidationEngine(outputs_dir)
    print("Running Split-Half Test (Odd vs Even Matches)...")
    split_half_res = val_engine.run_split_half_test(pi_df, ratings_engine)
    print("Split-Half Results:", split_half_res)

    print("Running Temporal Stability (2022-24 vs 2025-26)...")
    temporal_res = val_engine.run_temporal_stability(pi_df, ratings_engine)
    print("Temporal Results:", temporal_res)

    print("Running Threshold Sensitivity Analysis (20%, 25%, 30%)...")
    sensitivity_res = val_engine.run_threshold_sensitivity(wpa_df, wp_engine, ratings_engine)
    print(f"Batters 20% vs 30% Overlap: {sensitivity_res['batters_overlap_pct_20_vs_30']:.1f}%")
    print(f"Bowlers 20% vs 30% Overlap: {sensitivity_res['bowlers_overlap_pct_20_vs_30']:.1f}%")

    print("Running Bootstrap 95% Confidence Intervals...")
    bootstrap_res = val_engine.run_bootstrap_ci(pi_df, ratings_engine, batters_combined, bowlers_combined, n_bootstraps=120)

    val_summary = {
        "split_half": split_half_res,
        "temporal": temporal_res,
        "sensitivity": sensitivity_res,
        "bootstrap": bootstrap_res,
        "wp_metrics": wp_engine.metrics,
    }
    with open(os.path.join(artifacts_dir, "validation_summary.json"), "w") as f:
        json.dump(val_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 7: GENERATING CHARTS & METHODOLOGY NOTES")
    print("=" * 70)
    val_engine.generate_all_charts(pi_df, wp_engine, batters_combined, bowlers_combined)
    val_engine.generate_methodology_markdown(
        quality_report,
        split_half_res,
        temporal_res,
        sensitivity_res,
        batters_combined,
        bowlers_combined,
    )
    print("Charts and methodology notes saved to outputs/")

    print("\n" + "=" * 70)
    print("STEP 8: ANALYZING RAJASTHAN ROYALS SQUAD GAPS")
    print("=" * 70)
    squad_gaps = squad_engine.analyze_squad_gaps(pi_df, batters_combined, bowlers_combined)
    with open(os.path.join(artifacts_dir, "squad_gaps.json"), "w") as f:
        json.dump(squad_gaps, f, indent=2)
    print(f"Diagnosed {len(squad_gaps['tactical_gaps'])} tactical scenarios for Rajasthan Royals.")

    print("\n" + "=" * 70)
    print("STEP 9: EXPORTING OFFICIAL CSV ARTIFACTS")
    print("=" * 70)
    # 1. Top 10 Batters
    top10_bat_export_cols = [
        "rank", "striker_id", "striker_name_clean", "role", "retention_status",
        "pressure_balls", "pressure_runs", "pressure_sr", "norm_sr", "sr_delta",
        "dot_pct", "boundary_pct", "total_wpa", "wpa_per_ball",
        "score_level", "score_delta", "score_consistency", "score_sample", "reliability_score"
    ]
    top10_batters = batters_combined.head(10)[top10_bat_export_cols].copy()
    top10_batters.rename(columns={"striker_id": "player_id", "striker_name_clean": "player_name"}, inplace=True)
    top10_batters.to_csv(os.path.join(outputs_dir, "top10_batters.csv"), index=False)

    # 2. Top 10 Bowlers
    top10_bowl_export_cols = [
        "rank", "bowler_id", "bowler_name_clean", "role", "retention_status",
        "pressure_balls", "pressure_runs_conc", "pressure_econ", "norm_econ", "econ_delta",
        "dot_pct", "boundary_conc_pct", "total_wpa", "wpa_per_ball",
        "score_level", "score_delta", "score_consistency", "score_sample", "reliability_score"
    ]
    top10_bowlers = bowlers_combined.head(10)[top10_bowl_export_cols].copy()
    top10_bowlers.rename(columns={"bowler_id": "player_id", "bowler_name_clean": "player_name"}, inplace=True)
    top10_bowlers.to_csv(os.path.join(outputs_dir, "top10_bowlers.csv"), index=False)

    # 3. Available Auction Targets (Excluding players retained by ANY franchise)
    all_retained = squad_engine.get_all_retained_ids()
    avail_batters = batters_combined[~batters_combined["striker_id"].isin(all_retained)].copy()
    avail_batters["auction_rank"] = range(1, len(avail_batters) + 1)
    top10_avail_batters = avail_batters.head(10)[["auction_rank", "striker_id", "striker_name_clean", "role", "pressure_balls", "pressure_sr", "sr_delta", "reliability_score"]].copy()
    top10_avail_batters.rename(columns={"auction_rank": "rank", "striker_id": "player_id", "striker_name_clean": "player_name"}, inplace=True)
    top10_avail_batters.to_csv(os.path.join(outputs_dir, "top10_auction_targets_batters.csv"), index=False)

    avail_bowlers = bowlers_combined[~bowlers_combined["bowler_id"].isin(all_retained)].copy()
    avail_bowlers["auction_rank"] = range(1, len(avail_bowlers) + 1)
    top10_avail_bowlers = avail_bowlers.head(10)[["auction_rank", "bowler_id", "bowler_name_clean", "role", "pressure_balls", "pressure_econ", "econ_delta", "reliability_score"]].copy()
    top10_avail_bowlers.rename(columns={"auction_rank": "rank", "bowler_id": "player_id", "bowler_name_clean": "player_name"}, inplace=True)
    top10_avail_bowlers.to_csv(os.path.join(outputs_dir, "top10_auction_targets_bowlers.csv"), index=False)

    # 4. Uncapped Domestic Gems (Indian players with zero international T20I caps)
    comps = pd.read_csv("Data Pack/competitions.csv")
    teams = pd.read_csv("Data Pack/teams.csv")
    mapping = pd.read_csv("Data Pack/competition_team_player_mapping.csv")
    t20i_comps = comps[comps["match_format"] == "T20I"]["comp_id"].unique()
    t20i_teams = set(teams[teams["comp_id"].isin(t20i_comps)]["team_id"].unique())
    intl_players = set(mapping[mapping["team_id"].isin(t20i_teams)]["player_id"].unique())

    uncapped_bat = avail_batters[~avail_batters["striker_id"].isin(intl_players)].head(5).copy()
    uncapped_bat_export = uncapped_bat[["striker_id", "striker_name_clean", "role", "pressure_balls", "pressure_sr", "reliability_score"]].rename(
        columns={"striker_id": "player_id", "striker_name_clean": "player_name"}
    )
    uncapped_bat_export["type"] = "Batter"

    uncapped_bowl = avail_bowlers[~avail_bowlers["bowler_id"].isin(intl_players)].head(5).copy()
    uncapped_bowl_export = uncapped_bowl[["bowler_id", "bowler_name_clean", "role", "pressure_balls", "pressure_econ", "reliability_score"]].rename(
        columns={"bowler_id": "player_id", "bowler_name_clean": "player_name", "pressure_econ": "pressure_metric"}
    )
    uncapped_bat_export.rename(columns={"pressure_sr": "pressure_metric"}, inplace=True)
    uncapped_bowl_export["type"] = "Bowler"
    uncapped_gems = pd.concat([uncapped_bat_export, uncapped_bowl_export], ignore_index=True)
    uncapped_gems.to_csv(os.path.join(outputs_dir, "top10_uncapped_gems.csv"), index=False)

    # 5. Full Ratings
    batters_all = batters_combined.copy()
    batters_all["category"] = "Batter"
    batters_all.rename(columns={"striker_id": "player_id", "striker_name_clean": "player_name"}, inplace=True)

    bowlers_all = bowlers_combined.copy()
    bowlers_all["category"] = "Bowler"
    bowlers_all.rename(columns={"bowler_id": "player_id", "bowler_name_clean": "player_name"}, inplace=True)

    common_cols = ["rank", "player_id", "player_name", "category", "role", "retention_status", "pressure_balls", "reliability_score"]
    full_ratings = pd.concat([batters_all[common_cols], bowlers_all[common_cols]], ignore_index=True)
    full_ratings.to_csv(os.path.join(outputs_dir, "full_ratings.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {elapsed:.1f} seconds!")
    print("\n" + "=" * 70)
    print("FINAL OFFICIAL TOP 10 BATTERS UNDER PRESSURE (Round 1 Submission):")
    print("=" * 70)
    print(top10_batters[["rank", "player_id", "player_name", "role", "pressure_balls", "pressure_sr", "sr_delta", "reliability_score"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("FINAL OFFICIAL TOP 10 BOWLERS UNDER PRESSURE (Round 1 Submission):")
    print("=" * 70)
    print(top10_bowlers[["rank", "player_id", "player_name", "role", "pressure_balls", "pressure_econ", "econ_delta", "reliability_score"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 AVAILABLE AUCTION TARGETS FOR RAJASTHAN ROYALS (Excluding Retained):")
    print("=" * 70)
    print("BATTERS:")
    print(top10_avail_batters[["rank", "player_id", "player_name", "role", "pressure_balls", "pressure_sr", "reliability_score"]].to_string(index=False))
    print("\nBOWLERS:")
    print(top10_avail_bowlers[["rank", "player_id", "player_name", "role", "pressure_balls", "pressure_econ", "reliability_score"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
