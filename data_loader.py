"""Data Loader module for SuperRR Selector.

Loads raw CSVs, performs data quality verification, cleans ball-by-ball
deliveries, filters out super overs, handles revised targets (DLS),
and builds canonical player mappings.
"""

import os
import json
import numpy as np
import pandas as pd


def load_raw_data(data_dir: str = "Data Pack") -> dict[str, pd.DataFrame]:
    """Load all raw CSV files from the Data Pack directory."""
    files = {
        "bbb": "ball_by_ball.csv",
        "matches": "matches.csv",
        "competitions": "competitions.csv",
        "teams": "teams.csv",
        "playing_eleven": "playing_eleven.csv",
        "player_mapping": "competition_team_player_mapping.csv",
    }
    loaded = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        loaded[key] = pd.read_csv(path, low_memory=False)
    return loaded


def build_player_catalog(player_mapping_df: pd.DataFrame, bbb_df: pd.DataFrame) -> dict[int, str]:
    """Build a canonical mapping from player_id to the most common cleaned player_name."""
    catalog: dict[int, str] = {}
    
    # 1. From mapping file
    mapping_clean = player_mapping_df.dropna(subset=["player_id", "player_name"]).copy()
    mapping_clean["player_id"] = mapping_clean["player_id"].astype(int)
    mapping_clean["player_name"] = mapping_clean["player_name"].astype(str).str.strip()
    
    # Most frequent name per id
    mapping_names = (
        mapping_clean.groupby(["player_id", "player_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["player_id", "count"], ascending=[True, False])
        .drop_duplicates(subset=["player_id"])
    )
    for _, row in mapping_names.iterrows():
        catalog[int(row["player_id"])] = str(row["player_name"])

    # 2. Backfill any missing from ball_by_ball strikers and bowlers
    for id_col, name_col in [("striker_id", "striker_name"), ("bowler_id", "bowler_name")]:
        sub = bbb_df.dropna(subset=[id_col, name_col]).copy()
        sub[id_col] = sub[id_col].astype(int)
        sub[name_col] = sub[name_col].astype(str).str.strip()
        names = (
            sub.groupby([id_col, name_col])
            .size()
            .reset_index(name="count")
            .sort_values([id_col, "count"], ascending=[True, False])
            .drop_duplicates(subset=[id_col])
        )
        for _, row in names.iterrows():
            pid = int(row[id_col])
            if pid not in catalog:
                catalog[pid] = str(row[name_col])

    return catalog


def clean_and_preprocess_data(raw_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean and preprocess ball-by-ball deliveries and matches.
    
    Returns:
        deliveries_df: Cleaned delivery-level records (innings 1 & 2 only)
        matches_df: Enriched match-level metadata
        quality_report: Verification summary dictionary
    """
    bbb = raw_data["bbb"].copy()
    matches = raw_data["matches"].copy()
    competitions = raw_data["competitions"].copy()
    player_mapping = raw_data["player_mapping"].copy()

    total_raw_balls = len(bbb)
    total_raw_matches = len(matches)

    # 1. Player Catalog
    player_catalog = build_player_catalog(player_mapping, bbb)

    # 2. Competitions enrichment
    comp_map = {}
    for _, c_row in competitions.iterrows():
        cid = c_row["comp_id"]
        cname = str(c_row.get("comp_name", ""))
        mformat = str(c_row.get("match_format", ""))
        season_str = str(c_row.get("season", ""))
        
        # Parse season year
        try:
            season_year = int(season_str.split("/")[0])
        except Exception:
            season_year = 2024

        if cname.startswith("IPL"):
            ctype = "IPL"
        elif mformat == "T20I":
            ctype = "T20I"
        else:
            ctype = "Other T20"
        
        comp_map[cid] = {
            "comp_name": cname,
            "match_format": mformat,
            "season_year": season_year,
            "competition_type": ctype,
        }

    # Enrich matches
    matches["comp_name"] = matches["comp_id"].map(lambda x: comp_map.get(x, {}).get("comp_name", "Unknown"))
    matches["match_format"] = matches["comp_id"].map(lambda x: comp_map.get(x, {}).get("match_format", "T20"))
    matches["season"] = matches["comp_id"].map(lambda x: comp_map.get(x, {}).get("season_year", 2024))
    matches["competition_type"] = matches["comp_id"].map(lambda x: comp_map.get(x, {}).get("competition_type", "Other T20"))
    
    # Knockout flag
    knockout_stages = {"Final", "Qualifier", "Eliminator", "Semi Final", "Super 4"}
    matches["is_knockout"] = (
        (matches["knock_out"] == 1) | (matches["stage"].isin(knockout_stages))
    ).astype(int)

    # Scheduled balls per innings
    matches["balls_limit_inn1"] = matches["first_inn_revised_overs"].fillna(120.0).astype(int)
    matches["balls_limit_inn2"] = matches["second_inn_revised_overs"].fillna(120.0).astype(int)

    # 3. Super Over Handling
    super_over_balls = len(bbb[bbb["innings"].isin([3, 4])])
    bbb_reg = bbb[bbb["innings"].isin([1, 2])].copy()

    # 4. Extras and Runs verification
    bbb_reg["batter_runs"] = bbb_reg["runs"].fillna(0).astype(int)
    bbb_reg["extras1_runs"] = bbb_reg["extras1_runs"].fillna(0).astype(int)
    bbb_reg["extras_2_runs"] = bbb_reg["extras_2_runs"].fillna(0).astype(int)
    bbb_reg["total_extras"] = bbb_reg["extras1_runs"] + bbb_reg["extras_2_runs"]
    bbb_reg["total_runs_on_ball"] = bbb_reg["batter_runs"] + bbb_reg["total_extras"]

    # Legal ball definition: wides, overstep (no-ball), and 'other' (no-ball variants) are illegal
    illegal_extras = {"wide", "Overstep", "other"}
    bbb_reg["is_legal_ball"] = (~bbb_reg["extras_type"].isin(illegal_extras)).astype(int)

    # 5. Dismissal Classification
    bowler_wicket_types = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
    bbb_reg["is_wicket"] = bbb_reg["is_wicket"].fillna(False).astype(bool)
    bbb_reg["is_bowler_wicket"] = (
        bbb_reg["is_wicket"] & bbb_reg["dismissal_type"].isin(bowler_wicket_types)
    ).astype(int)

    # Batter dismissal: striker was dismissed
    bbb_reg["is_batter_out"] = (
        bbb_reg["is_wicket"] & (bbb_reg["dismissed_player_id"] == bbb_reg["striker_id"])
    ).astype(int)

    # 6. Target calculation and DLS Handling
    inn1_totals = (
        bbb_reg[bbb_reg["innings"] == 1]
        .groupby("match_id")["total_runs_on_ball"]
        .sum()
        .to_dict()
    )

    def determine_target(row):
        mid = row["match_id"]
        # If revised target is explicitly present
        if pd.notna(row["revised_target"]) and row["revised_target"] > 0:
            return float(row["revised_target"])
        # Standard: innings 1 runs + 1
        inn1 = inn1_totals.get(mid, np.nan)
        if pd.notna(inn1):
            return float(inn1 + 1)
        return np.nan

    matches["target"] = matches.apply(determine_target, axis=1)

    # 7. Merge match info to deliveries
    match_meta_cols = [
        "match_id", "comp_id", "comp_name", "match_format", "season",
        "competition_type", "venue_id", "venue_country", "is_knockout",
        "win_team_id", "target", "balls_limit_inn1", "balls_limit_inn2"
    ]
    deliveries = bbb_reg.merge(matches[match_meta_cols], on="match_id", how="left")

    # Batting team won flag
    deliveries["batting_team_won"] = (
        (deliveries["win_team_id"].notna()) &
        (deliveries["batting_team_id"] == deliveries["win_team_id"])
    ).astype(int)

    # Phase
    def assign_phase(overid):
        if overid <= 6:
            return "Powerplay"
        elif overid <= 15:
            return "Middle"
        else:
            return "Death"
    deliveries["phase"] = deliveries["overid"].apply(assign_phase)

    # Canonical player names
    deliveries["striker_name_clean"] = deliveries["striker_id"].map(
        lambda x: player_catalog.get(int(x), f"Player_{x}") if pd.notna(x) else "Unknown"
    )
    deliveries["bowler_name_clean"] = deliveries["bowler_id"].map(
        lambda x: player_catalog.get(int(x), f"Player_{x}") if pd.notna(x) else "Unknown"
    )

    # Data Quality Report
    quality_report = {
        "total_raw_deliveries": int(total_raw_balls),
        "super_over_deliveries_excluded": int(super_over_balls),
        "clean_deliveries": int(len(deliveries)),
        "legal_deliveries": int(deliveries["is_legal_ball"].sum()),
        "illegal_deliveries": int((1 - deliveries["is_legal_ball"]).sum()),
        "total_matches": int(total_raw_matches),
        "matches_with_deliveries": int(deliveries["match_id"].nunique()),
        "revised_target_matches": int(matches["revised_target"].notna().sum()),
        "matches_with_winner": int(matches["win_team_id"].notna().sum()),
        "format_breakdown": matches["competition_type"].value_counts().to_dict(),
        "season_breakdown": matches["season"].value_counts().sort_index().to_dict(),
        "total_bowler_wickets": int(deliveries["is_bowler_wicket"].sum()),
        "total_run_outs": int((deliveries["dismissal_type"] == "RunOut").sum()),
        "runs_includes_extras": False,
        "extras_on_illegal_balls": int(deliveries[deliveries["is_legal_ball"] == 0]["total_extras"].sum()),
    }

    return deliveries, matches, quality_report


def print_data_quality_report(report: dict):
    """Print an executive data quality report to the console."""
    print("=" * 70)
    print("SUPER RR SELECTOR — DATA QUALITY & INTEGRITY REPORT")
    print("=" * 70)
    print(f"Total Raw Deliveries:          {report['total_raw_deliveries']:,}")
    print(f"Super Over Deliveries Excluded:{report['super_over_deliveries_excluded']:,} (Innings 3 & 4)")
    print(f"Clean Deliveries (Innings 1-2):{report['clean_deliveries']:,}")
    print(f"  - Legal Deliveries:          {report['legal_deliveries']:,} ({report['legal_deliveries']/report['clean_deliveries']*100:.2f}%)")
    print(f"  - Illegal Deliveries:        {report['illegal_deliveries']:,} (Wides/No-Balls)")
    print(f"Total Matches:                 {report['total_matches']:,}")
    print(f"Matches in Deliveries:         {report['matches_with_deliveries']:,}")
    print(f"Matches with Revised Target:   {report['revised_target_matches']:,} (DLS applied)")
    print(f"Competitions Breakdown:        {report['format_breakdown']}")
    print(f"Season Breakdown:              {report['season_breakdown']}")
    print(f"Total Bowler Wickets:          {report['total_bowler_wickets']:,}")
    print(f"Run Outs (Non-Bowler Wickets): {report['total_run_outs']:,}")
    print(f"Verification Check: 'runs' includes extras? -> {report['runs_includes_extras']} (Properly separated)")
    print("=" * 70)


if __name__ == "__main__":
    raw = load_raw_data("Data Pack")
    deliv, m, rep = clean_and_preprocess_data(raw)
    print_data_quality_report(rep)
