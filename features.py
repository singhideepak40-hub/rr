"""Features and Match-State Tracking module for SuperRR Selector.

Tracks delivery-by-delivery cumulative match state (runs, wickets, balls bowled,
balls remaining, target, required run rate, Duckworth-Lewis style resource percentage),
and establishes contextual baseline expectations for runs and wickets.
"""

import numpy as np
import pandas as pd


def compute_match_state_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute ball-by-ball match state features.
    
    Tracks state BEFORE the delivery outcome, so models never leak current ball results.
    """
    df = deliveries.copy()

    # Sort deterministically
    df = df.sort_values(
        by=["match_id", "innings", "overid", "innings_ball_num", "over_ball_num"]
    ).reset_index(drop=True)

    # Group by match and innings
    grp = df.groupby(["match_id", "innings"])

    # 1. State BEFORE delivery
    df["runs_before"] = grp["total_runs_on_ball"].cumsum() - df["total_runs_on_ball"]
    df["wickets_before"] = grp["is_wicket"].cumsum() - df["is_wicket"].astype(int)
    df["legal_balls_before"] = grp["is_legal_ball"].cumsum() - df["is_legal_ball"]

    # 2. State limits
    # Balls limit (default 120 or rain-revised)
    df["balls_limit"] = np.where(
        df["innings"] == 1,
        df["balls_limit_inn1"].fillna(120),
        df["balls_limit_inn2"].fillna(120),
    ).astype(float)

    # Balls remaining & wickets in hand BEFORE delivery
    df["balls_left"] = np.maximum(0.0, df["balls_limit"] - df["legal_balls_before"])
    df["wickets_lost"] = np.clip(df["wickets_before"], 0, 10).astype(int)
    df["wickets_left"] = np.maximum(0, 10 - df["wickets_lost"])

    # 3. Rates BEFORE delivery
    df["crr"] = np.where(
        df["legal_balls_before"] > 0,
        (df["runs_before"] / df["legal_balls_before"]) * 6.0,
        0.0,
    )

    # Innings 2 Chase features
    df["runs_needed"] = np.where(
        df["innings"] == 2,
        np.maximum(0.0, df["target"].fillna(160.0) - df["runs_before"]),
        np.nan,
    )
    df["rrr"] = np.where(
        (df["innings"] == 2) & (df["balls_left"] > 0),
        (df["runs_needed"] / df["balls_left"]) * 6.0,
        np.where(df["innings"] == 2, df["runs_needed"] * 6.0, np.nan),
    )
    df["runs_per_ball_needed"] = np.where(
        df["innings"] == 2,
        df["runs_needed"] / np.maximum(1.0, df["balls_left"]),
        np.nan,
    )

    # 4. Resource Percentage (Duckworth-Lewis style resource proxy)
    # R(u, w) = (w/10) * (1 - exp(-0.045 * (balls_left/6))) / (1 - exp(-0.045 * 20))
    overs_left = df["balls_left"] / 6.0
    exp_factor = 1.0 - np.exp(-0.045 * overs_left)
    base_factor = 1.0 - np.exp(-0.045 * 20.0)
    df["resource_percentage"] = (df["wickets_left"] / 10.0) * (exp_factor / base_factor) * 100.0

    # 5. State AFTER delivery (used for transition & WPA)
    df["runs_after"] = df["runs_before"] + df["total_runs_on_ball"]
    df["wickets_after"] = np.clip(df["wickets_before"] + df["is_wicket"].astype(int), 0, 10)
    df["legal_balls_after"] = df["legal_balls_before"] + df["is_legal_ball"]
    df["balls_left_after"] = np.maximum(0.0, df["balls_limit"] - df["legal_balls_after"])
    df["wickets_left_after"] = np.maximum(0, 10 - df["wickets_after"])

    df["runs_needed_after"] = np.where(
        df["innings"] == 2,
        np.maximum(0.0, df["target"].fillna(160.0) - df["runs_after"]),
        np.nan,
    )
    df["rrr_after"] = np.where(
        (df["innings"] == 2) & (df["balls_left_after"] > 0),
        (df["runs_needed_after"] / df["balls_left_after"]) * 6.0,
        np.where(df["innings"] == 2, df["runs_needed_after"] * 6.0, np.nan),
    )

    # 6. Expected Runs and Wickets Baseline
    # Context buckets: phase x wickets_bucket x pace_or_spin
    df["wickets_bucket"] = pd.cut(
        df["wickets_lost"],
        bins=[-1, 2, 5, 10],
        labels=["0-2", "3-5", "6-10"],
    )
    df["pace_or_spin_clean"] = df["pace_or_spin"].fillna("Pace").map(
        lambda x: "Spin" if "spin" in str(x).lower() else "Pace"
    )

    expected_table = (
        df[df["is_legal_ball"] == 1]
        .groupby(["phase", "wickets_bucket", "pace_or_spin_clean"], observed=False)
        .agg(
            exp_runs=("total_runs_on_ball", "mean"),
            exp_wicket=("is_wicket", "mean"),
        )
        .reset_index()
    )

    df = df.merge(
        expected_table,
        on=["phase", "wickets_bucket", "pace_or_spin_clean"],
        how="left",
    )
    # Default fallbacks if any cell empty
    df["exp_runs"] = df["exp_runs"].fillna(1.25)
    df["exp_wicket"] = df["exp_wicket"].fillna(0.05)

    # Runs Above Expected (RAE) per delivery
    df["runs_above_expected"] = df["total_runs_on_ball"] - df["exp_runs"]
    df["wickets_above_expected"] = df["is_bowler_wicket"] - df["exp_wicket"]

    return df


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_and_preprocess_data
    raw = load_raw_data("Data Pack")
    deliv, matches, _ = clean_and_preprocess_data(raw)
    feat_df = compute_match_state_features(deliv)
    print("Features computed successfully! Shape:", feat_df.shape)
    print("Expected runs summary by phase:")
    print(feat_df.groupby("phase")["exp_runs"].mean())
