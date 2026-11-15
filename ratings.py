"""Player Ratings and Pressure Performance Engine for SuperRR Selector.

Calculates player-level baselines, high-pressure metrics, empirical Bayes shrinkage,
match-to-match consistency, sample confidence, composite reliability score (0-100),
role classification, and handles IPL vs T20I format toggles.
"""

import numpy as np
import pandas as pd


class RatingsEngine:
    """Computes comprehensive ratings for batters and bowlers under pressure."""

    def __init__(
        self,
        min_batter_balls: int = 120,
        min_batter_matches: int = 12,
        min_bowler_balls: int = 120,
        min_bowler_matches: int = 12,
        min_ipl_balls: int = 24,
        shrinkage_k_batter: float = 50.0,
        shrinkage_k_bowler: float = 50.0,
        weight_level: float = 0.35,
        weight_delta: float = 0.25,
        weight_consistency: float = 0.25,
        weight_sample: float = 0.15,
    ):
        self.min_batter_balls = min_batter_balls
        self.min_batter_matches = min_batter_matches
        self.min_bowler_balls = min_bowler_balls
        self.min_bowler_matches = min_bowler_matches
        self.min_ipl_balls = min_ipl_balls
        self.shrinkage_k_batter = shrinkage_k_batter
        self.shrinkage_k_bowler = shrinkage_k_bowler
        self.weight_level = weight_level
        self.weight_delta = weight_delta
        self.weight_consistency = weight_consistency
        self.weight_sample = weight_sample

    def apply_competition_weight(self, df: pd.DataFrame, format_mode: str = "combined") -> pd.DataFrame:
        """Filter or weight deliveries based on format mode (combined, ipl, t20i)."""
        sub = df.copy()
        if format_mode == "ipl":
            sub = sub[sub["competition_type"] == "IPL"].copy()
            sub["comp_weight"] = 1.0
        elif format_mode == "t20i":
            sub = sub[sub["competition_type"] == "T20I"].copy()
            sub["comp_weight"] = 1.0
        else:
            # Combined: IPL given 1.15x weight reflecting elite global talent concentration
            comp_weights = {"IPL": 1.15, "T20I": 1.00, "Other T20": 0.85}
            sub["comp_weight"] = sub["competition_type"].map(comp_weights).fillna(1.0)
        return sub

    def compute_batting_ratings(self, df: pd.DataFrame, format_mode: str = "combined") -> pd.DataFrame:
        """Compute full batting ratings, baseline deltas, consistency, and composite reliability score."""
        df_fmt = self.apply_competition_weight(df, format_mode)
        legal_deliv = df_fmt[df_fmt["is_legal_ball"] == 1].copy()

        # Weighted metrics
        legal_deliv["w_wpa"] = legal_deliv["wpa_batter"] * legal_deliv["comp_weight"]
        legal_deliv["w_rae"] = legal_deliv["runs_above_expected"] * legal_deliv["comp_weight"]
        legal_deliv["is_boundary"] = (legal_deliv["is_four"] | legal_deliv["is_six"]).astype(int)
        legal_deliv["is_dot"] = (legal_deliv["batter_runs"] == 0).astype(int)

        # Split into Pressure (PI >= 0.75) and Normal (PI < 0.75)
        press = legal_deliv[legal_deliv["is_high_pressure"] == 1]
        norm = legal_deliv[legal_deliv["is_high_pressure"] == 0]

        # 1. Normal Baseline per player
        norm_stats = norm.groupby("striker_id").agg(
            norm_balls=("is_legal_ball", "sum"),
            norm_runs=("batter_runs", "sum"),
            norm_dots=("is_dot", "sum"),
            norm_boundaries=("is_boundary", "sum"),
            norm_dismissals=("is_batter_out", "sum"),
        ).reset_index()
        norm_stats["norm_sr"] = np.where(
            norm_stats["norm_balls"] > 0,
            (norm_stats["norm_runs"] / norm_stats["norm_balls"]) * 100.0,
            120.0,
        )

        # 2. Pressure Performance per player
        press_stats = press.groupby(["striker_id", "striker_name_clean"]).agg(
            pressure_balls=("is_legal_ball", "sum"),
            pressure_runs=("batter_runs", "sum"),
            pressure_dots=("is_dot", "sum"),
            pressure_fours=("is_four", "sum"),
            pressure_sixes=("is_six", "sum"),
            pressure_boundaries=("is_boundary", "sum"),
            pressure_dismissals=("is_batter_out", "sum"),
            total_wpa=("w_wpa", "sum"),
            total_rae=("w_rae", "sum"),
            pressure_matches=("match_id", "nunique"),
        ).reset_index()

        # Merge normal baseline
        merged = press_stats.merge(norm_stats, on="striker_id", how="left")
        merged["norm_balls"] = merged["norm_balls"].fillna(0)
        merged["norm_sr"] = merged["norm_sr"].fillna(125.0)

        # Filter eligibility
        eligible = merged[
            (merged["pressure_balls"] >= self.min_batter_balls)
            & (merged["pressure_matches"] >= self.min_batter_matches)
        ].copy()

        if eligible.empty:
            return pd.DataFrame()

        # Rates
        eligible["pressure_sr"] = (eligible["pressure_runs"] / eligible["pressure_balls"]) * 100.0
        eligible["sr_delta"] = eligible["pressure_sr"] - eligible["norm_sr"]
        eligible["wpa_per_ball"] = eligible["total_wpa"] / eligible["pressure_balls"]
        eligible["rae_per_ball"] = eligible["total_rae"] / eligible["pressure_balls"]
        eligible["dot_pct"] = (eligible["pressure_dots"] / eligible["pressure_balls"]) * 100.0
        eligible["boundary_pct"] = (eligible["pressure_boundaries"] / eligible["pressure_balls"]) * 100.0
        eligible["dismissal_rate"] = eligible["pressure_dismissals"] / eligible["pressure_balls"]

        # 3. Empirical Bayes Shrinkage
        league_wpa_per_ball = eligible["total_wpa"].sum() / eligible["pressure_balls"].sum()
        k_b = self.shrinkage_k_batter
        n_b = eligible["pressure_balls"]
        eligible["shrunk_wpa_per_ball"] = (
            (n_b / (n_b + k_b)) * eligible["wpa_per_ball"]
            + (k_b / (n_b + k_b)) * league_wpa_per_ball
        )
        eligible["shrunk_rae_per_ball"] = (
            (n_b / (n_b + k_b)) * eligible["rae_per_ball"]
            + (k_b / (n_b + k_b)) * 0.0
        )

        # 4. Match-to-Match Consistency
        # Proportion of pressure matches with positive net contribution (WPA > 0 or RAE > 0)
        match_level = (
            press.groupby(["striker_id", "match_id"])
            .agg(m_wpa=("w_wpa", "sum"), m_rae=("w_rae", "sum"))
            .reset_index()
        )
        match_level["is_positive"] = ((match_level["m_wpa"] > 0) | (match_level["m_rae"] > 0)).astype(int)
        consistency_df = (
            match_level.groupby("striker_id")
            .agg(
                positive_share=("is_positive", "mean"),
                wpa_std=("m_wpa", "std"),
            )
            .reset_index()
        )
        consistency_df["wpa_std"] = consistency_df["wpa_std"].fillna(0.15)
        # Scaled consistency: high positive share and contained variance
        consistency_df["consistency_score"] = np.clip(
            (consistency_df["positive_share"] * 70.0) + (30.0 / (1.0 + consistency_df["wpa_std"] * 5.0)),
            0.0,
            100.0,
        )

        eligible = eligible.merge(consistency_df[["striker_id", "positive_share", "consistency_score"]], on="striker_id", how="left")
        eligible["positive_share"] = eligible["positive_share"].fillna(0.5)
        eligible["consistency_score"] = eligible["consistency_score"].fillna(50.0)

        # 5. Sample Size Confidence (logarithmic saturation)
        eligible["sample_confidence"] = np.clip(
            0.70 + 0.30 * (1.0 - np.exp(-(eligible["pressure_balls"] - self.min_batter_balls) / 90.0)),
            0.70,
            1.00,
        )

        # 6. Component Scores (0 to 100)
        # Level score: combination of shrunk WPA per ball and shrunk RAE per ball
        wpa_norm = np.clip((eligible["shrunk_wpa_per_ball"] + 0.005) / 0.015 * 50.0 + 50.0, 0.0, 100.0)
        rae_norm = np.clip((eligible["shrunk_rae_per_ball"] + 0.3) / 0.8 * 50.0 + 50.0, 0.0, 100.0)
        eligible["score_level"] = 0.60 * wpa_norm + 0.40 * rae_norm

        # Delta score: SR delta vs own baseline
        eligible["score_delta"] = np.clip(50.0 + eligible["sr_delta"] * 1.5, 0.0, 100.0)

        # Consistency score
        eligible["score_consistency"] = eligible["consistency_score"]

        # Sample score
        eligible["score_sample"] = eligible["sample_confidence"] * 100.0

        # Composite Reliability Score
        raw_composite = (
            self.weight_level * eligible["score_level"]
            + self.weight_delta * eligible["score_delta"]
            + self.weight_consistency * eligible["score_consistency"]
            + self.weight_sample * eligible["score_sample"]
        )
        eligible["reliability_score"] = np.round(raw_composite, 2)

        # Determine Role
        total_batting_counts = df.groupby("striker_id")["is_legal_ball"].sum().to_dict()
        total_bowling_counts = df.groupby("bowler_id")["is_legal_ball"].sum().to_dict()

        eligible["tot_faced"] = eligible["striker_id"].map(total_batting_counts).fillna(0).astype(int)
        eligible["tot_bowled"] = eligible["striker_id"].map(total_bowling_counts).fillna(0).astype(int)

        def assign_role_bat(row):
            bf = row["tot_faced"]
            bb = row["tot_bowled"]
            if bb >= 100 and (bf < 50 or bb >= 4.0 * max(1, bf)):
                return "Bowler"
            elif bf >= 100 and (bb < 50 or bf >= 4.0 * max(1, bb)):
                return "Batter"
            elif bf >= 80 and bb >= 80:
                return "All-Rounder"
            elif bb >= bf:
                return "Bowler"
            else:
                return "Batter"
        eligible["role"] = eligible.apply(assign_role_bat, axis=1)

        # Sort and rank
        eligible = eligible.sort_values("reliability_score", ascending=False).reset_index(drop=True)
        eligible["rank"] = eligible.index + 1
        return eligible

    def compute_bowling_ratings(self, df: pd.DataFrame, format_mode: str = "combined") -> pd.DataFrame:
        """Compute full bowling ratings, baseline deltas, consistency, and composite reliability score."""
        df_fmt = self.apply_competition_weight(df, format_mode)
        legal_deliv = df_fmt[df_fmt["is_legal_ball"] == 1].copy()

        # Weighted metrics
        legal_deliv["w_wpa"] = legal_deliv["wpa_bowler"] * legal_deliv["comp_weight"]
        legal_deliv["w_rcae"] = legal_deliv["runs_above_expected"] * legal_deliv["comp_weight"]
        legal_deliv["is_boundary"] = (legal_deliv["is_four"] | legal_deliv["is_six"]).astype(int)
        legal_deliv["is_dot"] = (legal_deliv["total_runs_on_ball"] == 0).astype(int)

        # Split into Pressure (PI >= 0.75) and Normal (PI < 0.75)
        press = legal_deliv[legal_deliv["is_high_pressure"] == 1]
        norm = legal_deliv[legal_deliv["is_high_pressure"] == 0]

        # 1. Normal Baseline per player
        norm_stats = norm.groupby("bowler_id").agg(
            norm_balls=("is_legal_ball", "sum"),
            norm_runs_conc=("total_runs_on_ball", "sum"),
            norm_wickets=("is_bowler_wicket", "sum"),
            norm_dots=("is_dot", "sum"),
        ).reset_index()
        norm_stats["norm_econ"] = np.where(
            norm_stats["norm_balls"] > 0,
            (norm_stats["norm_runs_conc"] / norm_stats["norm_balls"]) * 6.0,
            8.2,
        )

        # 2. Pressure Performance per player
        press_stats = press.groupby(["bowler_id", "bowler_name_clean"]).agg(
            pressure_balls=("is_legal_ball", "sum"),
            pressure_runs_conc=("total_runs_on_ball", "sum"),
            pressure_wickets=("is_bowler_wicket", "sum"),
            pressure_dots=("is_dot", "sum"),
            pressure_boundaries_conc=("is_boundary", "sum"),
            total_wpa=("w_wpa", "sum"),
            total_rcae=("w_rcae", "sum"),
            pressure_matches=("match_id", "nunique"),
            ipl_balls=("is_legal_ball", lambda s: s[press.loc[s.index, "competition_type"] == "IPL"].sum()),
        ).reset_index()

        # Merge normal baseline
        merged = press_stats.merge(norm_stats, on="bowler_id", how="left")
        merged["norm_balls"] = merged["norm_balls"].fillna(0)
        merged["norm_econ"] = merged["norm_econ"].fillna(8.2)

        # Filter eligibility (for combined mode, ensure player has IPL experience)
        if format_mode == "combined":
            eligible = merged[
                (merged["pressure_balls"] >= self.min_bowler_balls)
                & (merged["pressure_matches"] >= self.min_bowler_matches)
                & (merged["ipl_balls"] >= self.min_ipl_balls)
            ].copy()
        elif format_mode == "ipl":
            eligible = merged[
                (merged["pressure_balls"] >= max(60, self.min_bowler_balls // 2))
                & (merged["pressure_matches"] >= 6)
            ].copy()
        else:
            eligible = merged[
                (merged["pressure_balls"] >= self.min_bowler_balls)
                & (merged["pressure_matches"] >= self.min_bowler_matches)
            ].copy()

        if eligible.empty:
            return pd.DataFrame()

        # Rates
        eligible["pressure_econ"] = (eligible["pressure_runs_conc"] / eligible["pressure_balls"]) * 6.0
        # Negative econ_delta is good (concedes fewer runs under pressure)
        eligible["econ_delta"] = eligible["pressure_econ"] - eligible["norm_econ"]
        eligible["wpa_per_ball"] = eligible["total_wpa"] / eligible["pressure_balls"]
        # RCAE: lower is better
        eligible["rcae_per_ball"] = eligible["total_rcae"] / eligible["pressure_balls"]
        eligible["dot_pct"] = (eligible["pressure_dots"] / eligible["pressure_balls"]) * 100.0
        eligible["boundary_conc_pct"] = (eligible["pressure_boundaries_conc"] / eligible["pressure_balls"]) * 100.0
        eligible["wicket_rate"] = eligible["pressure_wickets"] / eligible["pressure_balls"]

        # 3. Empirical Bayes Shrinkage
        league_wpa_per_ball = eligible["total_wpa"].sum() / eligible["pressure_balls"].sum()
        k_b = self.shrinkage_k_bowler
        n_b = eligible["pressure_balls"]
        eligible["shrunk_wpa_per_ball"] = (
            (n_b / (n_b + k_b)) * eligible["wpa_per_ball"]
            + (k_b / (n_b + k_b)) * league_wpa_per_ball
        )
        eligible["shrunk_rcae_per_ball"] = (
            (n_b / (n_b + k_b)) * eligible["rcae_per_ball"]
            + (k_b / (n_b + k_b)) * 0.0
        )

        # 4. Match-to-Match Consistency
        # Proportion of pressure spells with positive net contribution (WPA > 0 or RCAE < 0)
        match_level = (
            press.groupby(["bowler_id", "match_id"])
            .agg(m_wpa=("w_wpa", "sum"), m_rcae=("w_rcae", "sum"))
            .reset_index()
        )
        match_level["is_positive"] = ((match_level["m_wpa"] > 0) | (match_level["m_rcae"] < 0)).astype(int)
        consistency_df = (
            match_level.groupby("bowler_id")
            .agg(
                positive_share=("is_positive", "mean"),
                wpa_std=("m_wpa", "std"),
            )
            .reset_index()
        )
        consistency_df["wpa_std"] = consistency_df["wpa_std"].fillna(0.15)
        consistency_df["consistency_score"] = np.clip(
            (consistency_df["positive_share"] * 70.0) + (30.0 / (1.0 + consistency_df["wpa_std"] * 5.0)),
            0.0,
            100.0,
        )

        eligible = eligible.merge(consistency_df[["bowler_id", "positive_share", "consistency_score"]], on="bowler_id", how="left")
        eligible["positive_share"] = eligible["positive_share"].fillna(0.5)
        eligible["consistency_score"] = eligible["consistency_score"].fillna(50.0)

        # 5. Sample Size Confidence
        eligible["sample_confidence"] = np.clip(
            0.70 + 0.30 * (1.0 - np.exp(-(eligible["pressure_balls"] - self.min_bowler_balls) / 150.0)),
            0.70,
            1.00,
        )

        # 6. Component Scores (0 to 100)
        # Level score: high shrunk WPA per ball and negative shrunk RCAE per ball
        wpa_norm = np.clip((eligible["shrunk_wpa_per_ball"] + 0.002) / 0.015 * 50.0 + 50.0, 0.0, 100.0)
        rcae_norm = np.clip((-eligible["shrunk_rcae_per_ball"] + 0.2) / 0.6 * 50.0 + 50.0, 0.0, 100.0)
        eligible["score_level"] = 0.65 * wpa_norm + 0.35 * rcae_norm

        # Delta score: Economy suppression relative to expected phase par (RCAE)
        eligible["score_delta"] = np.clip(50.0 - (eligible["rcae_per_ball"] * 25.0), 10.0, 100.0)

        # Consistency score
        eligible["score_consistency"] = eligible["consistency_score"]

        # Sample score
        eligible["score_sample"] = eligible["sample_confidence"] * 100.0

        # Composite Reliability Score
        raw_composite = (
            self.weight_level * eligible["score_level"]
            + self.weight_delta * eligible["score_delta"]
            + self.weight_consistency * eligible["score_consistency"]
            + self.weight_sample * eligible["score_sample"]
        )
        eligible["reliability_score"] = np.round(raw_composite, 2)

        # Determine Role
        total_batting_counts = df.groupby("striker_id")["is_legal_ball"].sum().to_dict()
        total_bowling_counts = df.groupby("bowler_id")["is_legal_ball"].sum().to_dict()

        eligible["tot_faced"] = eligible["bowler_id"].map(total_batting_counts).fillna(0).astype(int)
        eligible["tot_bowled"] = eligible["bowler_id"].map(total_bowling_counts).fillna(0).astype(int)

        def assign_role_bowl(row):
            bf = row["tot_faced"]
            bb = row["tot_bowled"]
            if bb >= 100 and (bf < 50 or bb >= 4.0 * max(1, bf)):
                return "Bowler"
            elif bf >= 100 and (bb < 50 or bf >= 4.0 * max(1, bb)):
                return "Batter"
            elif bf >= 80 and bb >= 80:
                return "All-Rounder"
            elif bb >= bf:
                return "Bowler"
            else:
                return "Batter"
        eligible["role"] = eligible.apply(assign_role_bowl, axis=1)

        # Sort and rank
        eligible = eligible.sort_values("reliability_score", ascending=False).reset_index(drop=True)
        eligible["rank"] = eligible.index + 1
        return eligible


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_and_preprocess_data
    from features import compute_match_state_features
    from win_probability import WinProbabilityEngine
    from pressure_index import PressureIndexEngine

    raw = load_raw_data("Data Pack")
    deliv, matches, _ = clean_and_preprocess_data(raw)
    feat_df = compute_match_state_features(deliv)

    wp_engine = WinProbabilityEngine(random_state=42)
    wp_engine.fit_inn1_par_model(feat_df)
    wp_engine.fit_inn2_model(feat_df)
    wpa_df = wp_engine.compute_wpa_for_dataframe(feat_df)

    pi_engine = PressureIndexEngine(wp_engine=wp_engine, default_threshold_pct=25.0)
    pi_df = pi_engine.compute_pressure_index(wpa_df)

    ratings_engine = RatingsEngine()
    batters = ratings_engine.compute_batting_ratings(pi_df, format_mode="combined")
    bowlers = ratings_engine.compute_bowling_ratings(pi_df, format_mode="combined")

    print("\nTOP 10 BATTERS UNDER PRESSURE (Combined):")
    cols_bat = ["rank", "striker_id", "striker_name_clean", "role", "pressure_balls", "pressure_sr", "sr_delta", "reliability_score"]
    print(batters[cols_bat].head(10).to_string(index=False))

    print("\nTOP 10 BOWLERS UNDER PRESSURE (Combined):")
    cols_bowl = ["rank", "bowler_id", "bowler_name_clean", "role", "pressure_balls", "pressure_econ", "econ_delta", "reliability_score"]
    print(bowlers[cols_bowl].head(10).to_string(index=False))
