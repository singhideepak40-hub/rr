"""Pressure Index (PI) module for SuperRR Selector.

Defines delivery-level leverage, expected win probability swing, closeness,
stage stakes, normalizes PI to [0, 1], assigns human-readable pressure buckets,
and produces diagnostic distribution summaries.
"""

import numpy as np
import pandas as pd
from win_probability import WinProbabilityEngine


class PressureIndexEngine:
    """Computes Pressure Index (PI) and pressure classifications."""

    def __init__(self, wp_engine: WinProbabilityEngine, default_threshold_pct: float = 25.0):
        self.wp_engine = wp_engine
        self.default_threshold_pct = default_threshold_pct

    def compute_pressure_index(self, df: pd.DataFrame, threshold_pct: float = None) -> pd.DataFrame:
        """Compute raw and normalized PI, assign pressure buckets, and label high pressure."""
        if threshold_pct is None:
            threshold_pct = self.default_threshold_pct

        res = df.copy()

        # 1. Balls left and Urgency Factor: Inverse square root of remaining balls
        balls_left = np.maximum(1.0, res["balls_left"].fillna(120).values)
        urgency = 1.0 / np.sqrt(balls_left / 20.0 + 0.5)
        res["urgency"] = urgency

        # 2. Closeness factor: 4 * wp * (1 - wp)
        wp = np.clip(res["wp_before"].fillna(0.5).values, 0.001, 0.999)
        closeness = 4.0 * wp * (1.0 - wp)
        res["closeness"] = closeness

        # 3. Knockout / stage multiplier: 1.15x for playoffs / knockouts
        stage_mult = np.where(res["is_knockout"] == 1, 1.15, 1.0)
        res["stage_multiplier"] = stage_mult

        # 4. Innings calibration: In innings 1, early overs set par, pressure builds toward death
        inn_factor = np.where(res["innings"] == 1, 0.55 + 0.45 * (res["overid"].clip(1, 20) / 20.0), 1.0)
        res["inn_factor"] = inn_factor

        # 5. Volatility sensitivity: Actual / potential outcome delta
        delta_mag = np.abs(res["delta_wp"].fillna(0.0).values)

        # 6. Raw PI combining urgency, closeness, stage, and volatility
        raw_pi = urgency * np.sqrt(closeness) * stage_mult * inn_factor * (0.60 + 0.40 * np.clip(delta_mag * 15.0, 0, 1))
        res["pi_raw"] = raw_pi

        # 7. Normalized PI: percentile rank [0.0, 1.0] across all legal deliveries
        legal_mask = res["is_legal_ball"] == 1
        res["pressure_index"] = 0.5
        res.loc[legal_mask, "pressure_index"] = (
            res.loc[legal_mask, "pi_raw"].rank(pct=True).astype(float)
        )
        res["pressure_index"] = res["pressure_index"].ffill().bfill()

        # 8. High Pressure Classification (configurable threshold, default top 25% => PI >= 0.75)
        cutoff = 1.0 - (threshold_pct / 100.0)
        res["is_high_pressure"] = (res["pressure_index"] >= cutoff).astype(int)

        # 9. Pressure Buckets
        def assign_bucket(pi):
            if pi >= 0.85:
                return "Extreme Crisis"
            elif pi >= 0.70:
                return "High Pressure"
            elif pi >= 0.45:
                return "Moderate Pressure"
            elif pi >= 0.20:
                return "Low-Moderate"
            else:
                return "Low Pressure / Garbage Time"

        res["pressure_bucket"] = res["pressure_index"].apply(assign_bucket)
        return res

    @staticmethod
    def get_bucket_explanations() -> dict[str, dict]:
        """Return clear, human-readable explanations of each pressure tier."""
        return {
            "Extreme Crisis": {
                "pi_range": ">= 0.85 (Top 15%)",
                "description": "Death overs chase (overs 16-20) with RRR > 10.0, tight finishes where every delivery swings win probability by >12%, or knockout playoff nailbiters.",
                "example_scenario": "Innings 2, 18 balls left, 38 runs required, 5 wickets in hand (RRR 12.67). A boundary puts chasing team at 65% WP; a dot or wicket plunges them to 25%."
            },
            "High Pressure": {
                "pi_range": "0.70 – 0.84 (Top 15-30%)",
                "description": "Tense middle-overs turning points, powerplay collapses in knockouts, or death overs setting high par where wickets and boundaries determine target par.",
                "example_scenario": "Innings 2, over 12, required rate climbing past 9.5 RPO, 4 wickets down against frontline pace."
            },
            "Moderate Pressure": {
                "pi_range": "0.45 – 0.69 (Middle 25%)",
                "description": "Balanced match state; required run rate manageable (7.5-9.0 RPO) with set batters at the crease.",
                "example_scenario": "Innings 1, over 10, score 82/2, normal phase progression."
            },
            "Low-Moderate": {
                "pi_range": "0.20 – 0.44 (Lower 25%)",
                "description": "Controlled phase where one team is noticeably ahead of par, but game remains nominally active.",
                "example_scenario": "Innings 2, 45 balls left, 40 runs needed, 8 wickets in hand."
            },
            "Low Pressure / Garbage Time": {
                "pi_range": "< 0.20 (Bottom 20%)",
                "description": "Match outcome virtually settled (Win Probability > 95% or < 5%). Performance has negligible leverage on winner.",
                "example_scenario": "Innings 2, 60 runs required from 12 balls with 8 down (hopeless chase), or 4 runs required from 30 balls."
            }
        }
