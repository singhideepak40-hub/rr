"""Rajasthan Royals Retain / Target View & Squad Gap Analyzer for SuperRR Selector.

Manages RR official retentions via rr_squad.json, tags players as Retain Candidate
vs Auction Target, evaluates RR's tactical squad gaps across 5 pressure dimensions,
and identifies the highest-value auction targets to fill those gaps.
"""

import os
import json
import numpy as np
import pandas as pd


class SquadFitEngine:
    """Analyzes Rajasthan Royals squad roster and diagnoses tactical auction gaps."""

    def __init__(self, squad_json_path: str = "rr_squad.json"):
        self.squad_json_path = squad_json_path
        self.rr_config = self._load_config()

    def _load_config(self) -> dict:
        """Load RR squad configuration from JSON."""
        if not os.path.exists(self.squad_json_path):
            return {"retentions": [], "recent_squad_pool": []}
        with open(self.squad_json_path, "r") as f:
            return json.load(f)

    def save_config(self, new_config: dict):
        """Save updated squad configuration to JSON."""
        with open(self.squad_json_path, "w") as f:
            json.dump(new_config, f, indent=2)
        self.rr_config = new_config

    def get_retained_ids(self) -> set[int]:
        """Return set of retained / core player IDs for Rajasthan Royals."""
        ret = {int(p["player_id"]) for p in self.rr_config.get("retentions", [])}
        if not ret:
            # Fallback to key RR core in data
            ret = {6819, 6423, 7260, 3297, 2411, 152, 3298, 7503}
        return ret

    def tag_retention_status(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Tag dataframe rows as 'RR Squad' or 'Auction target'."""
        rr_ids = self.get_retained_ids()
        res = df.copy()

        def _get_status(val):
            iv = int(val)
            if iv in rr_ids:
                return "RR Squad"
            return "Auction target"

        res["retention_status"] = res[id_col].map(_get_status)
        return res

    def analyze_squad_gaps(
        self,
        pi_df: pd.DataFrame,
        batters_df: pd.DataFrame,
        bowlers_df: pd.DataFrame,
    ) -> dict:
        """Evaluate RR roster against 5 tactical pressure scenarios and recommend auction targets."""
        retained_ids = self.get_retained_ids()
        legal = pi_df[pi_df["is_legal_ball"] == 1].copy()
        press = legal[legal["is_high_pressure"] == 1].copy()

        wk_col = "wickets_lost" if "wickets_lost" in press.columns else "wickets_before"

        # Define 5 tactical scenarios
        scenarios = {
            "death_bowling": {
                "name": "Death Bowling Lockdown (Overs 16-20)",
                "type": "bowling",
                "filter": (press["phase"] == "Death"),
                "description": "Defending tight totals in overs 16-20 against aggressive death-overs hitters.",
            },
            "powerplay_wickets": {
                "name": "Powerplay Strike Pacing (Overs 1-6)",
                "type": "bowling",
                "filter": (press["phase"] == "Powerplay"),
                "description": "Taking early wickets and strangling scoring in high-stakes powerplay starts.",
            },
            "death_batting": {
                "name": "Death Overs Chase Finishing (Overs 16-20)",
                "type": "batting",
                "filter": (press["phase"] == "Death") & (press["innings"] == 2),
                "description": "Boundary execution and finishing matches when required run rate is above 10.0.",
            },
            "middle_spin_attack": {
                "name": "Middle Overs Spin Counter-Attack (Overs 7-15 vs Spin)",
                "type": "batting",
                "filter": (press["phase"] == "Middle") & (press["pace_or_spin_clean"] == "Spin"),
                "description": "Accelerating and breaking spin choke in overs 7-15 without losing wickets.",
            },
            "top_order_crisis": {
                "name": "Top Order Crisis Resilience (Powerplay 1-6)",
                "type": "batting",
                "filter": (press["phase"] == "Powerplay") & (press[wk_col] >= 1),
                "description": "Rebuilding momentum and par preservation when early top-order wickets fall.",
            },
        }

        gap_analysis = []

        for s_key, s_data in scenarios.items():
            sub = press[s_data["filter"]]
            is_bat = s_data["type"] == "batting"

            if is_bat:
                rr_perf = sub[sub["striker_id"].isin(retained_ids)]
                all_perf = sub
                # Metric: Strike rate and WPA per ball
                rr_balls = len(rr_perf)
                rr_metric = (
                    round((rr_perf["batter_runs"].sum() / rr_balls * 100.0), 1)
                    if rr_balls >= 10
                    else 145.0
                )
                league_metric = round((all_perf["batter_runs"].sum() / max(1, len(all_perf)) * 100.0), 1)

                # Benchmark rating out of 100
                rr_score = np.clip((rr_metric - 110.0) / 100.0 * 100.0, 20.0, 95.0)
                league_score = np.clip((league_metric - 110.0) / 100.0 * 100.0, 20.0, 95.0)
                deficit = league_score - rr_score
                status = "Critical Gap" if deficit > 10 else ("Moderate Gap" if deficit > 0 else "Strong Asset")

                # Recommend top 3 non-RR auction targets from data
                target_cand = (
                    batters_df[~batters_df["striker_id"].isin(retained_ids)]
                    .head(10)
                )
                recommendations = []
                for _, cand in target_cand.head(3).iterrows():
                    recommendations.append({
                        "player_id": int(cand["striker_id"]),
                        "player_name": cand["striker_name_clean"],
                        "role": cand["role"],
                        "score": float(cand["reliability_score"]),
                        "stat_line": f"Pressure SR: {cand['pressure_sr']:.1f} (Delta: +{cand['sr_delta']:.1f})",
                        "fit_rationale": f"Supplies elite {s_data['name']} execution; fills RR tactical deficit."
                    })

            else:
                rr_perf = sub[sub["bowler_id"].isin(retained_ids)]
                all_perf = sub
                rr_balls = len(rr_perf)
                rr_metric = (
                    round((rr_perf["total_runs_on_ball"].sum() / rr_balls * 6.0), 2)
                    if rr_balls >= 10
                    else 10.5
                )
                league_metric = round((all_perf["total_runs_on_ball"].sum() / max(1, len(all_perf)) * 6.0), 2)

                # For bowling, lower economy is better
                rr_score = np.clip((13.0 - rr_metric) / 6.0 * 100.0, 20.0, 95.0)
                league_score = np.clip((13.0 - league_metric) / 6.0 * 100.0, 20.0, 95.0)
                deficit = league_score - rr_score
                status = "Critical Gap" if deficit > 10 else ("Moderate Gap" if deficit > 0 else "Strong Asset")

                target_cand = (
                    bowlers_df[~bowlers_df["bowler_id"].isin(retained_ids)]
                    .head(10)
                )
                recommendations = []
                for _, cand in target_cand.head(3).iterrows():
                    recommendations.append({
                        "player_id": int(cand["bowler_id"]),
                        "player_name": cand["bowler_name_clean"],
                        "role": cand["role"],
                        "score": float(cand["reliability_score"]),
                        "stat_line": f"Pressure Econ: {cand['pressure_econ']:.2f} (Delta: {cand['econ_delta']:+.2f})",
                        "fit_rationale": f"Provides proven {s_data['name']} control; solves RR vulnerability."
                    })

            gap_analysis.append({
                "scenario_key": s_key,
                "scenario_name": s_data["name"],
                "description": s_data["description"],
                "type": s_data["type"],
                "rr_score": round(float(rr_score), 1),
                "league_benchmark": round(float(league_score), 1),
                "deficit": round(float(deficit), 1),
                "status": status,
                "recommended_targets": recommendations,
            })

        # Rank gaps from biggest deficit to strongest
        gap_analysis = sorted(gap_analysis, key=lambda x: x["deficit"], reverse=True)

        return {
            "retained_count": len(retained_ids),
            "retained_players": self.rr_config.get("retentions", []),
            "tactical_gaps": gap_analysis,
        }
