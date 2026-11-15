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
        # Official retentions across the other 9 IPL franchises for IPL Mega Auction
        self.other_teams_retained_ids = {
            # Mumbai Indians
            2933, # Jasprit Bumrah
            1699, # Suryakumar Yadav
            126,  # Rohit Sharma
            3330, # Hardik Pandya
            7897, # Tilak Varma
            # Sunrisers Hyderabad
            4260, # Heinrich Klaasen
            1758, # Pat Cummins
            6281, # Abhishek Sharma
            2367, # Travis Head
            23511,# Nitish Kumar Reddy
            # Kolkata Knight Riders
            781,  # Sunil Narine
            4708, # Rinku Singh
            1318, # Andre Russell
            29888,# Harshit Rana
            20006,# Ramandeep Singh
            # Gujarat Titans
            4722, # Rashid Khan
            6288, # Shubman Gill
            23450,# Sai Sudharsan
            3926, # Rahul Tewatia
            3782, # Shahrukh Khan
            # Delhi Capitals
            2876, # Axar Patel
            3644, # Kuldeep Yadav
            20702,# Tristan Stubbs
            19714,# Abishek Porel
            # Chennai Super Kings
            6059, # Ruturaj Gaikwad
            8661, # Matheesha Pathirana
            4755, # Shivam Dube
            152,  # Ravindra Jadeja
            8,    # MS Dhoni
            # Royal Challengers Bengaluru
            47,   # Virat Kohli
            5039, # Rajat Patidar
            7322, # Yash Dayal
            # Lucknow Super Giants
            3125, # Nicholas Pooran
            27960,# Mayank Yadav
            7503, # Ravi Bishnoi
            6460, # Ayush Badoni
            6669, # Mohsin Khan
            # Punjab Kings
            2756, # Shashank Singh
            7216, # Prabhsimran Singh
        }

    def _load_config(self) -> dict:
        """Load RR squad configuration from JSON."""
        if not os.path.exists(self.squad_json_path):
            # Default fallback
            return {
                "retentions": [
                    {"player_id": 2258, "player_name": "Sanju Samson", "role": "Batter / Wicketkeeper"},
                    {"player_id": 6819, "player_name": "Yashasvi Jaiswal", "role": "Batter"},
                    {"player_id": 6423, "player_name": "Riyan Parag", "role": "All-Rounder"},
                    {"player_id": 7260, "player_name": "Dhruv Jurel", "role": "Batter / Wicketkeeper"},
                    {"player_id": 3297, "player_name": "Shimron Hetmyer", "role": "Batter"},
                    {"player_id": 2411, "player_name": "Sandeep Sharma", "role": "Bowler"},
                ],
                "recent_squad_pool": []
            }
        with open(self.squad_json_path, "r") as f:
            return json.load(f)

    def save_config(self, new_config: dict):
        """Save updated squad configuration to JSON."""
        with open(self.squad_json_path, "w") as f:
            json.dump(new_config, f, indent=2)
        self.rr_config = new_config

    def get_retained_ids(self) -> set[int]:
        """Return set of retained player IDs for Rajasthan Royals."""
        return {int(p["player_id"]) for p in self.rr_config.get("retentions", [])}

    def get_all_retained_ids(self) -> set[int]:
        """Return union of RR retentions and all other franchises' retentions."""
        return self.get_retained_ids() | self.other_teams_retained_ids

    def tag_retention_status(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Tag dataframe rows as 'Retained (RR)', 'Retained (Other Team)', or 'Auction target'."""
        rr_retained = self.get_retained_ids()
        other_retained = self.other_teams_retained_ids
        res = df.copy()

        def _get_status(val):
            iv = int(val)
            if iv in rr_retained:
                return "Retained (RR)"
            if iv in other_retained:
                return "Retained (Other Team)"
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
                "filter": (press["phase"] == "Powerplay") & (press["wickets_lost"] >= 1),
                "description": "Rebuilding momentum and par preservation when early top-order wickets fall.",
            },
        }

        all_retained_ids = self.get_all_retained_ids()
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

                # Recommend top 3 non-retained auction targets (excluding all teams' retentions)
                target_cand = (
                    batters_df[~batters_df["striker_id"].isin(all_retained_ids)]
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
                        "fit_rationale": f"Supplies elite {s_data['name']} execution; available auction target to fill RR deficit."
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
                    bowlers_df[~bowlers_df["bowler_id"].isin(all_retained_ids)]
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
                        "fit_rationale": f"Provides proven {s_data['name']} control; available auction target solving RR vulnerability."
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
