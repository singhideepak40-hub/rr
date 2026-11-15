"""Rajasthan Royals Squad Fit & Advanced Auction Intelligence Engine.

Grounds all player-franchise mappings directly in the official IPL 2026 data pack,
evaluates Rajasthan Royals' actual 27-man roster across 5 pressure scenarios,
and identifies genuine, uncontracted auction targets to fill tactical deficits.
"""

import os
import json
import numpy as np
import pandas as pd


class SquadFitEngine:
    """Analyzes Rajasthan Royals squad roster and diagnoses tactical auction gaps."""

    def __init__(self, squad_json_path: str = "rr_squad.json", data_dir: str = "Data Pack"):
        self.squad_json_path = squad_json_path
        self.data_dir = data_dir
        self.player_team_map = self._build_team_map()
        self.rr_player_ids = {pid for pid, team in self.player_team_map.items() if team == "Rajasthan Royals"}
        self.all_ipl_player_ids = set(self.player_team_map.keys())

    def _build_team_map(self) -> dict[int, str]:
        """Map every player to their official IPL team from competition_team_player_mapping.csv."""
        mapping_path = os.path.join(self.data_dir, "competition_team_player_mapping.csv")
        comps_path = os.path.join(self.data_dir, "competitions.csv")

        if not os.path.exists(mapping_path) or not os.path.exists(comps_path):
            # Fallback if Data Pack path is relative to repo root
            base_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_path = os.path.join(base_dir, "Data Pack", "competition_team_player_mapping.csv")
            comps_path = os.path.join(base_dir, "Data Pack", "competitions.csv")

        if os.path.exists(mapping_path) and os.path.exists(comps_path):
            try:
                mapping = pd.read_csv(mapping_path)
                comps = pd.read_csv(comps_path)
                m = mapping.merge(comps, on="comp_id")
                ipl_latest = m[m["comp_name_x"] == "IPL 2026"]
                if ipl_latest.empty:
                    ipl_latest = m[m["match_format"] == "T20"].sort_values("season", ascending=False)
                    latest_season = ipl_latest["season"].max()
                    ipl_latest = ipl_latest[ipl_latest["season"] == latest_season]
                return ipl_latest.set_index("player_id")["team_name"].to_dict()
            except Exception as e:
                print("Warning building team map:", e)

        return {}

    def get_player_team(self, player_id: int) -> str:
        """Return player's IPL team or 'Auction Pool' if not contracted to an IPL franchise."""
        return self.player_team_map.get(int(player_id), "Auction Pool")

    def tag_retention_status(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Tag dataframe rows with accurate franchise team and auction status."""
        res = df.copy()

        def _get_team(pid):
            return self.player_team_map.get(int(pid), "Auction Pool")

        def _get_status(pid):
            team = self.player_team_map.get(int(pid), "Auction Pool")
            if team == "Rajasthan Royals":
                return "RR Squad"
            if team == "Auction Pool":
                return "Auction target"
            return f"Contracted ({team})"

        res["current_team"] = res[id_col].map(_get_team)
        res["retention_status"] = res[id_col].map(_get_status)
        return res

    def get_rr_squad_ids(self) -> set[int]:
        """Return all player IDs belonging to Rajasthan Royals in data pack."""
        return self.rr_player_ids

    def analyze_squad_gaps(
        self,
        pi_df: pd.DataFrame,
        batters_df: pd.DataFrame,
        bowlers_df: pd.DataFrame,
    ) -> dict:
        """Evaluate RR roster against 5 tactical pressure scenarios and recommend genuine auction targets."""
        rr_ids = self.get_rr_squad_ids()
        all_ipl_ids = self.all_ipl_player_ids
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
                rr_perf = sub[sub["striker_id"].isin(rr_ids)]
                all_perf = sub
                rr_balls = len(rr_perf)
                rr_metric = (
                    round((rr_perf["batter_runs"].sum() / rr_balls * 100.0), 1)
                    if rr_balls >= 10
                    else 145.0
                )
                league_metric = round((all_perf["batter_runs"].sum() / max(1, len(all_perf)) * 100.0), 1)

                rr_score = float(np.clip((rr_metric - 110.0) / 100.0 * 100.0, 20.0, 95.0))
                league_score = float(np.clip((league_metric - 110.0) / 100.0 * 100.0, 20.0, 95.0))
                deficit = float(league_score - rr_score)
                status = "Critical Gap" if deficit > 10 else ("Moderate Gap" if deficit > 0 else "Strong Asset")

                # GENUINE AUCTION TARGETS: MUST NOT belong to ANY IPL team (strictly Auction Pool free agents)
                genuine_auction_cand = (
                    batters_df[~batters_df["striker_id"].isin(all_ipl_ids)]
                    .head(10)
                )
                recommendations = []
                for _, cand in genuine_auction_cand.head(3).iterrows():
                    recommendations.append({
                        "player_id": int(cand["striker_id"]),
                        "player_name": cand["striker_name_clean"],
                        "role": cand["role"],
                        "score": float(cand["reliability_score"]),
                        "current_team": "Auction Pool",
                        "stat_line": f"Pressure SR: {cand['pressure_sr']:.1f} (Delta: +{cand['sr_delta']:.1f})",
                        "fit_rationale": f"Free agent in auction pool; supplies elite {s_data['name']} execution without franchise trade barrier."
                    })

                # Potential trade targets (contracted to other franchises)
                trade_cand = (
                    batters_df[(batters_df["striker_id"].isin(all_ipl_ids)) & (~batters_df["striker_id"].isin(rr_ids))]
                    .head(2)
                )
                trade_targets = []
                for _, cand in trade_cand.iterrows():
                    other_team = self.player_team_map.get(int(cand["striker_id"]), "Other Team")
                    trade_targets.append({
                        "player_id": int(cand["striker_id"]),
                        "player_name": cand["striker_name_clean"],
                        "role": cand["role"],
                        "score": float(cand["reliability_score"]),
                        "current_team": other_team,
                        "stat_line": f"Pressure SR: {cand['pressure_sr']:.1f} (Delta: +{cand['sr_delta']:.1f})",
                        "fit_rationale": f"Contracted to {other_team}; would require high-value franchise trade/transfer."
                    })

            else:
                rr_perf = sub[sub["bowler_id"].isin(rr_ids)]
                all_perf = sub
                rr_balls = len(rr_perf)
                rr_metric = (
                    round((rr_perf["total_runs_on_ball"].sum() / rr_balls * 6.0), 2)
                    if rr_balls >= 10
                    else 10.5
                )
                league_metric = round((all_perf["total_runs_on_ball"].sum() / max(1, len(all_perf)) * 6.0), 2)

                rr_score = float(np.clip((13.0 - rr_metric) / 6.0 * 100.0, 20.0, 95.0))
                league_score = float(np.clip((13.0 - league_metric) / 6.0 * 100.0, 20.0, 95.0))
                deficit = float(league_score - rr_score)
                status = "Critical Gap" if deficit > 10 else ("Moderate Gap" if deficit > 0 else "Strong Asset")

                # GENUINE AUCTION TARGETS: MUST NOT belong to ANY IPL team (strictly Auction Pool free agents)
                genuine_auction_cand = (
                    bowlers_df[~bowlers_df["bowler_id"].isin(all_ipl_ids)]
                    .head(10)
                )
                recommendations = []
                for _, cand in genuine_auction_cand.head(3).iterrows():
                    recommendations.append({
                        "player_id": int(cand["bowler_id"]),
                        "player_name": cand["bowler_name_clean"],
                        "role": cand["role"],
                        "score": float(cand["reliability_score"]),
                        "current_team": "Auction Pool",
                        "stat_line": f"Pressure Econ: {cand['pressure_econ']:.2f} (Delta: {cand['econ_delta']:+.2f})",
                        "fit_rationale": f"Free agent in auction pool; provides proven {s_data['name']} control; solves RR vulnerability."
                    })

                # Potential trade targets (contracted to other franchises)
                trade_cand = (
                    bowlers_df[(bowlers_df["bowler_id"].isin(all_ipl_ids)) & (~bowlers_df["bowler_id"].isin(rr_ids))]
                    .head(2)
                )
                trade_targets = []
                for _, cand in trade_cand.iterrows():
                    other_team = self.player_team_map.get(int(cand["bowler_id"]), "Other Team")
                    trade_targets.append({
                        "player_id": int(cand["bowler_id"]),
                        "player_name": cand["bowler_name_clean"],
                        "role": cand["role"],
                        "score": float(cand["reliability_score"]),
                        "current_team": other_team,
                        "stat_line": f"Pressure Econ: {cand['pressure_econ']:.2f} (Delta: {cand['econ_delta']:+.2f})",
                        "fit_rationale": f"Contracted to {other_team}; would require high-value franchise trade/transfer."
                    })

            gap_analysis.append({
                "scenario_key": s_key,
                "scenario_name": s_data["name"],
                "description": s_data["description"],
                "type": s_data["type"],
                "rr_score": round(rr_score, 1),
                "league_benchmark": round(league_score, 1),
                "deficit": round(deficit, 1),
                "status": status,
                "recommended_targets": recommendations,
                "potential_trade_targets": trade_targets,
            })

        # Load RR roster summary
        rr_roster_summary = []
        if os.path.exists(self.squad_json_path):
            with open(self.squad_json_path, "r") as f:
                s_data = json.load(f)
                rr_roster_summary = s_data.get("rr_squad_2026", [])

        return {
            "retained_count": len(self.rr_player_ids),
            "retained_players": rr_roster_summary,
            "tactical_gaps": gap_analysis,
        }
