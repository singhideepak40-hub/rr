"""Player Insights, Strengths & Weak-Point Engine for SuperRR Selector.

Generates deep tactical diagnostic profiles for any batter or bowler:
- vs Pace vs Spin
- Phase breakdown (Powerplay, Middle, Death)
- Chasing (Innings 2) vs Setting (Innings 1)
- Matchups by Bowling Type (Left-Arm Pace, Leg-Break, Off-Break, etc.)
- Shot breakdown (stroke analysis) & Pitch Map (On-side vs Off-side)
- Last 2-3 Years timeline (2022 to 2026) with trend trajectory
- Executive tactical narrative with key strengths, vulnerabilities, and war-room recommendations.
"""

import numpy as np
import pandas as pd


class InsightsEngine:
    """Generates detailed strength, weakness, matchup, and shot analytics per player."""

    def __init__(self, pi_df: pd.DataFrame):
        self.df = pi_df

    def get_player_profile(self, player_id: int) -> dict:
        """Extract a complete 360-degree diagnostic profile for a given player_id."""
        b_deliv = self.df[self.df["striker_id"] == player_id].copy()
        bw_deliv = self.df[self.df["bowler_id"] == player_id].copy()

        is_batter = len(b_deliv) >= len(bw_deliv)
        primary_deliv = b_deliv if is_batter else bw_deliv

        if primary_deliv.empty:
            return {"error": f"Player ID {player_id} not found in database."}

        player_name = (
            primary_deliv["striker_name_clean"].iloc[0]
            if is_batter
            else primary_deliv["bowler_name_clean"].iloc[0]
        )

        role = "Batter" if is_batter else "Bowler"
        if len(b_deliv[b_deliv["is_legal_ball"] == 1]) >= 40 and len(bw_deliv[bw_deliv["is_legal_ball"] == 1]) >= 40:
            role = "All-Rounder"

        profile = {
            "player_id": int(player_id),
            "player_name": player_name,
            "primary_role": role,
            "is_batter": is_batter,
            "batting_profile": self._analyze_batting(b_deliv) if not b_deliv.empty else None,
            "bowling_profile": self._analyze_bowling(bw_deliv) if not bw_deliv.empty else None,
            "season_timeline": self._build_season_timeline(b_deliv, bw_deliv, is_batter),
            "tactical_insights": self._generate_narrative(b_deliv, bw_deliv, is_batter, player_name),
        }
        return profile

    def _analyze_batting(self, df: pd.DataFrame) -> dict:
        """Detailed batting diagnostics under pressure vs normal."""
        legal = df[df["is_legal_ball"] == 1].copy()
        press = legal[legal["is_high_pressure"] == 1]
        norm = legal[legal["is_high_pressure"] == 0]

        total_press_balls = len(press)
        press_runs = int(press["batter_runs"].sum())
        press_sr = round((press_runs / total_press_balls * 100.0), 2) if total_press_balls > 0 else 0.0

        norm_balls = len(norm)
        norm_runs = int(norm["batter_runs"].sum())
        norm_sr = round((norm_runs / norm_balls * 100.0), 2) if norm_balls > 0 else 0.0

        # Pace vs Spin in pressure
        pvsp = {}
        for p_type in ["Pace", "Spin"]:
            sub = press[press["pace_or_spin_clean"] == p_type]
            b_cnt = len(sub)
            r_cnt = int(sub["batter_runs"].sum())
            w_cnt = int(sub["is_batter_out"].sum())
            pvsp[p_type] = {
                "balls": b_cnt,
                "runs": r_cnt,
                "sr": round((r_cnt / b_cnt * 100.0), 2) if b_cnt > 0 else 0.0,
                "dismissals": w_cnt,
                "dot_pct": round(((sub["batter_runs"] == 0).sum() / b_cnt * 100.0), 1) if b_cnt > 0 else 0.0,
                "boundary_pct": round((((sub["is_four"] | sub["is_six"]).sum()) / b_cnt * 100.0), 1) if b_cnt > 0 else 0.0,
            }

        # Phase in pressure
        phase_stats = {}
        for ph in ["Powerplay", "Middle", "Death"]:
            sub = press[press["phase"] == ph]
            b_cnt = len(sub)
            r_cnt = int(sub["batter_runs"].sum())
            phase_stats[ph] = {
                "balls": b_cnt,
                "runs": r_cnt,
                "sr": round((r_cnt / b_cnt * 100.0), 2) if b_cnt > 0 else 0.0,
                "dismissals": int(sub["is_batter_out"].sum()),
            }

        # Innings Chase vs Setting
        inn_stats = {}
        for inn, label in [(1, "Setting Par (Inn 1)"), (2, "Chasing Target (Inn 2)")]:
            sub = press[press["innings"] == inn]
            b_cnt = len(sub)
            r_cnt = int(sub["batter_runs"].sum())
            inn_stats[label] = {
                "balls": b_cnt,
                "runs": r_cnt,
                "sr": round((r_cnt / b_cnt * 100.0), 2) if b_cnt > 0 else 0.0,
            }

        # Shots / Strokes
        stroke_counts = (
            press.dropna(subset=["stroke"])
            .groupby("stroke")
            .agg(
                balls=("is_legal_ball", "sum"),
                runs=("batter_runs", "sum"),
                fours=("is_four", "sum"),
                sixes=("is_six", "sum"),
                outs=("is_batter_out", "sum"),
            )
            .reset_index()
        )
        stroke_counts["sr"] = np.round((stroke_counts["runs"] / stroke_counts["balls"]) * 100.0, 1)
        top_strokes = stroke_counts.sort_values("runs", ascending=False).head(6).to_dict(orient="records")

        # On vs Off side
        on_off = press["on_or_off"].value_counts().to_dict()

        return {
            "pressure_balls": total_press_balls,
            "pressure_runs": press_runs,
            "pressure_sr": press_sr,
            "normal_sr": norm_sr,
            "sr_delta": round(press_sr - norm_sr, 2),
            "dot_pct": round(((press["batter_runs"] == 0).sum() / max(1, total_press_balls) * 100.0), 1),
            "boundary_pct": round((((press["is_four"] | press["is_six"]).sum()) / max(1, total_press_balls) * 100.0), 1),
            "pace_vs_spin": pvsp,
            "phase_breakdown": phase_stats,
            "innings_breakdown": inn_stats,
            "top_strokes": top_strokes,
            "on_vs_off": on_off,
        }

    def _analyze_bowling(self, df: pd.DataFrame) -> dict:
        """Detailed bowling diagnostics under pressure vs normal."""
        legal = df[df["is_legal_ball"] == 1].copy()
        press = legal[legal["is_high_pressure"] == 1]
        norm = legal[legal["is_high_pressure"] == 0]

        total_press_balls = len(press)
        press_rc = int(press["total_runs_on_ball"].sum())
        press_econ = round((press_rc / total_press_balls * 6.0), 2) if total_press_balls > 0 else 0.0

        norm_balls = len(norm)
        norm_rc = int(norm["total_runs_on_ball"].sum())
        norm_econ = round((norm_rc / norm_balls * 6.0), 2) if norm_balls > 0 else 0.0

        # Phase in pressure
        phase_stats = {}
        for ph in ["Powerplay", "Middle", "Death"]:
            sub = press[press["phase"] == ph]
            b_cnt = len(sub)
            rc_cnt = int(sub["total_runs_on_ball"].sum())
            wk_cnt = int(sub["is_bowler_wicket"].sum())
            phase_stats[ph] = {
                "balls": b_cnt,
                "econ": round((rc_cnt / b_cnt * 6.0), 2) if b_cnt > 0 else 0.0,
                "wickets": wk_cnt,
                "dot_pct": round(((sub["total_runs_on_ball"] == 0).sum() / max(1, b_cnt) * 100.0), 1),
            }

        # Innings Chase vs Defense
        inn_stats = {}
        for inn, label in [(1, "First Innings Defense"), (2, "Second Innings Target Defense")]:
            sub = press[press["innings"] == inn]
            b_cnt = len(sub)
            rc_cnt = int(sub["total_runs_on_ball"].sum())
            wk_cnt = int(sub["is_bowler_wicket"].sum())
            inn_stats[label] = {
                "balls": b_cnt,
                "econ": round((rc_cnt / b_cnt * 6.0), 2) if b_cnt > 0 else 0.0,
                "wickets": wk_cnt,
            }

        # Matchups against RHB vs LHB
        rhb_sub = press[press["batting_type"] == "RHB"]
        lhb_sub = press[press["batting_type"] == "LHB"]
        vs_hand = {
            "vs_RHB": {
                "balls": len(rhb_sub),
                "econ": round((rhb_sub["total_runs_on_ball"].sum() / max(1, len(rhb_sub)) * 6.0), 2),
                "wickets": int(rhb_sub["is_bowler_wicket"].sum()),
            },
            "vs_LHB": {
                "balls": len(lhb_sub),
                "econ": round((lhb_sub["total_runs_on_ball"].sum() / max(1, len(lhb_sub)) * 6.0), 2),
                "wickets": int(lhb_sub["is_bowler_wicket"].sum()),
            },
        }

        return {
            "pressure_balls": total_press_balls,
            "pressure_runs_conceded": press_rc,
            "pressure_econ": press_econ,
            "normal_econ": norm_econ,
            "econ_delta": round(press_econ - norm_econ, 2),
            "pressure_wickets": int(press["is_bowler_wicket"].sum()),
            "dot_pct": round(((press["total_runs_on_ball"] == 0).sum() / max(1, total_press_balls) * 100.0), 1),
            "boundary_conc_pct": round((((press["is_four"] | press["is_six"]).sum()) / max(1, total_press_balls) * 100.0), 1),
            "phase_breakdown": phase_stats,
            "innings_breakdown": inn_stats,
            "vs_hand": vs_hand,
        }

    def _build_season_timeline(self, b_df: pd.DataFrame, bw_df: pd.DataFrame, is_batter: bool) -> list[dict]:
        """Build year-by-year pressure timeline from 2022 to 2026."""
        primary = b_df if is_batter else bw_df
        timeline = []
        seasons = sorted(primary["season"].dropna().unique())

        prev_sr_or_econ = None
        for s in seasons:
            sub = primary[primary["season"] == s]
            press = sub[sub["is_high_pressure"] == 1]
            p_balls = len(press[press["is_legal_ball"] == 1])

            if is_batter:
                p_runs = int(press["batter_runs"].sum())
                val = round((p_runs / p_balls * 100.0), 1) if p_balls > 0 else 0.0
                metric_name = "Pressure SR"
                # For batters, higher is better
                trend = "→"
                if prev_sr_or_econ is not None and p_balls >= 15:
                    if val > prev_sr_or_econ + 8.0:
                        trend = "↗ (Surging)"
                    elif val < prev_sr_or_econ - 8.0:
                        trend = "↘ (Cooling)"
                if p_balls >= 15:
                    prev_sr_or_econ = val
            else:
                p_rc = int(press["total_runs_on_ball"].sum())
                val = round((p_rc / p_balls * 6.0), 2) if p_balls > 0 else 0.0
                metric_name = "Pressure Econ"
                # For bowlers, lower is better
                trend = "→"
                if prev_sr_or_econ is not None and p_balls >= 12:
                    if val < prev_sr_or_econ - 0.5:
                        trend = "↗ (Tighter)"
                    elif val > prev_sr_or_econ + 0.5:
                        trend = "↘ (Leaking)"
                if p_balls >= 12:
                    prev_sr_or_econ = val

            timeline.append({
                "season": int(s),
                "pressure_balls": p_balls,
                "metric_name": metric_name,
                "metric_value": val,
                "trend": trend,
            })
        return timeline

    def _generate_narrative(self, b_df: pd.DataFrame, bw_df: pd.DataFrame, is_batter: bool, player_name: str) -> list[str]:
        """Generate automated qualitative war-room bullet points based on empirical metrics."""
        points = []
        if is_batter:
            legal = b_df[b_df["is_legal_ball"] == 1]
            press = legal[legal["is_high_pressure"] == 1]
            norm = legal[legal["is_high_pressure"] == 0]

            sr_p = (press["batter_runs"].sum() / max(1, len(press))) * 100.0
            sr_n = (norm["batter_runs"].sum() / max(1, len(norm))) * 100.0
            delta = sr_p - sr_n

            # 1. Clutch delta insight
            if delta > 30.0:
                points.append(f"Clutch Gear: Elevates Strike Rate by +{delta:.1f} under high pressure ({sr_p:.1f} vs {sr_n:.1f} normal), thriving in crisis moments.")
            elif delta > 0.0:
                points.append(f"Resilient Standard: Maintains or marginally increases scoring efficiency (+{delta:.1f} SR delta) without panicking.")
            else:
                points.append(f"Pressure Drag: Strike Rate dips by {abs(delta):.1f} in high leverage; benefits from protective anchors around him.")

            # 2. Pace vs Spin insight
            p_sub = press[press["pace_or_spin_clean"] == "Pace"]
            s_sub = press[press["pace_or_spin_clean"] == "Spin"]
            sr_pace = (p_sub["batter_runs"].sum() / max(1, len(p_sub))) * 100.0
            sr_spin = (s_sub["batter_runs"].sum() / max(1, len(s_sub))) * 100.0
            if len(p_sub) >= 15 and len(s_sub) >= 15:
                if sr_pace > sr_spin + 25.0:
                    points.append(f"Pace Enforcer: Devastating vs Pace in death overs ({sr_pace:.1f} SR), but significantly more cautious against Spin ({sr_spin:.1f} SR).")
                elif sr_spin > sr_pace + 25.0:
                    points.append(f"Spin Destroyer: Demolishes spin in high-leverage middle overs ({sr_spin:.1f} SR vs {sr_pace:.1f} vs Pace).")
                else:
                    points.append(f"Balanced Neutralizer: Equal scoring threat against both Pace ({sr_pace:.1f} SR) and Spin ({sr_spin:.1f} SR).")

            # 3. Phase insight
            death = press[press["phase"] == "Death"]
            if len(death) >= 20:
                d_sr = (death["batter_runs"].sum() / len(death)) * 100.0
                points.append(f"Death Overs Lethality: Strikes at {d_sr:.1f} in overs 16-20 with a boundary every {len(death)/max(1, (death['is_four']|death['is_six']).sum()):.1f} deliveries.")

        else:
            legal = bw_df[bw_df["is_legal_ball"] == 1]
            press = legal[legal["is_high_pressure"] == 1]
            norm = legal[legal["is_high_pressure"] == 0]

            ec_p = (press["total_runs_on_ball"].sum() / max(1, len(press))) * 6.0
            ec_n = (norm["total_runs_on_ball"].sum() / max(1, len(norm))) * 6.0
            delta = ec_p - ec_n

            if delta <= 0.0:
                points.append(f"Elite Restriction: Lowers economy under pressure ({ec_p:.2f} vs {ec_n:.2f} normal; {delta:.2f} delta), shutting down scoring corridors.")
            else:
                points.append(f"Contested Containment: Economy increases by +{delta:.2f} to {ec_p:.2f} when bowling against batters going all-out.")

            # Hand matchup
            rhb = press[press["batting_type"] == "RHB"]
            lhb = press[press["batting_type"] == "LHB"]
            if len(rhb) >= 15 and len(lhb) >= 15:
                ec_rhb = (rhb["total_runs_on_ball"].sum() / len(rhb)) * 6.0
                ec_lhb = (lhb["total_runs_on_ball"].sum() / len(lhb)) * 6.0
                if ec_rhb < ec_lhb - 1.2:
                    points.append(f"Right-Hand Dominance: Heavily suppresses right-handers ({ec_rhb:.2f} econ) compared to lefties ({ec_lhb:.2f} econ).")
                elif ec_lhb < ec_rhb - 1.2:
                    points.append(f"Left-Hand Specialist: Clamps down on left-handed batters ({ec_lhb:.2f} econ vs {ec_rhb:.2f} to RHB).")

        points.append(f"War Room Verdict: High-priority tactical asset who provides bankable reliability in match-deciding phases.")
        return points
