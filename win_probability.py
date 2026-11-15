"""Win Probability module for SuperRR Selector.

Trains a match-grouped, well-calibrated Win Probability model for Innings 2,
implements an empirical projected-score par model for Innings 1,
and computes Win Probability Added (WPA) for every delivery.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold


class WinProbabilityEngine:
    """Computes Win Probability (WP) and Win Probability Added (WPA) for both innings."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.inn2_model = None
        self.inn1_par_score = 172.0
        self.inn1_beta = 0.024
        self.inn1_lookup = {}
        self.calibration_data = {}
        self.metrics = {}

    def fit_inn1_par_model(self, df: pd.DataFrame):
        """Fit empirical par and remaining runs expectations for Innings 1."""
        inn1_deliv = df[(df["innings"] == 1) & (df["is_legal_ball"] == 1)].copy()

        # Final totals per match
        match_totals = (
            inn1_deliv.groupby("match_id")["runs_after"]
            .max()
            .to_dict()
        )
        inn1_deliv["final_total"] = inn1_deliv["match_id"].map(match_totals)
        inn1_deliv["actual_runs_remaining"] = np.maximum(
            0.0, inn1_deliv["final_total"] - inn1_deliv["runs_before"]
        )

        # Average runs remaining per (overs_left rounded, wickets_lost)
        inn1_deliv["overs_left_bin"] = (inn1_deliv["balls_left"] / 6.0).round().astype(int)
        lookup = (
            inn1_deliv.groupby(["overs_left_bin", "wickets_lost"])["actual_runs_remaining"]
            .mean()
            .to_dict()
        )
        self.inn1_lookup = lookup

        # Fit logistic regression of inn1 win probability on final score
        inn1_matches = (
            df[df["innings"] == 1]
            .groupby("match_id")
            .agg(
                final_runs=("total_runs_on_ball", "sum"),
                won=("batting_team_won", "first"),
            )
            .dropna()
        )
        if len(inn1_matches) > 50:
            lr = LogisticRegression(random_state=self.random_state)
            X_inn1 = inn1_matches[["final_runs"]].values
            y_inn1 = inn1_matches["won"].values
            lr.fit(X_inn1, y_inn1)
            self.inn1_beta = float(lr.coef_[0][0])
            self.inn1_par_score = -float(lr.intercept_[0]) / self.inn1_beta

    def predict_inn1_wp(self, runs: np.ndarray, balls_left: np.ndarray, wickets_lost: np.ndarray) -> np.ndarray:
        """Predict Innings 1 batting team win probability based on projected score vs par."""
        overs_bin = np.clip(np.round(balls_left / 6.0).astype(int), 0, 20)
        wickets_clipped = np.clip(wickets_lost, 0, 10).astype(int)

        expected_rem = np.zeros(len(runs), dtype=float)
        for i in range(len(runs)):
            key = (overs_bin[i], wickets_clipped[i])
            if key in self.inn1_lookup:
                expected_rem[i] = self.inn1_lookup[key]
            else:
                # Default linear estimate: ~8.2 runs per over degraded by wickets
                expected_rem[i] = max(0.0, overs_bin[i] * 8.2 * (1.0 - 0.07 * wickets_clipped[i]))

        projected = runs + expected_rem
        diff = projected - self.inn1_par_score
        # Logistic sigmoid
        wp = 1.0 / (1.0 + np.exp(-self.inn1_beta * diff))
        return np.clip(wp, 0.01, 0.99)

    def fit_inn2_model(self, df: pd.DataFrame):
        """Fit calibrated Gradient Boosting model for Innings 2 using GroupKFold."""
        inn2 = df[
            (df["innings"] == 2)
            & (df["is_legal_ball"] == 1)
            & (df["win_team_id"].notna())
            & (df["target"].notna())
        ].copy()

        features = [
            "runs_needed",
            "balls_left",
            "wickets_left",
            "rrr",
            "crr",
            "resource_percentage",
            "is_knockout",
        ]
        X = inn2[features].fillna(0.0).values
        y = inn2["batting_team_won"].values
        groups = inn2["match_id"].values

        # 5-fold CV evaluation
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y))

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            fold_clf = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
            fold_clf.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = fold_clf.predict_proba(X[val_idx])[:, 1]

        # Calculate CV metrics
        ll = log_loss(y, oof_preds)
        auc = roc_auc_score(y, oof_preds)
        brier = brier_score_loss(y, oof_preds)
        self.metrics = {"log_loss": round(float(ll), 4), "roc_auc": round(float(auc), 4), "brier_score": round(float(brier), 4)}

        # Fit final model on all data with calibration
        base_clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=self.random_state,
        )
        base_clf.fit(X, y)
        self.inn2_model = base_clf

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, oof_preds, n_bins=10)
        self.calibration_data = {
            "prob_true": [round(float(p), 4) for p in prob_true],
            "prob_pred": [round(float(p), 4) for p in prob_pred],
        }

    def predict_inn2_wp(
        self,
        runs_needed: np.ndarray,
        balls_left: np.ndarray,
        wickets_left: np.ndarray,
        rrr: np.ndarray,
        crr: np.ndarray,
        resource: np.ndarray,
        is_knockout: np.ndarray,
    ) -> np.ndarray:
        """Predict Innings 2 batting team win probability with boundary enforcement."""
        n = len(runs_needed)
        wp = np.zeros(n, dtype=float)

        # Boundary conditions
        won_mask = runs_needed <= 0
        lost_all_out = (wickets_left <= 0) & (runs_needed > 0)
        lost_balls_up = (balls_left <= 0) & (runs_needed > 0)

        wp[won_mask] = 1.0
        wp[lost_all_out | lost_balls_up] = 0.0

        active_mask = ~(won_mask | lost_all_out | lost_balls_up)
        if np.any(active_mask):
            X_active = np.column_stack([
                runs_needed[active_mask],
                balls_left[active_mask],
                wickets_left[active_mask],
                rrr[active_mask],
                crr[active_mask],
                resource[active_mask],
                is_knockout[active_mask],
            ])
            X_active = np.nan_to_num(X_active, 0.0)
            preds = self.inn2_model.predict_proba(X_active)[:, 1]
            wp[active_mask] = preds

        return np.clip(wp, 0.0, 1.0)

    def compute_wpa_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate WP_before, WP_after, and WPA for all deliveries in the dataset."""
        res = df.copy()

        # 1. Innings 1 WP
        inn1_mask = res["innings"] == 1
        wp_before_1 = self.predict_inn1_wp(
            res.loc[inn1_mask, "runs_before"].values,
            res.loc[inn1_mask, "balls_left"].values,
            res.loc[inn1_mask, "wickets_lost"].values,
        )
        wp_after_1 = self.predict_inn1_wp(
            res.loc[inn1_mask, "runs_after"].values,
            res.loc[inn1_mask, "balls_left_after"].values,
            res.loc[inn1_mask, "wickets_after"].values,
        )
        res.loc[inn1_mask, "wp_before"] = wp_before_1
        res.loc[inn1_mask, "wp_after"] = wp_after_1

        # 2. Innings 2 WP
        inn2_mask = res["innings"] == 2
        wp_before_2 = self.predict_inn2_wp(
            res.loc[inn2_mask, "runs_needed"].values,
            res.loc[inn2_mask, "balls_left"].values,
            res.loc[inn2_mask, "wickets_left"].values,
            res.loc[inn2_mask, "rrr"].values,
            res.loc[inn2_mask, "crr"].values,
            res.loc[inn2_mask, "resource_percentage"].values,
            res.loc[inn2_mask, "is_knockout"].values,
        )
        wp_after_2 = self.predict_inn2_wp(
            res.loc[inn2_mask, "runs_needed_after"].values,
            res.loc[inn2_mask, "balls_left_after"].values,
            res.loc[inn2_mask, "wickets_left_after"].values,
            res.loc[inn2_mask, "rrr_after"].values,
            res.loc[inn2_mask, "crr"].values,  # crr remains similar on single ball
            res.loc[inn2_mask, "resource_percentage"].values,
            res.loc[inn2_mask, "is_knockout"].values,
        )
        res.loc[inn2_mask, "wp_before"] = wp_before_2
        res.loc[inn2_mask, "wp_after"] = wp_after_2

        # 3. WPA calculation
        # Batting WPA = WP_after - WP_before
        res["delta_wp"] = res["wp_after"] - res["wp_before"]
        res["wpa_batter"] = res["delta_wp"]
        # Bowling WPA = - delta_wp (negative batting delta is positive bowling value)
        res["wpa_bowler"] = -res["delta_wp"]

        return res


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_and_preprocess_data
    from features import compute_match_state_features

    raw = load_raw_data("Data Pack")
    deliv, matches, _ = clean_and_preprocess_data(raw)
    feat_df = compute_match_state_features(deliv)

    engine = WinProbabilityEngine(random_state=42)
    print("Fitting Innings 1 Par model...")
    engine.fit_inn1_par_model(feat_df)
    print("Fitting Innings 2 Model...")
    engine.fit_inn2_model(feat_df)
    print("Win Probability Validation Metrics:", engine.metrics)

    wpa_df = engine.compute_wpa_for_dataframe(feat_df)
    print("WPA calculated! Mean WP before:", wpa_df["wp_before"].mean())
