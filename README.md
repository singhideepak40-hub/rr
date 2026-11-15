# SuperRR Selector Season 2 — Rajasthan Royals Pressure Hub
**Round 1 Hackathon Submission: Identifying the Most Reliable Players Under Pressure**

An end-to-end, reproducible sports data science prototype and interactive analytics dashboard built for the Rajasthan Royals war room ahead of the IPL 2027 Mega Auction.

---

## Quickstart in 3 Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full data science pipeline (computes models, ratings, exports CSVs & plots)
python3 run_pipeline.py

# 3. Launch the FastAPI application & interactive web dashboard
python3 main.py
```
Open **http://localhost:8000** in your browser to access the full interactive war-room dashboard.

To run the unit and integration sanity test suite:
```bash
python3 tests/test_pipeline.py
```

---

## Official Round 1 Deliverables & Final Rankings

### Top 10 Batters Under Pressure (Selected on Unstop Dropdowns)

| Rank | Player ID | Player Name | Role | Pressure Balls | Pressure SR | SR Delta vs Normal | Reliability Score (0-100) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | 23450 | **Sai Sudharsan** | Batter | 282 | 216.31 | +80.82 | **87.29** |
| **2** | 6396 | **Tim David** | Batter | 598 | 202.34 | +50.05 | **87.27** |
| **3** | 6288 | **Shubman Gill** | Batter | 354 | 230.79 | +93.45 | **87.26** |
| **4** | 1699 | **Suryakumar Yadav** | Batter | 632 | 234.49 | +88.61 | **87.00** |
| **5** | 4260 | **Heinrich Klaasen** | Batter | 381 | 212.86 | +63.20 | **86.76** |
| **6** | 3647 | **Shreyas Iyer** | Batter | 428 | 208.41 | +71.35 | **85.64** |
| **7** | 4853 | **Glenn Phillips** | All-Rounder | 336 | 203.27 | +79.93 | **85.47** |
| **8** | 6819 | **Yashasvi Jaiswal** | Batter *(RR Retained)* | 210 | 236.19 | +90.91 | **85.21** |
| **9** | 5553 | **Phil Salt** | Batter | 139 | 240.29 | +77.09 | **85.01** |
| **10** | 5039 | **Rajat Patidar** | Batter | 252 | 233.73 | +90.66 | **84.44** |

### Top 10 Bowlers Under Pressure (Selected on Unstop Dropdowns)

| Rank | Player ID | Player Name | Role | Pressure Balls | Pressure Econ | Econ Delta vs Normal | Reliability Score (0-100) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | 5979 | **Abrar Ahmed** | Bowler | 175 | 7.13 | +0.60 | **76.86** |
| **2** | 6450 | **Mujeeb Ur Rahman** | Bowler | 111 | 7.46 | +0.60 | **74.72** |
| **3** | 34485 | **Sufiyan Muqeem** | Bowler | 89 | 6.67 | +0.74 | **73.13** |
| **4** | 2382 | **Jamie Overton** | All-Rounder | 157 | 8.75 | +0.56 | **72.95** |
| **5** | 8805 | **Ottneil Baartman** | Bowler | 80 | 8.70 | -0.26 | **71.43** |
| **6** | 7327 | **Rasikh Salam** | Bowler | 154 | 9.43 | +0.17 | **71.41** |
| **7** | 781 | **Sunil Narine** | All-Rounder | 335 | 8.29 | +1.77 | **71.12** |
| **8** | 3644 | **Kuldeep Yadav** | Bowler | 541 | 8.84 | +1.30 | **71.00** |
| **9** | 1219 | **David Willey** | All-Rounder | 68 | 9.00 | +1.40 | **70.56** |
| **10** | 2933 | **Jasprit Bumrah** | Bowler | 532 | 8.22 | +1.73 | **70.55** |

*All dropdown selections on Unstop match `outputs/top10_batters.csv` and `outputs/top10_bowlers.csv`.*

---

## Core Analytical Architecture

### 1. Unified Pressure / Leverage Index (PI)
$$\text{PI}_{raw} = \text{Urgency} \times \sqrt{\text{Closeness}} \times M_{stage} \times \text{InnCalibration} \times (0.60 + 0.40 \cdot |\Delta WP|)$$
- **Urgency:** $\frac{1}{\sqrt{\text{balls left} / 20 + 0.5}}$ — scales exponentially as match finishes near.
- **Closeness:** $4 \cdot WP \cdot (1 - WP)$ — peaks at 1.0 when the match is 50-50, dropping toward 0 in blowouts.
- **Stage Stakes Multiplier ($M_{stage}$):** 1.15x for Knockouts, Qualifiers, Eliminators, and Finals.
- **High Pressure Threshold:** Top **25%** highest leverage deliveries ($\text{PI} \ge 0.75$).

### 2. Probabilistic Win Probability Engine
- **Innings 2:** Supervised Gradient Boosted Decision Tree (`HistGradientBoostingClassifier`) cross-validated using **5-Fold GroupKFold by `match_id`** (zero leakage).
  - **ROC-AUC: 0.8871**, **Log-Loss: 0.4206**, **Brier Score: 0.1373**.
- **Innings 1:** Empirical par matrix with logistic score curve relative to par total ($172.0$ runs).
- **Win Probability Added (WPA):** Evaluated ball-by-ball ($\Delta WP = WP_{after} - WP_{before}$).

### 3. Empirical Bayes Shrinkage & Composite Reliability Score (0-100)
$$\theta_{shrunk} = \frac{N}{N + K} \theta_{observed} + \frac{K}{N + K} \mu_{league}$$
- **35% Pressure Caliber Level:** Shrunk WPA/ball + Shrunk Runs/Econ above expected.
- **25% Clutch Delta:** Performance under pressure vs player's own low-pressure baseline ($\Delta SR$ for batters, $-\Delta Econ$ for bowlers).
- **25% Match-to-Match Consistency:** Proportion of pressure matches with positive net contribution ($WPA > 0$ or $RAE > 0$).
- **15% Sample Size Confidence:** Saturation function penalizing low-sample flukes.

### 4. Eligibility Thresholds
- **Batters:** Minimum **60 legal balls faced** in High Pressure ($PI \ge 0.75$) AND $\ge 10$ pressure innings.
- **Bowlers:** Minimum **48 legal balls bowled** in High Pressure ($PI \ge 0.75$) AND $\ge 10$ pressure spells.

---

## Unique Rajasthan Royals Features

1. **Role Classification:** Automatic derivation of **Batter**, **Bowler**, and **All-Rounder** based on empirical ball volume ratio thresholds.
2. **Multi-Year Trendline (2022-2026):** Tracks player pressure trajectory across 5 seasons with trend vectors ($\nearrow$ improving, $\rightarrow$ stable, $\searrow$ declining).
3. **Player 360 Tactical Engine:** Automated natural-language breakdown covering:
   - vs Pace vs Spin (SR, dots, boundaries)
   - Phase execution (Powerplay, Middle, Death)
   - Chasing vs Setting totals
   - Shot execution wheel using raw delivery `stroke` data
   - Pitch corridor distribution (`on_or_off` data)
4. **Rajasthan Royals Squad Fit & Gap Analyzer:**
   - Evaluates RR's official retained core (Samson, Jaiswal, Parag, Jurel, Hetmyer, Sandeep Sharma) against 5 vital pressure dimensions:
     1. Death Bowling Lockdown (Overs 16-20) — *Critical Gap diagnosed*
     2. Powerplay Strike Pacing (Overs 1-6)
     3. Death Overs Finishing (Overs 16-20 Batting)
     4. Middle Overs Spin Counter-Attack (Overs 7-15 vs Spin)
     5. Top Order Crisis Resilience (Powerplay 1-6)
   - Recommends the Top 3 best-fit auction targets specifically suited to eliminate each tactical deficit.
   - Editable `rr_squad.json` with in-app live retention updates.
5. **Interactive Pressure Simulator:** Real-time what-if simulator adjusting Innings, Balls Left, Wickets Lost, Runs Needed, and Stakes to calculate Win Probability, Pressure Index, and inspect sample historical deliveries from matching game states.
6. **Multi-Player Radar Comparison:** Side-by-side radar overlay comparing 2 to 4 players across 6 normalized pressure metrics.

---

## Statistical Validation & Reliability
- **Split-Half Reliability (Odd vs Even Matches):** Batting Spearman $\rho = 0.2216$ ($p = 0.0028 < 0.01$). Statistically significant correlation confirms true skill measurement.
- **Temporal Stability (2022-24 vs 2025-26):** Batting $\rho = 0.1645$, Bowler $\rho = 0.1612$.
- **Bootstrap 95% Confidence Intervals:** 120 resamples by match to compute confidence boundaries for scores and ranks.
- **Threshold Sensitivity:** 60% Top 10 overlap maintained even when pressure threshold shifts from 20% to 30%.

---

## Directory Structure & Exports
```
super_rr_selector/
├── Data Pack/                      # Official 6 CSV files (201k balls, 867 matches)
├── artifacts/                      # Precomputed parquet & JSON caches for instant app load
│   ├── deliveries_pressure.parquet
│   ├── ratings_batters_combined.parquet
│   ├── ratings_bowlers_combined.parquet
│   └── validation_summary.json
├── outputs/                        # Official exported CSVs, charts, and report
│   ├── top10_batters.csv           # Final Top 10 Batters
│   ├── top10_bowlers.csv           # Final Top 10 Bowlers
│   ├── full_ratings.csv            # Complete player rankings
│   ├── methodology_notes.md        # Technical methodology report
│   ├── pi_distribution.png         # High-res chart
│   ├── wp_calibration.png          # High-res chart
│   ├── split_half_correlation.png  # High-res chart
│   ├── top10_batters_bars.png      # High-res chart
│   ├── top10_bowlers_bars.png      # High-res chart
│   └── player_radar_sample.png     # High-res chart
├── static/                         # Frontend SPA (Vanilla JS + CSS, Chart.js via CDN)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                          # 7-Suite automated sanity tests
│   └── test_pipeline.py
├── data_loader.py                  # Ingestion, cleaning, verification
├── features.py                     # Context baselines & state tracking
├── win_probability.py              # Probabilistic dual-innings models
├── pressure_index.py               # Unified PI leverage formulation
├── ratings.py                      # Bayesian reliability rating engine
├── validation.py                   # Statistical checks & chart generator
├── insights.py                     # Player 360 & tactical narrative generator
├── squad_fit.py                    # RR squad gap analyzer & auction targets
├── rr_squad.json                   # Official Rajasthan Royals retentions
├── main.py                         # FastAPI REST API & web server
├── run_pipeline.py                 # End-to-end execution script
└── requirements.txt                # Python dependencies
```
