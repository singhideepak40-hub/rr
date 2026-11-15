// SuperRR Selector Season 2 — Frontend Application Logic

let CURRENT_RANKINGS = [];
let ALL_PLAYERS_LIST = [];
let RADAR_CHART_INSTANCE = null;
let DEBOUNCE_TIMER = null;

// Tab Switching
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const tabId = "tab-" + btn.dataset.tab;
    const targetPane = document.getElementById(tabId);
    if (targetPane) targetPane.classList.add("active");

    if (btn.dataset.tab === "compare") {
      initCompareDropdowns();
    } else if (btn.dataset.tab === "squad") {
      fetchSquadGaps();
    } else if (btn.dataset.tab === "methodology") {
      fetchValidationSummary();
    }
  });
});

// Initial Page Load
window.addEventListener("DOMContentLoaded", () => {
  fetchRankings();
  fetchInitialPlayerList();
  updateSimulation();
});

// 1. RANKINGS
async function fetchRankings() {
  const cat = document.getElementById("filter-category").value;
  const fmt = document.getElementById("filter-format").value;
  const role = document.getElementById("filter-role").value;
  const teamEl = document.getElementById("filter-team");
  const team = teamEl ? teamEl.value : "all";
  const search = document.getElementById("filter-search").value.trim();

  let url = `/api/rankings?category=${cat}&format_mode=${fmt}&role=${role}&team=${encodeURIComponent(team)}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    CURRENT_RANKINGS = data.results;
    renderRankingsTable(cat, CURRENT_RANKINGS);
  } catch (err) {
    console.error("Failed to load rankings:", err);
  }
}

function debounceFetchRankings() {
  clearTimeout(DEBOUNCE_TIMER);
  DEBOUNCE_TIMER = setTimeout(fetchRankings, 250);
}

function renderRankingsTable(cat, rows) {
  const thead = document.getElementById("rankings-table-head");
  const tbody = document.getElementById("rankings-table-body");

  const isBat = cat === "batters";
  thead.innerHTML = `
    <th>Rank</th>
    <th>Player</th>
    <th>Role</th>
    <th>Current Franchise</th>
    <th>Pressure Balls</th>
    <th>${isBat ? "Pressure SR" : "Pressure Econ"}</th>
    <th>${isBat ? "SR Delta" : "Econ Delta"}</th>
    <th>Match Consistency</th>
    <th style="min-width: 140px;">Reliability Score</th>
  `;

  if (!rows || rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 2rem;">No players match the selected filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r, i) => {
    const rankClass = r.rank === 1 ? "badge-rank-1" : (r.rank === 2 ? "badge-rank-2" : (r.rank === 3 ? "badge-rank-3" : ""));
    
    let teamBadge = `<span class="badge badge-auction">🟢 Auction Target</span>`;
    if (r.current_team === "Rajasthan Royals") {
      teamBadge = `<span class="badge badge-rr">🩷 Rajasthan Royals</span>`;
    } else if (r.current_team && r.current_team !== "Auction Pool") {
      teamBadge = `<span class="badge badge-other-team">🔒 ${r.current_team}</span>`;
    }
    
    const roleBadge = r.role === "Batter"
      ? `<span class="badge badge-role-batter">Batter</span>`
      : (r.role === "Bowler" ? `<span class="badge badge-role-bowler">Bowler</span>` : `<span class="badge badge-role-ar">All-Rounder</span>`);

    const pName = isBat ? r.striker_name_clean : r.bowler_name_clean;
    const pId = isBat ? r.striker_id : r.bowler_id;
    const rateVal = isBat ? `${r.pressure_sr.toFixed(1)}` : `${r.pressure_econ.toFixed(2)}`;
    const deltaVal = isBat
      ? `<span style="color: ${r.sr_delta >= 0 ? 'var(--rr-green)' : 'var(--rr-red)'}; font-weight:700;">${r.sr_delta >= 0 ? '+' : ''}${r.sr_delta.toFixed(1)}</span>`
      : `<span style="color: ${r.econ_delta <= 0 ? 'var(--rr-green)' : 'var(--rr-red)'}; font-weight:700;">${r.econ_delta >= 0 ? '+' : ''}${r.econ_delta.toFixed(2)}</span>`;

    const consistencyPct = (r.positive_share * 100.0).toFixed(0);

    return `
      <tr style="cursor: pointer;" onclick="openPlayerProfile(${pId})">
        <td><span class="badge badge-rank ${rankClass}">${r.rank}</span></td>
        <td>
          <div style="font-weight: 700; color: #fff;">${pName}</div>
          <div style="font-size: 0.7rem; color: var(--text-muted);">ID: ${pId}</div>
        </td>
        <td>${roleBadge}</td>
        <td>${teamBadge}</td>
        <td><strong>${r.pressure_balls}</strong></td>
        <td>${rateVal}</td>
        <td>${deltaVal}</td>
        <td>${consistencyPct}% matches +</td>
        <td>
          <div class="score-cell">
            <span class="score-num">${r.reliability_score.toFixed(1)}</span>
            <div class="score-bar-bg">
              <div class="score-bar-fill" style="width: ${Math.min(100, Math.max(10, r.reliability_score))}%;"></div>
            </div>
          </div>
        </td>
      </tr>
    `;
  }).join("");
}

// 2. PLAYER PROFILE
async function fetchInitialPlayerList() {
  try {
    const res = await fetch("/api/rankings?category=batters&limit=150");
    const data = await res.json();
    ALL_PLAYERS_LIST = data.results;

    const select = document.getElementById("profile-player-select");
    select.innerHTML = ALL_PLAYERS_LIST.map((p) => {
      return `<option value="${p.striker_id}">${p.striker_name_clean} (${p.role} — Score: ${p.reliability_score.toFixed(1)})</option>`;
    }).join("");

    if (ALL_PLAYERS_LIST.length > 0) {
      loadProfileData(ALL_PLAYERS_LIST[0].striker_id);
    }
  } catch (err) {
    console.error("Player list load failed:", err);
  }
}

function openPlayerProfile(pid) {
  const navBtn = document.querySelector('[data-tab="profile"]');
  if (navBtn) navBtn.click();
  const select = document.getElementById("profile-player-select");
  if (select) select.value = pid;
  loadProfileData(pid);
}

function loadSelectedProfile() {
  const pid = document.getElementById("profile-player-select").value;
  if (pid) loadProfileData(pid);
}

async function loadProfileData(pid) {
  const container = document.getElementById("profile-content");
  container.innerHTML = `<p style="text-align:center; padding: 2rem;">Loading diagnostic profile for Player ID ${pid}...</p>`;

  try {
    const res = await fetch(`/api/player/${pid}`);
    const data = await res.json();
    renderPlayerProfile(data);
  } catch (err) {
    container.innerHTML = `<p style="color:var(--rr-red); text-align:center;">Failed to load player dossier.</p>`;
  }
}

function renderPlayerProfile(p) {
  const container = document.getElementById("profile-content");
  const bp = p.batting_profile;
  const bwp = p.bowling_profile;

  const narrativeHtml = (p.tactical_insights || []).map((t) => `<li>${t}</li>`).join("");

  const timelineRows = (p.season_timeline || []).map((s) => `
    <tr>
      <td><strong>${s.season}</strong></td>
      <td>${s.pressure_balls}</td>
      <td>${s.metric_name}: <strong>${s.metric_value}</strong></td>
      <td><span style="font-weight:700; color:var(--rr-gold);">${s.trend}</span></td>
    </tr>
  `).join("");

  let tacticalTablesHtml = "";
  if (p.is_batter && bp) {
    tacticalTablesHtml = `
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div class="card" style="padding:1rem;">
          <h4 style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.6rem;">vs Pace vs Spin in Pressure</h4>
          <table>
            <tr><td><strong>vs Pace:</strong></td><td>${bp.pace_vs_spin.Pace.balls} balls</td><td>${bp.pace_vs_spin.Pace.sr} SR</td><td>${bp.pace_vs_spin.Pace.dot_pct}% dots</td></tr>
            <tr><td><strong>vs Spin:</strong></td><td>${bp.pace_vs_spin.Spin.balls} balls</td><td>${bp.pace_vs_spin.Spin.sr} SR</td><td>${bp.pace_vs_spin.Spin.dot_pct}% dots</td></tr>
          </table>
        </div>
        <div class="card" style="padding:1rem;">
          <h4 style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.6rem;">Phase Execution</h4>
          <table>
            <tr><td>Powerplay (1-6):</td><td>${bp.phase_breakdown.Powerplay.balls}b</td><td>${bp.phase_breakdown.Powerplay.sr} SR</td></tr>
            <tr><td>Middle (7-15):</td><td>${bp.phase_breakdown.Middle.balls}b</td><td>${bp.phase_breakdown.Middle.sr} SR</td></tr>
            <tr><td>Death (16-20):</td><td>${bp.phase_breakdown.Death.balls}b</td><td>${bp.phase_breakdown.Death.sr} SR</td></tr>
          </table>
        </div>
      </div>

      <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
        <div class="card" style="padding:1rem;">
          <h4 style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.6rem;">Top Scoring Strokes (ball_by_ball data)</h4>
          <table>
            ${(bp.top_strokes || []).map((st) => `<tr><td><strong>${st.stroke}</strong></td><td>${st.balls} balls</td><td>${st.runs} runs</td><td>${st.sr} SR</td></tr>`).join("")}
          </table>
        </div>
        <div class="card" style="padding:1rem;">
          <h4 style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.6rem;">Pitch Corridor Distribution</h4>
          <table>
            <tr><td>Off-Side Deliveries:</td><td><strong>${bp.on_vs_off["Off-side"] || 0}</strong></td></tr>
            <tr><td>On-Side Deliveries:</td><td><strong>${bp.on_vs_off["On-side"] || 0}</strong></td></tr>
          </table>
        </div>
      </div>
    `;
  } else if (bwp) {
    tacticalTablesHtml = `
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div class="card" style="padding:1rem;">
          <h4 style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.6rem;">vs Batter Handedness</h4>
          <table>
            <tr><td>vs RHB:</td><td>${bwp.vs_hand.vs_RHB.balls}b</td><td>${bwp.vs_hand.vs_RHB.econ} Econ</td><td>${bwp.vs_hand.vs_RHB.wickets} Wkts</td></tr>
            <tr><td>vs LHB:</td><td>${bwp.vs_hand.vs_LHB.balls}b</td><td>${bwp.vs_hand.vs_LHB.econ} Econ</td><td>${bwp.vs_hand.vs_LHB.wickets} Wkts</td></tr>
          </table>
        </div>
        <div class="card" style="padding:1rem;">
          <h4 style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.6rem;">Phase Defense</h4>
          <table>
            <tr><td>Powerplay:</td><td>${bwp.phase_breakdown.Powerplay.balls}b</td><td>${bwp.phase_breakdown.Powerplay.econ} Econ</td></tr>
            <tr><td>Middle:</td><td>${bwp.phase_breakdown.Middle.balls}b</td><td>${bwp.phase_breakdown.Middle.econ} Econ</td></tr>
            <tr><td>Death:</td><td>${bwp.phase_breakdown.Death.balls}b</td><td>${bwp.phase_breakdown.Death.econ} Econ</td></tr>
          </table>
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.25rem;">
      <div>
        <h2 style="font-size:1.6rem; color:#fff;">${p.player_name}</h2>
        <span style="color:var(--text-muted); font-size:0.85rem;">Role: <strong>${p.primary_role}</strong> • ID: ${p.player_id}</span>
      </div>
    </div>

    <div style="display:grid; grid-template-columns: 2fr 1fr; gap:1.5rem;">
      <div>
        <div class="card" style="padding:1rem; margin-bottom:1rem;">
          <h3 style="font-size:0.95rem; margin-bottom:0.6rem; color:var(--rr-gold);">Tactical War-Room Insights</h3>
          <ul class="tactical-list">${narrativeHtml}</ul>
        </div>
        ${tacticalTablesHtml}
      </div>

      <div>
        <div class="card" style="padding:1rem;">
          <h3 style="font-size:0.95rem; margin-bottom:0.6rem; color:var(--rr-cyan);">Multi-Season Timeline (2022–26)</h3>
          <table>
            <thead><tr><th>Year</th><th>Balls</th><th>Metric</th><th>Trend</th></tr></thead>
            <tbody>${timelineRows}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

// 3. COMPARE PLAYERS
function initCompareDropdowns() {
  const p1 = document.getElementById("compare-p1");
  const p2 = document.getElementById("compare-p2");
  const p3 = document.getElementById("compare-p3");

  const opts = ALL_PLAYERS_LIST.map((p) => `<option value="${p.striker_id}">${p.striker_name_clean}</option>`).join("");
  p1.innerHTML = opts;
  p2.innerHTML = opts;
  p3.innerHTML = `<option value="">-- None --</option>` + opts;

  if (ALL_PLAYERS_LIST.length >= 2) {
    p1.value = ALL_PLAYERS_LIST[0].striker_id;
    p2.value = ALL_PLAYERS_LIST[1].striker_id;
    runComparison();
  }
}

async function runComparison() {
  const p1 = document.getElementById("compare-p1").value;
  const p2 = document.getElementById("compare-p2").value;
  const p3 = document.getElementById("compare-p3").value;

  const ids = [p1, p2, p3].filter((x) => x).join(",");
  try {
    const res = await fetch(`/api/compare?player_ids=${ids}`);
    const data = await res.json();
    renderRadarChart(data);
    renderComparisonTable(data);
  } catch (err) {
    console.error("Comparison load error:", err);
  }
}

function renderRadarChart(data) {
  const ctx = document.getElementById("radarChart").getContext("2d");
  if (RADAR_CHART_INSTANCE) RADAR_CHART_INSTANCE.destroy();

  const colors = [
    { border: "#ea1a7f", bg: "rgba(234, 26, 127, 0.25)" },
    { border: "#00d2ff", bg: "rgba(0, 210, 255, 0.25)" },
    { border: "#ffbe3d", bg: "rgba(255, 190, 61, 0.25)" },
  ];

  const datasets = data.radar_comparison.map((p, idx) => ({
    label: p.player_name,
    data: p.metrics,
    borderColor: colors[idx].border,
    backgroundColor: colors[idx].bg,
    borderWidth: 2.5,
    pointRadius: 4,
  }));

  RADAR_CHART_INSTANCE = new Chart(ctx, {
    type: "radar",
    data: {
      labels: data.dimensions,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: "rgba(255, 255, 255, 0.1)" },
          grid: { color: "rgba(255, 255, 255, 0.1)" },
          pointLabels: { color: "#fff", font: { size: 10, weight: "bold" } },
          ticks: { backdropColor: "transparent", color: "#8e9db7", stepSize: 20 },
          min: 0,
          max: 100,
        },
      },
      plugins: {
        legend: { labels: { color: "#fff", font: { weight: "bold" } } },
      },
    },
  });
}

function renderComparisonTable(data) {
  const container = document.getElementById("compare-stats-table");
  const headers = data.player_profiles.map((p) => `<th>${p.player_name}</th>`).join("");

  const rows = [
    "Role",
    "Pressure Balls",
    "Pressure SR / Econ",
    "Normal Baseline",
    "Clutch Delta",
    "Dot Ball %",
    "Boundary %",
  ];

  const getVal = (p, label) => {
    const bp = p.batting_profile;
    const bwp = p.bowling_profile;
    if (label === "Role") return p.primary_role;
    if (label === "Pressure Balls") return (bp ? bp.pressure_balls : bwp.pressure_balls);
    if (label === "Pressure SR / Econ") return bp ? `${bp.pressure_sr.toFixed(1)} SR` : `${bwp.pressure_econ.toFixed(2)} Econ`;
    if (label === "Normal Baseline") return bp ? `${bp.normal_sr.toFixed(1)} SR` : `${bwp.normal_econ.toFixed(2)} Econ`;
    if (label === "Clutch Delta") return bp ? `+${bp.sr_delta.toFixed(1)} SR` : `${bwp.econ_delta.toFixed(2)} Econ`;
    if (label === "Dot Ball %") return bp ? `${bp.dot_pct}%` : `${bwp.dot_pct}%`;
    if (label === "Boundary %") return bp ? `${bp.boundary_pct}%` : `${bwp.boundary_conc_pct}%`;
    return "-";
  };

  const tbody = rows.map((rLabel) => `
    <tr>
      <td><strong>${rLabel}</strong></td>
      ${data.player_profiles.map((p) => `<td>${getVal(p, rLabel)}</td>`).join("")}
    </tr>
  `).join("");

  container.innerHTML = `
    <table>
      <thead><tr><th>Metric</th>${headers}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
}

// 4. PRESSURE EXPLORER SIMULATOR
async function updateSimulation() {
  const inn = parseInt(document.getElementById("sim-innings").value);
  const ko = parseInt(document.getElementById("sim-knockout").value);
  const bLeft = parseInt(document.getElementById("sim-balls-left").value);
  const wLost = parseInt(document.getElementById("sim-wickets-lost").value);
  const rNeed = parseFloat(document.getElementById("sim-runs-needed").value);

  document.getElementById("val-balls-left").innerText = bLeft;
  document.getElementById("val-wickets-lost").innerText = wLost;
  document.getElementById("val-runs-needed").innerText = rNeed;

  if (inn === 1) {
    document.getElementById("label-runs-needed").innerText = "Current Score:";
  } else {
    document.getElementById("label-runs-needed").innerText = "Runs Needed to Win:";
  }

  const url = `/api/simulate-pressure?innings=${inn}&balls_left=${bLeft}&wickets_lost=${wLost}&runs_needed=${rNeed}&is_knockout=${ko}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    const sim = data.simulation;

    document.getElementById("meter-pi").innerText = sim.pressure_index.toFixed(3);
    document.getElementById("badge-bucket").innerText = sim.pressure_bucket;
    document.getElementById("meter-wp").innerText = sim.win_probability_pct;
    document.getElementById("meter-rrr").innerText = data.inputs.required_run_rate.toFixed(2);

    const descMap = {
      "Extreme Crisis": "Death overs chase (overs 16-20) with RRR > 10.0; single ball outcome swings WP drastically.",
      "High Pressure": "Tense middle overs with wickets down or powerplay collapses in knockouts.",
      "Moderate Pressure": "Balanced match state with manageable required rate.",
      "Low-Moderate": "Controlled phase where one team is noticeably ahead.",
      "Low Pressure / Garbage Time": "Match virtually decided; win probability near 0% or 100%.",
    };
    document.getElementById("desc-bucket").innerText = descMap[sim.pressure_bucket] || "";

    // Similar balls
    const simBalls = data.similar_historical_deliveries || [];
    document.getElementById("sim-similar-balls").innerHTML = simBalls.map((b) => `
      <div style="background:var(--bg-main); border:1px solid var(--border-color); border-radius:6px; padding:0.5rem 0.75rem; margin-bottom:0.4rem; font-size:0.75rem; display:flex; justify-content:space-between;">
        <span><strong>${b.competition}</strong> (Over ${b.over}): ${b.striker} vs ${b.bowler}</span>
        <span style="color:var(--rr-pink); font-weight:700;">Result: ${b.runs_scored} runs ${b.is_wicket ? '(WICKET)' : ''} (PI: ${b.historical_pi})</span>
      </div>
    `).join("");
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

// 5. SQUAD FIT (RR)
async function fetchSquadGaps() {
  const container = document.getElementById("squad-gap-container");
  container.innerHTML = `<p style="text-align:center; padding: 2rem;">Evaluating Rajasthan Royals squad pressure profiles...</p>`;

  try {
    const res = await fetch("/api/squad-fit");
    const data = await res.json();
    renderSquadGaps(data);
  } catch (err) {
    container.innerHTML = `<p style="color:var(--rr-red); text-align:center;">Failed to evaluate squad gaps.</p>`;
  }
}

function renderSquadGaps(data) {
  const container = document.getElementById("squad-gap-container");
  const gaps = data.tactical_gaps || [];

  container.innerHTML = gaps.map((g) => {
    const statusColor = g.status === "Critical Gap" ? "var(--rr-red)" : (g.status === "Moderate Gap" ? "var(--rr-gold)" : "var(--rr-green)");
    
    // Genuine free agent auction targets
    const targetCards = (g.recommended_targets || []).map((t) => `
      <div class="target-pill" style="border-left: 3px solid var(--rr-green);">
        <span style="color:#fff; font-weight:700;">${t.player_name} (${t.role})</span>
        <span style="color:var(--rr-green); font-size:0.7rem;">• 🟢 Free Agent (Auction Pool)</span>
        <span style="color:var(--rr-gold); font-size:0.7rem;">• Score: ${t.score.toFixed(1)}</span>
        <span style="color:var(--text-muted); font-size:0.65rem;">• ${t.stat_line}</span>
      </div>
    `).join("");

    // Franchise trade targets
    const tradeCards = (g.potential_trade_targets || []).map((t) => `
      <div class="target-pill" style="border-left: 3px solid #64748b;">
        <span style="color:#fff; font-weight:700;">${t.player_name} (${t.role})</span>
        <span style="color:#94a3b8; font-size:0.7rem;">• 🔒 Contracted to ${t.current_team}</span>
        <span style="color:var(--rr-gold); font-size:0.7rem;">• Score: ${t.score.toFixed(1)}</span>
        <span style="color:var(--text-muted); font-size:0.65rem;">• Trade Target</span>
      </div>
    `).join("");

    return `
      <div class="gap-item">
        <div class="gap-header">
          <div>
            <div class="gap-title">${g.scenario_name}</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">${g.description}</div>
          </div>
          <span class="badge" style="background:${statusColor}; color:#000;">${g.status}</span>
        </div>

        <div class="gap-progress">
          <span style="font-size:0.75rem; font-weight:700; width:100px;">RR Score: ${g.rr_score}</span>
          <div class="gap-bar-container">
            <div class="gap-bar-rr" style="width: ${Math.min(100, Math.max(10, g.rr_score))}%;"></div>
          </div>
          <span style="font-size:0.75rem; color:var(--text-muted); width:130px;">Benchmark: ${g.league_benchmark}</span>
        </div>

        <div style="font-size:0.75rem; font-weight:700; color:var(--rr-green); margin-top:0.6rem;">🟢 Top Free-Agent Auction Targets (To Buy at Auction):</div>
        <div class="target-pill-group">${targetCards}</div>

        <div style="font-size:0.72rem; font-weight:600; color:#94a3b8; margin-top:0.4rem;">🔒 Franchise Trade Targets (Requires Intra-Team Transfer):</div>
        <div class="target-pill-group">${tradeCards}</div>
      </div>
    `;
  }).join("");
}

// Retention Modal
async function openRetentionModal() {
  const modal = document.getElementById("retention-modal");
  const textarea = document.getElementById("retention-json-editor");
  modal.style.display = "flex";

  try {
    const res = await fetch("/api/retention-list");
    const data = await res.json();
    textarea.value = JSON.stringify(data, null, 2);
  } catch (err) {
    textarea.value = "{}";
  }
}

function closeRetentionModal() {
  document.getElementById("retention-modal").style.display = "none";
}

async function saveRetentionChanges() {
  const textarea = document.getElementById("retention-json-editor");
  try {
    const parsed = JSON.parse(textarea.value);
    const res = await fetch("/api/retention-list", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed),
    });
    if (res.ok) {
      closeRetentionModal();
      fetchRankings();
      fetchSquadGaps();
    }
  } catch (err) {
    alert("Invalid JSON format. Please fix and try again.");
  }
}

// 6. METHODOLOGY & VALIDATION
async function fetchValidationSummary() {
  const grid = document.getElementById("methodology-metrics-grid");
  try {
    const res = await fetch("/api/validation");
    const data = await res.json();
    const m = data.metrics;

    grid.innerHTML = `
      <div class="card" style="padding:1.25rem;">
        <h4 style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">Innings 2 Model (GBDT)</h4>
        <div style="font-size:1.8rem; font-weight:900; color:var(--rr-pink); margin:0.3rem 0;">${m.wp_metrics.roc_auc} AUC</div>
        <p style="font-size:0.75rem; color:var(--text-muted);">Log-Loss: ${m.wp_metrics.log_loss} • Brier Score: ${m.wp_metrics.brier_score} (5-Fold Match CV)</p>
      </div>

      <div class="card" style="padding:1.25rem;">
        <h4 style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">Split-Half Stability (Odd vs Even)</h4>
        <div style="font-size:1.8rem; font-weight:900; color:var(--rr-cyan); margin:0.3rem 0;">ρ = ${m.split_half.batting.spearman_rho}</div>
        <p style="font-size:0.75rem; color:var(--text-muted);">Batting Spearman rho (p < 0.01) across independent matches</p>
      </div>

      <div class="card" style="padding:1.25rem;">
        <h4 style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">Temporal Period Stability</h4>
        <div style="font-size:1.8rem; font-weight:900; color:var(--rr-gold); margin:0.3rem 0;">ρ = ${m.temporal.batting_early_vs_late_rho}</div>
        <p style="font-size:0.75rem; color:var(--text-muted);">2022–24 vs 2025–26 multi-season skill persistence</p>
      </div>

      <div class="card" style="padding:1.25rem;">
        <h4 style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">Threshold Sensitivity (20% vs 30%)</h4>
        <div style="font-size:1.8rem; font-weight:900; color:var(--rr-green); margin:0.3rem 0;">${m.sensitivity.batters_overlap_pct_20_vs_30}% Overlap</div>
        <p style="font-size:0.75rem; color:var(--text-muted);">Top 10 rank stability when pressure cutoff shifts</p>
      </div>
    `;
  } catch (err) {
    console.error("Failed to load validation:", err);
  }
}
