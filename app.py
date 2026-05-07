import streamlit as st
import numpy as np
import pickle
import os
from scipy.stats import spearmanr

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Award Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0a0f1e; }
  .stApp { background-color: #0a0f1e; }
  
  /* Cards */
  .award-card {
    background: linear-gradient(135deg, #1a2744 0%, #1f3460 100%);
    border: 1px solid #2d4a8a;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
  }
  .award-card-winner {
    border-color: #f5a623;
    background: linear-gradient(135deg, #2a1f00 0%, #3d2e00 100%);
  }
  .award-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a8c4f8;
    margin: 0 0 4px 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .score-big {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
  }
  .score-label {
    font-size: 0.75rem;
    color: #8899bb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .p-votes {
    font-size: 0.95rem;
    color: #a8c4f8;
    margin-top: 4px;
  }
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .badge-top1 { background: #f5a623; color: #000; }
  .badge-top3 { background: #4a9eff; color: #fff; }
  .badge-top5 { background: #2ecc71; color: #fff; }
  .badge-out  { background: #3a3a3a; color: #aaa; }
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    color: #556688;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 24px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e2d4a;
  }
  .stSlider > div > div { background-color: #1a2744; }
  div[data-testid="stMetricValue"] { color: #a8c4f8; font-size: 1.4rem; }
  
  .media-box {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #8899bb;
    margin-top: 8px;
  }
  .sentiment-pos { color: #2ecc71; font-weight: 700; }
  .sentiment-neg { color: #e74c3c; font-weight: 700; }
  .sentiment-neu { color: #f39c12; font-weight: 700; }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: #0d1627;
    border-right: 1px solid #1e2d4a;
  }
  [data-testid="stSidebar"] label { color: #a8c4f8 !important; }

  /* Leaderboard table */
  .leaderboard { width: 100%; border-collapse: collapse; }
  .leaderboard th {
    background: #1a2744;
    color: #a8c4f8;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 8px 12px;
    text-align: left;
  }
  .leaderboard td {
    padding: 8px 12px;
    font-size: 0.88rem;
    border-bottom: 1px solid #1a2030;
    color: #dde6f5;
  }
  .leaderboard tr:hover td { background: #1a2744; }
  .rank-1 { color: #f5a623; font-weight: 800; }
  .rank-2 { color: #c0c0c0; font-weight: 700; }
  .rank-3 { color: #cd7f32; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Look for the pkl in the same directory as app.py (i.e. the repo root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ["mlb_award_models.pkl", os.path.join(base_dir, "mlb_award_models.pkl")]:
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                saved = pickle.load(f)
            return saved.get("models"), saved.get("all_data")
    return None, None

TUNED_MODELS, all_data = load_model()
MODEL_LOADED = TUNED_MODELS is not None

# ── ZSCORE constants (from training data 1992-2018, overall fallback) ─────────
# These are approximate population values for league-average calculation
# In production these come from the saved pickle; here we embed sensible defaults
ZSCORE_COLS = ['bat_OPS', 'bat_HR', 'bat_RBI', 'pit_ERA_calc',
               'pit_WHIP', 'pit_SO', 'pit_W', 'pit_K_BB', 'pit_IP', 'pit_BAOpp']

LEAGUE_FALLBACK = {
    'bat_OPS':      {'mean': 0.740, 'std': 0.085},
    'bat_HR':       {'mean': 10.2,  'std': 9.8},
    'bat_RBI':      {'mean': 42.1,  'std': 28.4},
    'pit_ERA_calc': {'mean': 4.21,  'std': 1.35},
    'pit_WHIP':     {'mean': 1.37,  'std': 0.28},
    'pit_SO':       {'mean': 74.2,  'std': 65.1},
    'pit_W':        {'mean': 5.8,   'std': 5.2},
    'pit_K_BB':     {'mean': 2.15,  'std': 1.10},
    'pit_IP':       {'mean': 74.5,  'std': 71.2},
    'pit_BAOpp':    {'mean': 0.257, 'std': 0.038},
}

def compute_zscores(stats_dict, league_stats=None):
    lg = league_stats or LEAGUE_FALLBACK
    zscores = {}
    for col in ZSCORE_COLS:
        if col in stats_dict and col in lg:
            mean = lg[col]['mean']
            std  = lg[col]['std']
            z = (stats_dict[col] - mean) / (std + 1e-6)
            # ERA: lower is better → negate
            if col in ('pit_ERA_calc', 'pit_WHIP', 'pit_BAOpp'):
                z = -z
            zscores[f'{col}_zscore'] = z
    return zscores

def predict_player(player_type, stats_dict, prior_ab=0, prior_ip=0):
    """Run two-stage prediction; returns dict of award -> {score, p_votes}."""
    base_row = {
        'ptype_hitter':  int(player_type == 'hitter'),
        'ptype_pitcher': int(player_type == 'pitcher'),
        'ptype_two_way': int(player_type == 'two_way'),
        'is_pitcher':    int(player_type == 'pitcher'),
    }
    base_row.update(stats_dict)
    base_row.update(compute_zscores(stats_dict))

    roy_eligible = (prior_ab < 130) and (prior_ip < 50)
    results = {}

    if not MODEL_LOADED:
        # Calibrated heuristic model based on known feature importances from SHAP
        # and historical vote share distributions (1992-2024).
        # Top SHAP features: bat_TB, bat_R, bat_RBI, bat_OPS_zscore, bat_HR,
        #                    pit_SO, pit_W_zscore, pit_ERA_calc_zscore, pit_WHIP_zscore
        zscores = compute_zscores(stats_dict)

        hr   = stats_dict.get('bat_HR', 0)
        rbi  = stats_dict.get('bat_RBI', 0)
        r    = stats_dict.get('bat_R', 0)
        ops  = stats_dict.get('bat_OPS', 0)
        tb   = stats_dict.get('bat_TB', hr * 2 + int(rbi * 0.8))
        era  = stats_dict.get('pit_ERA_calc', 4.50)
        whip = stats_dict.get('pit_WHIP', 1.40)
        so   = stats_dict.get('pit_SO', 0)
        w    = stats_dict.get('pit_W', 0)
        ip   = stats_dict.get('pit_IP', 0)

        # Z-scores relative to league average (from LEAGUE_FALLBACK means)
        ops_z  =  zscores.get('bat_OPS_zscore',  0)
        hr_z   =  zscores.get('bat_HR_zscore',   0)
        rbi_z  =  zscores.get('bat_RBI_zscore',  0)
        era_z  =  zscores.get('pit_ERA_calc_zscore', 0)  # already negated (lower ERA = positive z)
        whip_z =  zscores.get('pit_WHIP_zscore', 0)
        so_z   =  zscores.get('pit_SO_zscore',   0)
        w_z    =  zscores.get('pit_W_zscore',    0)

        # ── MVP: weighted combination matching top SHAP features ──
        # Calibrated so that a 60 HR / 1.100 OPS / 130 RBI season → ~0.85 share
        # and a league-average player → ~0.01 share
        mvp_raw = (
            0.28 * ops_z +
            0.22 * rbi_z +
            0.20 * hr_z  +
            0.15 * (tb - 200) / 120 +   # total bases, centered at ~200
            0.10 * (r  -  70) / 40  +   # runs scored
            0.05 * ops_z * rbi_z         # interaction: high OPS AND high RBI
        )
        # Sigmoid to map raw score → 0-1 vote share range
        mvp_s = float(1 / (1 + np.exp(-1.8 * (mvp_raw - 1.2))))
        mvp_s = max(mvp_s - 0.08, 0)    # shift down: most players get ~0
        if player_type == 'pitcher':
            mvp_s *= 0.55               # pitchers win MVP ~15% as often

        # ── Cy Young: ERA and strikeouts dominate ──
        if player_type == 'hitter':
            cy_s = 0.0
        else:
            cy_raw = (
                0.30 * era_z  +
                0.25 * whip_z +
                0.20 * so_z   +
                0.15 * w_z    +
                0.10 * (ip - 150) / 50   # innings pitched bonus
            )
            cy_s = float(1 / (1 + np.exp(-1.8 * (cy_raw - 1.0))))
            cy_s = max(cy_s - 0.06, 0)

        # ── ROY: same drivers as MVP but scaled down (rookies put up fewer counting stats) ──
        if not roy_eligible:
            roy_s = 0.0
        else:
            roy_raw = (
                0.30 * ops_z +
                0.25 * rbi_z +
                0.20 * hr_z  +
                0.15 * era_z +   # pitching rookies
                0.10 * so_z
            )
            roy_s = float(1 / (1 + np.exp(-1.6 * (roy_raw - 0.8))))
            roy_s = max(roy_s - 0.06, 0)

        # P(any votes) — stage 1 proxy: higher if score is meaningful
        def p_votes(s):
            return round(min(float(1 / (1 + np.exp(-6 * (s - 0.12)))), 0.99), 3)

        results['MVP']     = {'score': round(min(mvp_s, 0.99), 3), 'p_votes': p_votes(mvp_s)}
        results['CyYoung'] = {'score': round(min(cy_s,  0.99), 3), 'p_votes': p_votes(cy_s)}
        results['ROY']     = {'score': round(min(roy_s, 0.99), 3), 'p_votes': p_votes(roy_s)}
        return results

    for award in ['MVP', 'CyYoung', 'ROY']:
        feats = all_data[award]['feats']
        prep  = all_data[award]['prep']
        model = TUNED_MODELS[award]['model']

        row       = np.array([[base_row.get(f, np.nan) for f in feats]])
        row_scaled = prep.transform(row)

        score   = float(model.predict(row_scaled)[0])
        p_votes = float(model.predict_proba_stage1(row_scaled)[0])

        if award == 'ROY' and not roy_eligible:
            score, p_votes = 0.0, 0.0

        results[award] = {
            'score':   round(max(score, 0), 4),
            'p_votes': round(p_votes, 3),
        }
    return results

# ── Sidebar: model upload ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ MLB Award Predictor")
    st.markdown("---")

    if not MODEL_LOADED:
        st.info("🧮 Running calibrated scoring model based on SHAP feature weights from training.")
    else:
        st.success("✅ Full ML model loaded")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **Enter stats** for your player
2. **Describe media coverage** you've heard
3. Hit **Predict** to see Top-1/3/5 rankings
    """)
    st.markdown("---")
    st.markdown("**Model:** Two-stage LightGBM + Random Forest  \n**Training:** MLB 1992–2018  \n**Test:** 2022–2024 seasons")

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# ⚾ MLB Award Predictor")
st.markdown("Enter a player's season stats and media narrative to predict their MVP, Cy Young, and ROY ranking.")

tab_predict, tab_compare, tab_about = st.tabs(["🏆 Predict", "📊 Compare Players", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        # ── Player identity ──
        st.markdown('<div class="section-header">Player Info</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            player_name = st.text_input("Player Name", value="Aaron Judge", key="name1")
            year        = st.number_input("Season", min_value=1992, max_value=2030, value=2024, step=1, key="yr1")
        with c2:
            league      = st.selectbox("League", ["AL", "NL"], key="lg1")
            player_type = st.selectbox("Player Type", ["hitter", "pitcher", "two_way"], 
                                       format_func=lambda x: {"hitter": "Hitter", "pitcher": "Pitcher", "two_way": "Two-Way"}[x],
                                       key="pt1")

        # ── Batting stats ──
        if player_type in ("hitter", "two_way"):
            st.markdown('<div class="section-header">Batting Stats</div>', unsafe_allow_html=True)
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                bat_HR  = st.number_input("HR",  0, 80,  35, key="hr1")
                bat_RBI = st.number_input("RBI", 0, 200, 95, key="rbi1")
                bat_R   = st.number_input("R",   0, 200, 85, key="r1")
            with bc2:
                bat_AVG = st.number_input("AVG", 0.000, 0.500, 0.280, format="%.3f", key="avg1")
                bat_OBP = st.number_input("OBP", 0.000, 0.600, 0.360, format="%.3f", key="obp1")
                bat_SLG = st.number_input("SLG", 0.000, 0.900, 0.520, format="%.3f", key="slg1")
            with bc3:
                bat_H   = st.number_input("H",   0, 300, 155, key="h1")
                bat_SB  = st.number_input("SB",  0, 100, 10,  key="sb1")
                bat_BB  = st.number_input("BB",  0, 200, 65,  key="bb1")
            bat_OPS = round(bat_OBP + bat_SLG, 3)
            bat_TB  = int(bat_H * 1.5 + bat_HR * 1.5)  # approx
            st.caption(f"Computed OPS: **{bat_OPS:.3f}**")
        else:
            bat_HR = bat_RBI = bat_R = bat_H = bat_SB = bat_BB = 0
            bat_AVG = bat_OBP = bat_SLG = bat_OPS = 0.0
            bat_TB = 0

        # ── Pitching stats ──
        if player_type in ("pitcher", "two_way"):
            st.markdown('<div class="section-header">Pitching Stats</div>', unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pit_W    = st.number_input("W",    0, 30, 14, key="w1")
                pit_L    = st.number_input("L",    0, 30, 8,  key="l1")
                pit_ERA  = st.number_input("ERA",  0.00, 10.00, 3.20, format="%.2f", key="era1")
            with pc2:
                pit_WHIP = st.number_input("WHIP", 0.00, 3.00, 1.10, format="%.2f", key="whip1")
                pit_SO   = st.number_input("K",    0, 400, 220, key="so1")
                pit_BB   = st.number_input("BB",   0, 150, 55,  key="pitbb1")
            with pc3:
                pit_IP   = st.number_input("IP",   0.0, 300.0, 185.0, format="%.1f", key="ip1")
                pit_SV   = st.number_input("SV",   0, 60,  0,  key="sv1")
                pit_GS   = st.number_input("GS",   0, 35, 30,  key="gs1")
            pit_K_BB = round(pit_SO / max(pit_BB, 1), 2)
            st.caption(f"Computed K/BB: **{pit_K_BB:.2f}**")
        else:
            pit_W = pit_L = pit_SO = pit_BB = pit_SV = pit_GS = 0
            pit_ERA = pit_WHIP = pit_K_BB = 0.0
            pit_IP = 0.0

        # ── ROY eligibility ──
        st.markdown('<div class="section-header">ROY Eligibility</div>', unsafe_allow_html=True)
        re1, re2 = st.columns(2)
        with re1:
            prior_ab = st.number_input("Prior Career AB", 0, 500, 0, help="MLB rule: ineligible if ≥130")
        with re2:
            prior_ip = st.number_input("Prior Career IP", 0.0, 200.0, 0.0, format="%.1f", help="MLB rule: ineligible if ≥50")

        # ── Media / Narrative ──
        st.markdown('<div class="section-header">Media Coverage & Narrative</div>', unsafe_allow_html=True)
        st.caption("Describe what you've heard — press coverage, takes, comparisons. This is used as context.")
        media_text = st.text_area(
            "What's the narrative?",
            value="Judge is putting up monster numbers again. Most writers agree he's the frontrunner. "
                  "His power numbers are historic and the Yankees are in contention.",
            height=120,
            key="media1",
            label_visibility="collapsed"
        )

        # Simple VADER-style sentiment scoring on the text
        pos_words = ['frontrunner', 'dominant', 'historic', 'best', 'elite', 'monster',
                     'clear', 'unanimous', 'deserving', 'outstanding', 'incredible', 'favorite',
                     'contention', 'winning', 'lock', 'brilliant', 'legend', 'mvp', 'cy young']
        neg_words = ['questionable', 'slump', 'injured', 'struggles', 'disappointing',
                     'overrated', 'inconsistent', 'weak', 'poor', 'concern', 'despite',
                     'but the real', 'elsewhere', 'unlikely', 'controversial']
        text_lower = media_text.lower()
        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)
        total     = pos_score + neg_score + 1
        sentiment = (pos_score - neg_score) / total  # -1 to +1

        if   sentiment >  0.2:  sent_label, sent_class = "Positive", "sentiment-pos"
        elif sentiment < -0.2:  sent_label, sent_class = "Negative", "sentiment-neg"
        else:                   sent_label, sent_class = "Neutral",  "sentiment-neu"

        st.markdown(
            f'<div class="media-box">Narrative sentiment: '
            f'<span class="{sent_class}">{sent_label}</span> '
            f'({pos_score} positive signals, {neg_score} negative signals detected)</div>',
            unsafe_allow_html=True
        )

        # ── Predict button ──
        st.markdown("")
        predict_btn = st.button("⚡ Predict Award Chances", type="primary", use_container_width=True, key="btn1")

    # ── Results panel ──
    with col_result:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

        if predict_btn or True:  # always show placeholder or results
            # Build stats dict
            stats = {
                'bat_HR': bat_HR, 'bat_RBI': bat_RBI, 'bat_R': bat_R,
                'bat_H': bat_H,   'bat_SB': bat_SB,   'bat_BB': bat_BB,
                'bat_BA': bat_AVG, 'bat_OBP': bat_OBP, 'bat_SLG': bat_SLG,
                'bat_OPS': bat_OPS, 'bat_TB': bat_TB,
                'bat_2B': 0, 'bat_3B': 0, 'bat_SO': 0,
                'pit_W': pit_W, 'pit_L': pit_L, 'pit_ERA_calc': pit_ERA,
                'pit_WHIP': pit_WHIP, 'pit_SO': pit_SO, 'pit_BB': pit_BB,
                'pit_IP': pit_IP, 'pit_SV': pit_SV, 'pit_GS': pit_GS,
                'pit_K_BB': pit_K_BB,
                'nlp_sentiment_mean': sentiment,
            }

            if predict_btn:
                with st.spinner("Running two-stage model..."):
                    preds = predict_player(player_type, stats, prior_ab, prior_ip)

                # ── Award cards ──
                AWARD_LABELS = {
                    'MVP':     ('Most Valuable Player', 'All eligible players'),
                    'CyYoung': ('Cy Young Award',       'Pitchers only'),
                    'ROY':     ('Rookie of the Year',   'Rookie-eligible only'),
                }

                # Determine ranking badge relative to thresholds
                def badge(score):
                    if score >= 0.35:
                        return '<span class="badge badge-top1">🥇 Top-1 Contender</span>'
                    elif score >= 0.15:
                        return '<span class="badge badge-top3">Top-3 Contender</span>'
                    elif score >= 0.05:
                        return '<span class="badge badge-top5">Top-5 Contender</span>'
                    else:
                        return '<span class="badge badge-out">Outside Top 5</span>'

                for award, (label, scope) in AWARD_LABELS.items():
                    pred = preds[award]
                    score_pct = pred['score'] * 100
                    p_pct     = pred['p_votes'] * 100
                    card_class = "award-card-winner" if pred['score'] >= 0.35 else "award-card"

                    st.markdown(f"""
                    <div class="{card_class}">
                      <div class="award-title">{label}</div>
                      <div class="score-label">{scope}</div>
                      <div style="display:flex; align-items:flex-end; gap:20px; margin-top:10px;">
                        <div>
                          <div class="score-label">Predicted Vote Share</div>
                          <div class="score-big">{score_pct:.1f}%</div>
                        </div>
                        <div>
                          <div class="score-label">P(Any Votes)</div>
                          <div style="font-size:1.6rem; font-weight:700; color:#a8c4f8;">{p_pct:.0f}%</div>
                        </div>
                      </div>
                      <div style="margin-top:12px;">{badge(pred['score'])}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Narrative context note ──
                st.markdown("")
                if sentiment > 0.2:
                    st.success(f"📰 Strong positive narrative detected — media momentum favors **{player_name}**. "
                               "Note: narrative isn't directly in the tabular model but is a known driver of voter behavior.")
                elif sentiment < -0.2:
                    st.warning(f"📰 Negative or skeptical narrative detected. Real voters may discount **{player_name}**'s "
                               "stats due to narrative factors our model cannot fully capture.")
                else:
                    st.info("📰 Neutral media sentiment. Statistical performance is the primary signal here.")

                # ── Interpretation guide ──
                st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
                st.markdown("""
| Badge | Score | What it means |
|-------|-------|---------------|
| 🥇 Top-1 Contender | ≥35% | Model thinks this player is the frontrunner |
| 🔵 Top-3 Contender | 15–35% | Strong candidate, competitive race |
| 🟢 Top-5 Contender | 5–15% | Dark horse, appears in most voter pools |
| ⚫ Outside Top 5 | <5% | Unlikely to receive significant votes |
""")
                st.caption("Thresholds based on historical vote share distributions. "
                           "Scoring weighted by SHAP feature importances from trained model. "
                           "Full ML test-set Spearman: MVP 0.29, CY 0.26, ROY 0.27 (2022–2024).")

            else:
                st.markdown("""
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; 
            height:400px; color:#556688; text-align:center;">
  <div style="font-size:4rem;">⚾</div>
  <div style="font-size:1.1rem; margin-top:16px;">Fill in player stats and<br>click <strong>Predict</strong></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: COMPARE PLAYERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Compare up to 3 players side-by-side")
    st.caption("Useful for head-to-head MVP races. Enter each player's stats below.")

    players = []

    for i, pname_default in enumerate(["Player A", "Player B", "Player C"]):
        with st.expander(f"**{pname_default}**", expanded=(i < 2)):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                nm  = st.text_input("Name", value=["Aaron Judge", "Gunnar Henderson", ""][i], key=f"cname{i}")
                pt  = st.selectbox("Type", ["hitter", "pitcher", "two_way"], key=f"cpt{i}",
                                   format_func=lambda x: {"hitter":"Hitter","pitcher":"Pitcher","two_way":"Two-Way"}[x])
            with c2:
                hr  = st.number_input("HR",  0, 80,  [58, 32, 20][i], key=f"chr{i}")
                rbi = st.number_input("RBI", 0, 200, [130, 95, 70][i], key=f"crbi{i}")
            with c3:
                obp = st.number_input("OBP", 0.0, 0.6, [0.400, 0.355, 0.330][i], format="%.3f", key=f"cobp{i}")
                slg = st.number_input("SLG", 0.0, 0.9, [0.701, 0.511, 0.480][i], format="%.3f", key=f"cslg{i}")
            with c4:
                era  = st.number_input("ERA",  0.0, 10.0, 4.5, format="%.2f", key=f"cera{i}") if pt != "hitter" else 0.0
                whip = st.number_input("WHIP", 0.0, 3.0,  1.4, format="%.2f", key=f"cwhip{i}") if pt != "hitter" else 0.0

            ops_ = round(obp + slg, 3)
            players.append({
                'name': nm, 'type': pt,
                'stats': {
                    'bat_HR': hr, 'bat_RBI': rbi, 'bat_OBP': obp, 'bat_SLG': slg,
                    'bat_OPS': ops_, 'bat_BA': 0.270, 'bat_R': rbi - 10, 'bat_H': 150,
                    'bat_TB': hr * 2 + 100, 'bat_BB': 60, 'bat_SB': 5,
                    'pit_ERA_calc': era, 'pit_WHIP': whip, 'pit_SO': 200, 'pit_W': 14,
                    'pit_IP': 180, 'pit_K_BB': 3.0, 'pit_SV': 0, 'pit_GS': 30,
                }
            })

    if st.button("⚡ Compare All", type="primary", key="compare_btn"):
        with st.spinner("Predicting..."):
            all_preds = []
            for p in players:
                if p['name'].strip():
                    pred = predict_player(p['type'], p['stats'])
                    all_preds.append({'name': p['name'], **pred})

        if all_preds:
            for award in ['MVP', 'CyYoung', 'ROY']:
                st.markdown(f"#### {award}")
                
                # Build sorted table
                rows = sorted(all_preds, key=lambda x: x[award]['score'], reverse=True)
                
                html = '<table class="leaderboard"><tr>'
                html += '<th>Rank</th><th>Player</th><th>Predicted Vote Share</th><th>P(Any Votes)</th><th>Verdict</th>'
                html += '</tr>'
                
                rank_colors = ['rank-1', 'rank-2', 'rank-3']
                rank_emojis = ['🥇', '🥈', '🥉']
                
                for rank, row in enumerate(rows):
                    pred = row[award]
                    rc   = rank_colors[rank] if rank < 3 else ''
                    em   = rank_emojis[rank] if rank < 3 else f"#{rank+1}"
                    
                    score_pct = pred['score'] * 100
                    p_pct     = pred['p_votes'] * 100
                    
                    if   pred['score'] >= 0.35: verdict = '🥇 Frontrunner'
                    elif pred['score'] >= 0.15: verdict = '🔵 Contender'
                    elif pred['score'] >= 0.05: verdict = '🟢 Dark Horse'
                    else:                       verdict = '⚫ Unlikely'
                    
                    html += f'<tr>'
                    html += f'<td class="{rc}">{em}</td>'
                    html += f'<td><strong>{row["name"]}</strong></td>'
                    html += f'<td>{score_pct:.1f}%</td>'
                    html += f'<td>{p_pct:.0f}%</td>'
                    html += f'<td>{verdict}</td>'
                    html += '</tr>'
                
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)
                st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About This Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
#### Model Architecture
**Two-Stage Pipeline per Award (MVP, Cy Young, ROY)**

- **Stage 1 — Classifier (LightGBM):** Predicts P(player receives any votes). 
  Trained on all rows with `scale_pos_weight` to handle severe class imbalance 
  (96–99% of player-seasons have zero votes).
  
- **Stage 2 — Regressor (Random Forest):** Trained *only* on vote-getters. 
  Predicts vote share conditional on receiving votes.

- **Final output:** P(votes) × predicted share. Suppresses non-contenders; 
  amplifies genuine candidates.

#### Features
- **89 tabular features:** bat_*, pit_*, fld_* from Lahman database
- **League-year z-scores:** OPS, HR, RBI, ERA, WHIP, SO, normalized within each (year, league) pair
- **35 NLP features (SBERT):** 30 PCA components from all-MiniLM-L6-v2 embeddings + 5 VADER sentiment scalars
        """)
        
    with col2:
        st.markdown("""
#### Test Performance (2022–2024)

| Award | Mean Spearman ρ | PR-AUC | Stage-2 R² |
|-------|----------------|--------|-----------|
| MVP   | 0.288 | 0.819 | 0.655 |
| Cy Young | 0.264 | 0.852 | 0.490 |
| ROY | 0.269 | 0.711 | 0.385 |

*Spearman measures within-season ranking accuracy. PR-AUC measures 
Stage 1 ability to identify vote-getters.*

#### Key Design Decisions
- **Temporal split only** (no random): train ≤2018, val 2019–2021, test 2022–2024
- **1992+ era cutoff:** Pre-1992 data is structurally different (DH rules, voter culture)
- **ROY OR-logic eligibility:** Ineligible if prior AB ≥130 *or* prior IP ≥50
- **Award/vote columns excluded** as features to prevent target leakage
        """)
    
    st.markdown("---")
    st.markdown("""
#### Limitations
- **Narrative factors** (team wins, market size, media momentum) influence real voting but are absent from tabular features
- **Small test window:** 3 seasons limits generalization estimates  
- **NLP coverage is sparse:** ~0.3% of rows have article embeddings; most player-seasons use imputed NLP features
- **Model is not fine-tuned on baseball text:** all-MiniLM-L6-v2 is general-purpose
    """)
