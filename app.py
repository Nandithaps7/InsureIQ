import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ · Smart Insurance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

:root {
    --ink: #0a0e1a;
    --ink-2: #1c2233;
    --ink-3: #2d3650;
    --mist: #8892a4;
    --mist-2: #b8c0cc;
    --fog: #e8ecf2;
    --frost: #f4f6fa;
    --white: #ffffff;
    --teal: #00c9b1;
    --teal-d: #00a896;
    --teal-l: #80e4d8;
    --teal-ghost: rgba(0,201,177,0.10);
    --amber: #f59e0b;
    --amber-ghost: rgba(245,158,11,0.10);
    --rose: #f43f5e;
    --rose-ghost: rgba(244,63,94,0.10);
    --violet: #8b5cf6;
    --violet-ghost: rgba(139,92,246,0.10);
    --sky: #0ea5e9;
    --sky-ghost: rgba(14,165,233,0.10);
    --emerald: #10b981;
    --emerald-ghost: rgba(16,185,129,0.10);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--frost);
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: 1px solid var(--ink-3) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] * { color: var(--mist-2) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #111 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button { color: var(--teal) !important; }
[data-testid="stSidebar"] label { color: var(--mist) !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; font-weight: 600 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: var(--ink-2) !important;
    color: var(--white) !important;
    border: 1px solid var(--ink-3) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    width: 100% !important;
    padding: 10px 0 !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
    color: var(--ink) !important;
}

/* ─── MAIN LAYOUT ─── */
.block-container { padding: 0 2rem 2rem !important; max-width: 100% !important; }

/* ─── TOP NAV ─── */
.top-nav {
    background: var(--white);
    border-bottom: 1px solid var(--fog);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 1px 0 var(--fog);
}
.top-nav-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--ink);
    padding: 1rem 0;
    letter-spacing: -0.03em;
}
.top-nav-brand .accent { color: var(--teal); }
.top-nav-tagline {
    font-size: 0.72rem;
    color: var(--mist);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ─── NAV PILLS ─── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent !important;
    color: var(--mist) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 7px 18px !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: var(--frost) !important;
    color: var(--ink) !important;
}

/* ─── PAGE HERO ─── */
.page-hero {
    background: var(--ink);
    border-radius: 16px;
    padding: 48px 52px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,201,177,0.15) 0%, transparent 70%);
}
.page-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%);
}
.page-hero .tag {
    display: inline-block;
    background: var(--teal-ghost);
    border: 1px solid rgba(0,201,177,0.3);
    color: var(--teal);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 30px;
    margin-bottom: 16px;
}
.page-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--white);
    margin: 0 0 10px;
    letter-spacing: -0.04em;
    line-height: 1.1;
}
.page-hero p {
    font-size: 0.95rem;
    color: var(--mist);
    margin: 0;
    max-width: 500px;
    line-height: 1.7;
}

/* ─── SECTION HEADER ─── */
.sec-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 4px;
}
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 20px;
    letter-spacing: -0.025em;
}

/* ─── COMPANY COMPARISON TABLE ─── */
.company-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.company-card {
    background: var(--white);
    border: 2px solid var(--fog);
    border-radius: 14px;
    padding: 22px 20px;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.company-card.recommended {
    border-color: var(--teal);
    box-shadow: 0 0 0 4px var(--teal-ghost);
}
.company-card.recommended::after {
    content: 'BEST PICK';
    position: absolute;
    top: 12px; right: -22px;
    background: var(--teal);
    color: var(--ink);
    font-size: 0.58rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    padding: 4px 28px;
    transform: rotate(45deg);
}
.company-card:hover:not(.recommended) { border-color: var(--mist-2); box-shadow: 0 4px 16px rgba(0,0,0,0.07); }
.company-logo-area {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.company-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 800;
    flex-shrink: 0;
}
.company-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.2;
}
.company-type { font-size: 0.7rem; color: var(--mist); }
.company-score-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.company-score-label { font-size: 0.72rem; color: var(--mist); font-weight: 500; }
.company-score-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--ink);
}
.score-bar-bg {
    height: 4px;
    background: var(--fog);
    border-radius: 4px;
    margin-bottom: 10px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.company-premium {
    background: var(--frost);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.premium-label { font-size: 0.7rem; color: var(--mist); }
.premium-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--teal-d);
}
.pro-con {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--fog);
}
.pro-con-item { font-size: 0.74rem; color: var(--mist); line-height: 1.8; }
.pro-con-item .green { color: var(--emerald); font-weight: 600; }
.pro-con-item .red { color: var(--rose); font-weight: 600; }

/* ─── FILTER BAR ─── */
.filter-bar {
    background: var(--white);
    border: 1px solid var(--fog);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}
.filter-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--frost);
    border: 1px solid var(--fog);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--ink-3);
    cursor: pointer;
    transition: all 0.15s;
}
.filter-chip.active { background: var(--teal); border-color: var(--teal); color: var(--ink); }
.filter-chip:hover:not(.active) { border-color: var(--mist-2); }

/* ─── METRIC STRIP ─── */
.metric-strip { display: flex; gap: 14px; margin-bottom: 24px; }
.mstrip-box {
    flex: 1;
    background: var(--white);
    border: 1px solid var(--fog);
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
}
.mstrip-box .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-bottom: 12px;
}
.mstrip-box .mlabel {
    font-size: 0.67rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--mist);
    margin-bottom: 6px;
    font-weight: 600;
}
.mstrip-box .mvalue {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.025em;
}

/* ─── COMPARISON TABLE ─── */
.cmp-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--white);
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--fog);
    margin-bottom: 24px;
    font-size: 0.83rem;
}
.cmp-table th {
    background: var(--ink);
    color: var(--mist);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 13px 16px;
    font-weight: 600;
    text-align: left;
}
.cmp-table td {
    padding: 13px 16px;
    border-bottom: 1px solid var(--fog);
    color: var(--ink-3);
    vertical-align: middle;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr:hover td { background: var(--frost); }
.cmp-table td.company-cell { font-weight: 600; color: var(--ink); }
.cmp-table td.best-cell { color: var(--teal-d); font-weight: 700; }
.badge-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-green { background: var(--emerald-ghost); color: var(--emerald); }
.badge-amber { background: var(--amber-ghost); color: var(--amber); }
.badge-rose  { background: var(--rose-ghost);  color: var(--rose);  }

/* ─── PREDICT FORM ─── */
.form-card {
    background: var(--white);
    border: 1px solid var(--fog);
    border-radius: 16px;
    padding: 28px 30px;
    margin-bottom: 20px;
}
.form-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--fog);
    letter-spacing: -0.01em;
}

/* ─── RESULT CARD ─── */
.result-card {
    background: var(--ink);
    border-radius: 16px;
    padding: 36px 40px;
    text-align: center;
    margin-top: 24px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,201,177,0.15) 0%, transparent 70%);
}
.result-card .rlabel {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--mist);
    margin-bottom: 10px;
    font-weight: 700;
}
.result-card .ramount {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--teal);
    letter-spacing: -0.04em;
}
.result-card .rsub { font-size: 0.80rem; color: var(--mist); margin-top: 8px; }
.result-card .company-rec {
    margin-top: 20px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--ink-2);
    border: 1px solid var(--ink-3);
    border-radius: 30px;
    padding: 8px 18px;
    font-size: 0.78rem;
    color: var(--mist-2);
}
.result-card .company-rec strong { color: var(--teal); }

/* ─── CARD ─── */
.card {
    background: var(--white);
    border-radius: 14px;
    padding: 24px 26px;
    border: 1px solid var(--fog);
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
}
.card-body { font-size: 0.84rem; color: var(--mist); line-height: 1.75; }

/* ─── FORM INPUTS ─── */
label { color: var(--ink-3) !important; font-weight: 600 !important; font-size: 0.82rem !important; }
div[data-testid="stNumberInput"] > div,
div[data-testid="stSelectbox"] > div {
    background-color: var(--frost) !important;
    border: 1.5px solid var(--fog) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input {
    background-color: transparent !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--frost) !important;
    color: var(--ink) !important;
    border: none !important;
}
ul[role="listbox"] {
    background-color: var(--white) !important;
    border: 1px solid var(--fog) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important;
}
ul[role="listbox"] li { color: var(--ink-3) !important; }
ul[role="listbox"] li:hover { background-color: var(--frost) !important; }

/* ─── PREDICT BUTTON ─── */
div[data-testid="stMainBlockContainer"] .stButton > button {
    background: var(--teal) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 12px 36px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 16px rgba(0,201,177,0.28) !important;
    transition: all 0.2s !important;
}
div[data-testid="stMainBlockContainer"] .stButton > button:hover {
    background: var(--teal-d) !important;
    box-shadow: 0 6px 20px rgba(0,201,177,0.38) !important;
    transform: translateY(-1px) !important;
}

/* ─── MISC ─── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--fog) !important; }
.stInfo, [data-testid="stAlert"] { background: var(--frost) !important; border: 1px solid var(--fog) !important; border-radius: 10px !important; color: var(--ink-3) !important; }
[data-testid="stMetric"] { background: var(--white); border: 1px solid var(--fog); border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricLabel"] { color: var(--mist) !important; font-weight: 600 !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; font-family: 'Syne', sans-serif !important; }
.divider { height: 1px; background: var(--fog); margin: 28px 0; }
.footer {
    text-align: center; color: var(--mist); font-size: 0.75rem;
    padding: 24px 0 12px; border-top: 1px solid var(--fog); margin-top: 40px;
}
.footer strong { color: var(--teal); }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--frost); }
::-webkit-scrollbar-thumb { background: var(--fog); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Insurance Company Data ─────────────────────────────────────────────────────
INSURANCE_DATA = {
    "health": {
        "title": "Health Insurance",
        "icon": "❤️‍🩹",
        "color": "#00c9b1",
        "gradient": "linear-gradient(135deg, #00c9b1, #10b981)",
        "companies": [
            {
                "name": "Star Health", "short": "STAR", "icon": "⭐", "icon_bg": "#fff3cd", "icon_color": "#92400e",
                "claim_ratio": 95.2, "cashless": 14000, "premium_range": "₹4,500 – ₹18,000",
                "network_score": 92, "settlement_speed": "3–5 days", "recommended": True,
                "rating": 4.5, "coverage_limit": "₹5L – ₹2Cr",
                "pros": ["Widest cashless network", "High claim ratio", "No co-payment"],
                "cons": ["Higher premium for seniors"],
                "badge": "Most Trusted", "badge_cls": "badge-green",
                "plan": "Family Floater Pro",
            },
            {
                "name": "HDFC ERGO", "short": "HDFC", "icon": "🏦", "icon_bg": "#dbeeff", "icon_color": "#1262a8",
                "claim_ratio": 91.4, "cashless": 13000, "premium_range": "₹3,800 – ₹16,500",
                "network_score": 88, "settlement_speed": "4–6 days", "recommended": False,
                "rating": 4.2, "coverage_limit": "₹3L – ₹1.5Cr",
                "pros": ["Affordable premiums", "Digital-first claims", "Quick renewal"],
                "cons": ["Slower claims vs Star"],
                "badge": "Best Value", "badge_cls": "badge-amber",
                "plan": "Optima Restore",
            },
            {
                "name": "Niva Bupa", "short": "NIVA", "icon": "🌿", "icon_bg": "#d1fae5", "icon_color": "#065f46",
                "claim_ratio": 88.6, "cashless": 10000, "premium_range": "₹5,200 – ₹22,000",
                "network_score": 84, "settlement_speed": "5–7 days", "recommended": False,
                "rating": 4.0, "coverage_limit": "₹5L – ₹3Cr",
                "pros": ["High coverage limits", "Maternity benefits", "Wellness perks"],
                "cons": ["Smaller network", "Higher premium"],
                "badge": "Premium", "badge_cls": "badge-rose",
                "plan": "ReAssure 2.0",
            },
            {
                "name": "Care Health", "short": "CARE", "icon": "💙", "icon_bg": "#ede9fe", "icon_color": "#5b21b6",
                "claim_ratio": 89.1, "cashless": 11500, "premium_range": "₹4,100 – ₹17,000",
                "network_score": 86, "settlement_speed": "4–5 days", "recommended": False,
                "rating": 4.1, "coverage_limit": "₹3L – ₹2Cr",
                "pros": ["Good OPD coverage", "Preventive checkups", "No room rent limit"],
                "cons": ["Limited riders available"],
                "badge": "Good Choice", "badge_cls": "badge-green",
                "plan": "Care Advantage",
            },
            {
                "name": "Max Bupa", "short": "MAX", "icon": "🔵", "icon_bg": "#fef3c7", "icon_color": "#92400e",
                "claim_ratio": 86.3, "cashless": 9500, "premium_range": "₹3,500 – ₹14,500",
                "network_score": 80, "settlement_speed": "6–8 days", "recommended": False,
                "rating": 3.8, "coverage_limit": "₹3L – ₹1Cr",
                "pros": ["Lowest entry premium", "Easy online process", "Quick issuance"],
                "cons": ["Lower claim ratio", "Smaller cashless network"],
                "badge": "Budget", "badge_cls": "badge-amber",
                "plan": "GoActive",
            },
            {
                "name": "Aditya Birla", "short": "ABHI", "icon": "🌟", "icon_bg": "#fce7f3", "icon_color": "#9d174d",
                "claim_ratio": 90.8, "cashless": 12000, "premium_range": "₹4,800 – ₹20,000",
                "network_score": 87, "settlement_speed": "4–6 days", "recommended": False,
                "rating": 4.3, "coverage_limit": "₹5L – ₹2Cr",
                "pros": ["Activ Health rewards", "OPD & wellness included", "Strong digital app"],
                "cons": ["Premium on higher side"],
                "badge": "Innovative", "badge_cls": "badge-green",
                "plan": "Activ Health Platinum",
            },
        ],
        "comparison_headers": ["Company", "Claim Ratio", "Cashless Hospitals", "Avg Premium", "Rating", "Best For"],
    },
    "car": {
        "title": "Car Insurance",
        "icon": "🚗",
        "color": "#0ea5e9",
        "gradient": "linear-gradient(135deg, #0ea5e9, #8b5cf6)",
        "companies": [
            {
                "name": "ICICI Lombard", "short": "ICICI", "icon": "🔶", "icon_bg": "#dbeeff", "icon_color": "#1262a8",
                "claim_ratio": 87.5, "cashless": 16000, "premium_range": "₹3,200 – ₹28,000",
                "network_score": 94, "settlement_speed": "1–3 days", "recommended": True,
                "rating": 4.6, "coverage_limit": "IDV Based",
                "pros": ["Largest garage network", "Fastest claim settlement", "AI-based claim processing"],
                "cons": ["Slightly higher premium"],
                "badge": "Top Rated", "badge_cls": "badge-green",
                "plan": "Comprehensive+ Zero Dep",
            },
            {
                "name": "Bajaj Allianz", "short": "BAJAJ", "icon": "🏎️", "icon_bg": "#fef3c7", "icon_color": "#92400e",
                "claim_ratio": 85.2, "cashless": 14500, "premium_range": "₹2,800 – ₹25,000",
                "network_score": 89, "settlement_speed": "2–4 days", "recommended": False,
                "rating": 4.3, "coverage_limit": "IDV Based",
                "pros": ["Competitive pricing", "On-the-spot claims", "24x7 towing"],
                "cons": ["Slower than ICICI"],
                "badge": "Best Value", "badge_cls": "badge-amber",
                "plan": "Comprehensive Own Damage",
            },
            {
                "name": "HDFC ERGO", "short": "HDFC", "icon": "🏦", "icon_bg": "#dbeeff", "icon_color": "#1262a8",
                "claim_ratio": 83.9, "cashless": 13000, "premium_range": "₹2,600 – ₹22,000",
                "network_score": 85, "settlement_speed": "3–5 days", "recommended": False,
                "rating": 4.1, "coverage_limit": "IDV Based",
                "pros": ["Affordable premium", "Smooth digital process", "Multi-car discount"],
                "cons": ["Limited garage network in Tier 3"],
                "badge": "Affordable", "badge_cls": "badge-amber",
                "plan": "Motor Comprehensive",
            },
            {
                "name": "Tata AIG", "short": "TATA", "icon": "🔷", "icon_bg": "#ede9fe", "icon_color": "#5b21b6",
                "claim_ratio": 86.1, "cashless": 12500, "premium_range": "₹2,900 – ₹24,000",
                "network_score": 83, "settlement_speed": "2–4 days", "recommended": False,
                "rating": 4.2, "coverage_limit": "IDV Based",
                "pros": ["Quick digital claims", "Wide add-on options", "Good NCB transfer"],
                "cons": ["Network smaller vs ICICI"],
                "badge": "Reliable", "badge_cls": "badge-green",
                "plan": "Auto Secure",
            },
            {
                "name": "New India Assurance", "short": "NIA", "icon": "🏛️", "icon_bg": "#d1fae5", "icon_color": "#065f46",
                "claim_ratio": 82.4, "cashless": 10000, "premium_range": "₹2,400 – ₹20,000",
                "network_score": 78, "settlement_speed": "5–8 days", "recommended": False,
                "rating": 3.7, "coverage_limit": "IDV Based",
                "pros": ["Govt. backed trust", "Lowest premium", "Pan-India presence"],
                "cons": ["Slow claim process", "Outdated systems"],
                "badge": "Budget", "badge_cls": "badge-amber",
                "plan": "Package Policy",
            },
            {
                "name": "Acko", "short": "ACKO", "icon": "⚡", "icon_bg": "#fce7f3", "icon_color": "#9d174d",
                "claim_ratio": 84.8, "cashless": 8500, "premium_range": "₹2,200 – ₹18,000",
                "network_score": 75, "settlement_speed": "1–2 days", "recommended": False,
                "rating": 4.0, "coverage_limit": "IDV Based",
                "pros": ["Lowest premium online", "Zero paperwork", "Instant claim approval"],
                "cons": ["Smaller garage network", "Primarily digital only"],
                "badge": "Digital-First", "badge_cls": "badge-rose",
                "plan": "Comprehensive Digital",
            },
        ],
        "comparison_headers": ["Company", "Claim Ratio", "Cashless Garages", "Avg Premium", "Rating", "Best For"],
    },
    "life": {
        "title": "Life Insurance",
        "icon": "🌱",
        "color": "#f59e0b",
        "gradient": "linear-gradient(135deg, #f59e0b, #f43f5e)",
        "companies": [
            {
                "name": "LIC of India", "short": "LIC", "icon": "🏛️", "icon_bg": "#fef3c7", "icon_color": "#92400e",
                "claim_ratio": 98.7, "cashless": 0, "premium_range": "₹8,000 – ₹60,000/yr",
                "network_score": 98, "settlement_speed": "30 days avg", "recommended": True,
                "rating": 4.6, "coverage_limit": "₹10L – No Limit",
                "pros": ["Highest claim settlement", "Government backed", "Largest agent network"],
                "cons": ["Returns lower vs private", "Slower digital process"],
                "badge": "Most Trusted", "badge_cls": "badge-green",
                "plan": "Tech Term",
            },
            {
                "name": "HDFC Life", "short": "HDFC", "icon": "🏦", "icon_bg": "#dbeeff", "icon_color": "#1262a8",
                "claim_ratio": 99.4, "cashless": 0, "premium_range": "₹7,500 – ₹55,000/yr",
                "network_score": 95, "settlement_speed": "7–15 days", "recommended": False,
                "rating": 4.7, "coverage_limit": "₹10L – No Limit",
                "pros": ["Highest claim ratio pvt", "Fastest settlement", "Strong ULIP options"],
                "cons": ["Higher premium vs LIC term"],
                "badge": "Best Private", "badge_cls": "badge-green",
                "plan": "Click2Protect Life",
            },
            {
                "name": "SBI Life", "short": "SBI", "icon": "🔵", "icon_bg": "#d1fae5", "icon_color": "#065f46",
                "claim_ratio": 95.3, "cashless": 0, "premium_range": "₹7,000 – ₹50,000/yr",
                "network_score": 92, "settlement_speed": "15–20 days", "recommended": False,
                "rating": 4.3, "coverage_limit": "₹10L – No Limit",
                "pros": ["Bank-backed trust", "Wide distribution", "Good rider options"],
                "cons": ["Average digital experience"],
                "badge": "Bank-Backed", "badge_cls": "badge-amber",
                "plan": "eShield Next",
            },
            {
                "name": "Max Life", "short": "MAX", "icon": "💼", "icon_bg": "#ede9fe", "icon_color": "#5b21b6",
                "claim_ratio": 99.5, "cashless": 0, "premium_range": "₹7,200 – ₹52,000/yr",
                "network_score": 90, "settlement_speed": "7–12 days", "recommended": False,
                "rating": 4.5, "coverage_limit": "₹10L – No Limit",
                "pros": ["Highest claim ratio", "Best customer service", "Excellent ULIP returns"],
                "cons": ["Limited offline branches"],
                "badge": "Top Claims", "badge_cls": "badge-green",
                "plan": "Smart Term Plan+",
            },
            {
                "name": "ICICI Prudential", "short": "IPRU", "icon": "🔶", "icon_bg": "#fce7f3", "icon_color": "#9d174d",
                "claim_ratio": 97.9, "cashless": 0, "premium_range": "₹6,800 – ₹48,000/yr",
                "network_score": 88, "settlement_speed": "10–14 days", "recommended": False,
                "rating": 4.3, "coverage_limit": "₹10L – No Limit",
                "pros": ["Good ULIP portfolio", "Competitive premium", "Strong digital app"],
                "cons": ["Slightly lower claim ratio"],
                "badge": "ULIP Leader", "badge_cls": "badge-amber",
                "plan": "iProtect Smart",
            },
            {
                "name": "Tata AIA", "short": "TATA", "icon": "🌟", "icon_bg": "#d1fae5", "icon_color": "#065f46",
                "claim_ratio": 99.1, "cashless": 0, "premium_range": "₹7,800 – ₹58,000/yr",
                "network_score": 86, "settlement_speed": "7–10 days", "recommended": False,
                "rating": 4.4, "coverage_limit": "₹10L – No Limit",
                "pros": ["Super-fast claims", "Joint life cover", "No medicals up to 75L"],
                "cons": ["Premium slightly higher"],
                "badge": "Fast Claims", "badge_cls": "badge-green",
                "plan": "Maha Raksha Supreme",
            },
        ],
        "comparison_headers": ["Company", "Claim Ratio", "Coverage Options", "Annual Premium", "Rating", "Best For"],
    },
}

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_COL    = "insurance_cost_inr"
EXPECTED_COLS = [
    "age", "bmi", "sex", "children", "smoker", "state",
    "annual_income_lpa", "family_income_lpa",
    "num_insurance_policies", "num_vehicles", "insurance_cost_inr",
]
NUMERIC_COLS     = ["age", "bmi", "children", "annual_income_lpa", "family_income_lpa", "num_insurance_policies", "num_vehicles"]
CATEGORICAL_COLS = ["sex", "smoker", "state"]


def load_dataset(raw_df):
    df = raw_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "annual_income": "annual_income_lpa",
        "family_income": "family_income_lpa",
        "num_insura": "num_insurance_policies",
        "num_insur": "num_insurance_policies",
        "num_vehicle": "num_vehicles",
        "insurance_cost": "insurance_cost_inr",
        "insurance_cost_in": "insurance_cost_inr",
    }
    df = df.rename(columns=rename_map)
    for col in EXPECTED_COLS:
        if col not in df.columns:
            if col == TARGET_COL:
                continue
            df[col] = np.nan
    for col in NUMERIC_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


def add_features(df):
    data = df.copy()
    data["age_bmi"]          = data["age"] * data["bmi"]
    data["income_per_child"] = data["family_income_lpa"] / (data["children"] + 1)
    data["coverage_load"]    = data["num_insurance_policies"] + data["num_vehicles"]
    data["is_smoker"]        = (data["smoker"].astype(str).str.lower() == "yes").astype(int)
    return data


FEATURE_COLS = [
    "age", "bmi", "children", "annual_income_lpa", "family_income_lpa",
    "num_insurance_policies", "num_vehicles",
    "sex", "smoker", "state",
    "age_bmi", "income_per_child", "coverage_load", "is_smoker",
]


def prepare_input_frame(df):
    data = df.copy()
    for col in NUMERIC_COLS:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        if col not in data.columns:
            data[col] = "unknown"
        data[col] = data[col].astype(str).str.strip().str.lower()
    data = add_features(data)
    for col in FEATURE_COLS:
        if col not in data.columns:
            data[col] = np.nan
    return data[FEATURE_COLS]


class InsurancePredictor:
    def __init__(self):
        numeric_features     = ["age", "bmi", "children", "annual_income_lpa", "family_income_lpa",
                                "num_insurance_policies", "num_vehicles",
                                "age_bmi", "income_per_child", "coverage_load", "is_smoker"]
        categorical_features = ["sex", "smoker", "state"]

        numeric_pipe     = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.preprocessor = ColumnTransformer([
            ("num", numeric_pipe,     numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ], remainder="drop")

        self.regressor = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.0, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
            objective="reg:squarederror", verbosity=0,
        )
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model",        self.regressor),
        ])
        self.r2         = None
        self.mae        = None
        self.is_trained = False

    def train(self, df):
        data = df.copy()
        X    = prepare_input_frame(data)
        y    = pd.to_numeric(data[TARGET_COL], errors="coerce")
        mask = y.notna()
        X    = X.loc[mask].reset_index(drop=True)
        y    = y.loc[mask].reset_index(drop=True)
        if len(X) < 10:
            raise ValueError("Not enough rows.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, np.log1p(y_train))
        preds    = np.expm1(self.pipeline.predict(X_test))
        self.mae = mean_absolute_error(y_test, preds)
        self.r2  = r2_score(y_test, preds)
        self.is_trained = True

    def predict_one(self, input_dict):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        row = pd.DataFrame([input_dict])
        row = prepare_input_frame(row)
        return max(0.0, float(np.expm1(self.pipeline.predict(row)[0])))


# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "data":                   None,
    "trained":                False,
    "predictor":              InsurancePredictor(),
    "prediction_result":      None,
    "prediction_error":       None,
    "active_page":            "Explore",
    "uploaded_file_id":       None,
    "selected_insurance_type": None,
    "selected_company":       None,
    "explore_sort":           "rating",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 24px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#fff;letter-spacing:-0.03em;'>
            🛡️ InsureIQ
        </div>
        <div style='font-size:0.68rem;color:rgba(255,255,255,0.35);margin-top:3px;letter-spacing:0.12em;text-transform:uppercase;'>
            Smart Insurance Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;"
        "color:rgba(255,255,255,0.35);margin-bottom:8px;'>Upload Dataset</div>",
        unsafe_allow_html=True,
    )
    file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if file is not None:
        file_id = (file.name, file.size)
        if st.session_state.uploaded_file_id != file_id:
            try:
                raw       = pd.read_csv(file)
                df_loaded = load_dataset(raw)
                st.session_state.data             = df_loaded
                st.session_state.trained          = False
                st.session_state.predictor        = InsurancePredictor()
                st.session_state.prediction_result = None
                st.session_state.prediction_error  = None
                st.session_state.uploaded_file_id  = file_id
                st.success(f"✅ Loaded {len(df_loaded):,} rows")
            except Exception as e:
                st.error("Failed to load dataset")
                st.text(str(e))

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if st.session_state.data is not None and not st.session_state.trained:
        if st.button("⚡  Train Model", use_container_width=True):
            with st.spinner("Training XGBoost…"):
                try:
                    st.session_state.predictor.train(st.session_state.data)
                    st.session_state.trained          = True
                    st.session_state.prediction_result = None
                    st.success("✅ Training complete!")
                except Exception as e:
                    st.error("Training failed")
                    st.text(str(e))

    if st.session_state.trained:
        p = st.session_state.predictor
        st.markdown(f"""
        <div style='background:rgba(0,201,177,0.08);border:1px solid rgba(0,201,177,0.20);
                    border-radius:10px;padding:14px 16px;font-size:0.80rem;
                    color:rgba(255,255,255,0.65);line-height:2.1;margin-bottom:12px;'>
            ✅&nbsp; XGBoost trained<br>
            R² &nbsp;·&nbsp; <strong style='color:#00c9b1'>{p.r2:.4f}</strong><br>
            MAE &nbsp;·&nbsp; <strong style='color:#00c9b1'>₹{p.mae:,.0f}</strong>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("↺  Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.68rem;color:rgba(255,255,255,0.22);letter-spacing:0.06em;line-height:2;'>
        NAVIGATION<br>
    </div>""", unsafe_allow_html=True)

    pages = [
        ("🧭  Explore Plans", "Explore"),
        ("🔮  Predict Cost",  "Predict"),
        ("📊  Dataset",       "Dataset"),
        ("ℹ️  About",         "About"),
    ]
    for label, page in pages:
        if st.button(label, key=f"nav_{page}", use_container_width=True):
            st.session_state.active_page = page
            st.rerun()

# ── Top Bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <div class="top-nav-brand">Insure<span class="accent">IQ</span></div>
    <div class="top-nav-tagline">Smart Insurance · India's Best Plans · AI-Powered</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: EXPLORE PLANS
# ══════════════════════════════════════════════════════════════════════
if st.session_state.active_page == "Explore":

    st.markdown("""
    <div class="page-hero">
        <div class="tag">Insurance Explorer · 2026</div>
        <h1>Find Your Perfect Plan</h1>
        <p>Compare India's top insurance providers across Health, Car, and Life categories.
           Data-driven ratings to help you choose with confidence.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Choose Insurance Type ──────────────────────────────────
    st.markdown('<div class="sec-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Choose Insurance Category</div>', unsafe_allow_html=True)

    col_h, col_c, col_l = st.columns(3)
    with col_h:
        if st.button("❤️‍🩹  Health Insurance\nCovers medical expenses & hospitalisation",
                     key="btn_health", use_container_width=True):
            st.session_state.selected_insurance_type = "health"
            st.session_state.selected_company = None
            st.rerun()
    with col_c:
        if st.button("🚗  Car Insurance\nProtects your vehicle against damage & theft",
                     key="btn_car", use_container_width=True):
            st.session_state.selected_insurance_type = "car"
            st.session_state.selected_company = None
            st.rerun()
    with col_l:
        if st.button("🌱  Life Insurance\nSecures your family's financial future",
                     key="btn_life", use_container_width=True):
            st.session_state.selected_insurance_type = "life"
            st.session_state.selected_company = None
            st.rerun()

    # Visual active-type indicator
    type_icons = {
        "health": ("❤️‍🩹", "Health Insurance", "#00c9b1"),
        "car":    ("🚗",    "Car Insurance",    "#0ea5e9"),
        "life":   ("🌱",    "Life Insurance",   "#f59e0b"),
    }
    if st.session_state.selected_insurance_type:
        ic, nm, cl = type_icons[st.session_state.selected_insurance_type]
        st.markdown(f"""
        <div style='background:var(--white);border:2px solid {cl};border-radius:12px;
                    padding:14px 20px;margin:16px 0 24px;
                    display:flex;align-items:center;gap:12px;'>
            <span style='font-size:1.5rem;'>{ic}</span>
            <div>
                <div style='font-family:Syne,sans-serif;font-weight:700;font-size:0.92rem;
                            color:var(--ink);'>{nm} Selected</div>
                <div style='font-size:0.76rem;color:var(--mist);'>
                    Showing top-rated companies for {nm.lower()}
                </div>
            </div>
            <div style='margin-left:auto;background:{cl}22;border:1px solid {cl}44;
                        border-radius:20px;padding:4px 12px;font-size:0.70rem;
                        font-weight:700;color:{cl};'>ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Companies section (only when a type is selected) ───────────────
    if st.session_state.selected_insurance_type:
        ins_type = st.session_state.selected_insurance_type
        ins_data = INSURANCE_DATA[ins_type]
        companies = ins_data["companies"]

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Step 2: Sort & Filter
        st.markdown('<div class="sec-label">Step 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Compare & Select Provider</div>', unsafe_allow_html=True)

        sort_col, _, info_col = st.columns([2, 3, 2])
        with sort_col:
            sort_by = st.selectbox(
                "Sort by",
                ["Rating ↓", "Claim Ratio ↓", "Premium ↑", "Network Score ↓"],
                label_visibility="collapsed",
            )
        with info_col:
            st.markdown(
                f"<div style='font-size:0.78rem;color:var(--mist);padding-top:8px;'>"
                f"Showing <strong style='color:var(--ink);'>{len(companies)} providers</strong>"
                f" for {ins_data['title']}</div>",
                unsafe_allow_html=True,
            )

        if sort_by == "Rating ↓":
            companies_sorted = sorted(companies, key=lambda x: x["rating"], reverse=True)
        elif sort_by == "Claim Ratio ↓":
            companies_sorted = sorted(companies, key=lambda x: x["claim_ratio"], reverse=True)
        elif sort_by == "Premium ↑":
            companies_sorted = sorted(
                companies,
                key=lambda x: int(
                    x["premium_range"].replace("₹", "").replace(",", "").split("–")[0].strip().split()[0]
                ),
            )
        else:
            companies_sorted = sorted(companies, key=lambda x: x["network_score"], reverse=True)

        # ── Company cards (3 per row) ───────────────────────────────────
        # Colours as literals — CSS variables don't resolve inside st.markdown columns
        CLR_WHITE  = "#ffffff"
        CLR_FOG    = "#e8ecf2"
        CLR_FROST  = "#f4f6fa"
        CLR_MIST   = "#8892a4"
        CLR_INK    = "#0a0e1a"

        for i in range(0, len(companies_sorted), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j >= len(companies_sorted):
                    break

                c      = companies_sorted[i + j]
                is_rec = c.get("recommended", False)
                is_sel = st.session_state.selected_company == c["name"]

                border    = "2px solid #00c9b1" if (is_sel or is_rec) else f"1.5px solid {CLR_FOG}"
                shadow    = ("0 0 0 4px rgba(0,201,177,0.18)" if is_sel
                             else "0 0 0 4px rgba(0,201,177,0.08)" if is_rec
                             else "0 2px 12px rgba(0,0,0,0.06)")
                best_html = (
                    "<div style='position:absolute;top:10px;right:10px;background:#00c9b1;"
                    "color:#0a0e1a;font-size:0.58rem;font-weight:800;letter-spacing:0.1em;"
                    "padding:3px 8px;border-radius:4px;'>BEST</div>"
                ) if is_rec else ""

                full   = int(c["rating"])
                half   = 1 if (c["rating"] - full) >= 0.4 else 0
                stars  = "★" * full + ("½" if half else "") + "☆" * (5 - full - half)

                bar_color = ins_data["color"]
                nr_score  = c["network_score"]
                cr_score  = min(100, c["claim_ratio"])

                icon_bg   = c["icon_bg"]
                icon_sym  = c["icon"]
                name      = c["name"]
                plan      = c["plan"]
                rating    = c["rating"]
                claim_r   = c["claim_ratio"]
                premium   = c["premium_range"]

                pros_html = "".join(
                    f"<span style='color:#10b981;font-weight:700;'>✓</span>&nbsp;{pro}<br>"
                    for pro in c["pros"][:2]
                )
                cons_html = (
                    f"<span style='color:#f43f5e;font-weight:700;'>✗</span>&nbsp;{c['cons'][0]}"
                    if c["cons"] else ""
                )

                card_html = (
                    f"<div style='background:{CLR_WHITE};border:{border};border-radius:14px;"
                    f"padding:22px 18px;position:relative;overflow:hidden;"
                    f"box-shadow:{shadow};margin-bottom:4px;'>"

                    f"{best_html}"

                    # Header row
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
                    f"  <div style='width:38px;height:38px;border-radius:9px;background:{icon_bg};"
                    f"              display:flex;align-items:center;justify-content:center;font-size:1.15rem;'>"
                    f"    {icon_sym}"
                    f"  </div>"
                    f"  <div>"
                    f"    <div style='font-family:Syne,sans-serif;font-size:0.88rem;"
                    f"                font-weight:700;color:{CLR_INK};'>{name}</div>"
                    f"    <div style='font-size:0.68rem;color:{CLR_MIST};'>{plan}</div>"
                    f"  </div>"
                    f"</div>"

                    # Stars
                    f"<div style='font-size:0.80rem;color:#f59e0b;margin-bottom:12px;'>"
                    f"  {stars} <span style='color:{CLR_MIST};font-size:0.72rem;'>({rating})</span>"
                    f"</div>"

                    # Claim ratio bar
                    f"<div style='margin-bottom:10px;'>"
                    f"  <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
                    f"    <span style='font-size:0.70rem;color:{CLR_MIST};'>Claim Ratio</span>"
                    f"    <span style='font-size:0.74rem;font-weight:700;color:{CLR_INK};'>{claim_r}%</span>"
                    f"  </div>"
                    f"  <div style='height:4px;background:{CLR_FOG};border-radius:4px;overflow:hidden;'>"
                    f"    <div style='height:100%;width:{cr_score}%;background:{bar_color};border-radius:4px;'></div>"
                    f"  </div>"
                    f"</div>"

                    # Network score bar
                    f"<div style='margin-bottom:12px;'>"
                    f"  <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
                    f"    <span style='font-size:0.70rem;color:{CLR_MIST};'>Network Score</span>"
                    f"    <span style='font-size:0.74rem;font-weight:700;color:{CLR_INK};'>{nr_score}/100</span>"
                    f"  </div>"
                    f"  <div style='height:4px;background:{CLR_FOG};border-radius:4px;overflow:hidden;'>"
                    f"    <div style='height:100%;width:{nr_score}%;background:{bar_color};"
                    f"                border-radius:4px;opacity:0.55;'></div>"
                    f"  </div>"
                    f"</div>"

                    # Premium pill
                    f"<div style='background:{CLR_FROST};border-radius:8px;padding:9px 11px;"
                    f"            display:flex;justify-content:space-between;margin-bottom:12px;'>"
                    f"  <span style='font-size:0.68rem;color:{CLR_MIST};'>Est. Premium</span>"
                    f"  <span style='font-size:0.78rem;font-weight:700;color:{bar_color};'>{premium}</span>"
                    f"</div>"

                    # Pros / cons
                    f"<div style='font-size:0.70rem;color:{CLR_MIST};line-height:1.9;'>"
                    f"  {pros_html}{cons_html}"
                    f"</div>"

                    f"</div>"
                )

                with col:
                    st.markdown(card_html, unsafe_allow_html=True)
                    btn_label = "✓ Selected" if is_sel else "Select Plan"
                    if st.button(btn_label, key=f"sel_{name}_{ins_type}", use_container_width=True):
                        st.session_state.selected_company = name
                        st.rerun()

        # ── Full Comparison Table ───────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Quick Reference</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Side-by-Side Comparison</div>', unsafe_allow_html=True)

        rows_html = ""
        for c in companies_sorted:
            net_val    = f"{c['cashless']:,}" if c["cashless"] > 0 else f"{c['network_score']}/100"
            best_cls   = "best-cell" if c.get("recommended") else ""
            badge_cls  = c["badge_cls"]
            badge_text = c["badge"]
            icon_name  = c["icon"]
            comp_name  = c["name"]
            claim_r    = c["claim_ratio"]
            premium    = c["premium_range"]
            stars      = "★" * int(c["rating"])
            rating_val = c["rating"]
            rows_html += (
                f"<tr>"
                f"<td class='company-cell {best_cls}'>{icon_name} {comp_name}</td>"
                f"<td class='{best_cls}'>{claim_r}%</td>"
                f"<td>{net_val}</td>"
                f"<td>{premium}</td>"
                f"<td>{stars} {rating_val}</td>"
                f"<td><span class='badge-pill {badge_cls}'>{badge_text}</span></td>"
                f"</tr>"
            )

        headers_html = "".join(f"<th>{h}</th>" for h in ins_data["comparison_headers"])
        st.markdown(f"""
        <table class='cmp-table'>
            <thead><tr>{headers_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Selected Company Detail ─────────────────────────────────────
        if st.session_state.selected_company:
            sel_c = next(
                (c for c in companies if c["name"] == st.session_state.selected_company),
                None,
            )
            if sel_c:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-label">Your Selection</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="sec-title">{sel_c["icon"]} {sel_c["name"]} — Detailed View</div>',
                    unsafe_allow_html=True,
                )

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Claim Ratio", f"{sel_c['claim_ratio']}%")
                d2.metric("Rating", f"{sel_c['rating']} / 5")
                net_disp = f"{sel_c['cashless']:,}" if sel_c["cashless"] > 0 else f"{sel_c['network_score']}/100"
                d3.metric("Network", net_disp)
                d4.metric("Settlement", sel_c["settlement_speed"])

                pros_list = "".join(f"✅ {p}<br>" for p in sel_c["pros"])
                cons_list = "".join(f"⚠️ {c}<br>" for c in sel_c["cons"])
                st.markdown(f"""
                <div class='card' style='border-left:3px solid {ins_data["color"]};'>
                    <div class='card-title'>📋 {sel_c["plan"]} — Plan Highlights</div>
                    <div class='card-body'>
                        <strong>Coverage:</strong> {sel_c["coverage_limit"]}<br>
                        <strong>Premium Range:</strong> {sel_c["premium_range"]}<br><br>
                        {pros_list}{cons_list}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style='background:var(--ink);border-radius:14px;padding:22px 26px;
                            display:flex;align-items:center;justify-content:space-between;'>
                    <div>
                        <div style='font-size:0.68rem;color:var(--mist);letter-spacing:0.1em;
                                    text-transform:uppercase;margin-bottom:6px;'>Selected Plan</div>
                        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                                    color:var(--white);'>{sel_c["name"]} · {sel_c["plan"]}</div>
                        <div style='font-size:0.78rem;color:var(--mist);margin-top:4px;'>
                            Premium: {sel_c["premium_range"]}
                        </div>
                    </div>
                    <div style='background:{ins_data["color"]}22;border:1px solid {ins_data["color"]}44;
                                border-radius:30px;padding:8px 20px;font-size:0.78rem;
                                font-weight:700;color:{ins_data["color"]};'>
                        ✓ Plan Selected
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        # No type selected yet — show prompt
        st.markdown("""
        <div style='background:var(--white);border:1.5px dashed var(--fog);border-radius:14px;
                    padding:44px;text-align:center;margin-top:10px;'>
            <div style='font-size:2.5rem;margin-bottom:12px;'>☝️</div>
            <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                        color:var(--ink);margin-bottom:6px;'>Select an Insurance Category Above</div>
            <div style='font-size:0.83rem;color:var(--mist);'>
                Choose Health, Car, or Life to see India's best providers with detailed comparisons.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: PREDICT COST
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Predict":

    st.markdown("""
    <div class="page-hero" style='background:linear-gradient(135deg,#0a0e1a 0%,#1c2233 60%,#2d3650 100%);'>
        <div class="tag">XGBoost · AI Model</div>
        <h1>Predict Insurance Cost</h1>
        <p>Enter policyholder details to get an AI-powered insurance cost estimate
           using our XGBoost regression model.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.data is None:
        st.info("👈  Upload a CSV file from the sidebar, then train the model to get started.")

    elif not st.session_state.trained:
        st.info("✅  Dataset loaded. Click **Train Model** in the sidebar to continue.")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)

    else:
        p = st.session_state.predictor
        st.markdown(f"""
        <div class="metric-strip">
            <div class="mstrip-box">
                <div class="dot" style="background:#00c9b1;"></div>
                <div class="mlabel">Mean Abs. Error</div>
                <div class="mvalue">₹{p.mae:,.0f}</div>
            </div>
            <div class="mstrip-box">
                <div class="dot" style="background:#8b5cf6;"></div>
                <div class="mlabel">R² Score</div>
                <div class="mvalue">{p.r2:.4f}</div>
            </div>
            <div class="mstrip-box">
                <div class="dot" style="background:#f59e0b;"></div>
                <div class="mlabel">Algorithm</div>
                <div class="mvalue" style="font-size:1rem;">XGBoost</div>
            </div>
            <div class="mstrip-box">
                <div class="dot" style="background:#f43f5e;"></div>
                <div class="mlabel">Train Rows</div>
                <div class="mvalue">{len(st.session_state.data):,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        data_vals = st.session_state.data

        st.markdown('<div class="sec-label">Form</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Enter Policyholder Details</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">Personal Information</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        age      = col1.number_input("Age",                min_value=0,   max_value=120,  value=30)
        bmi      = col2.number_input("BMI",                min_value=0.0, max_value=80.0, value=25.0, step=0.1)
        children = col3.number_input("Number of Children", min_value=0,   max_value=20,   value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">Profile & Location</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        sex    = col4.selectbox("Sex",    sorted(data_vals["sex"].dropna().astype(str).str.lower().unique()))
        smoker = col5.selectbox("Smoker", sorted(data_vals["smoker"].dropna().astype(str).str.lower().unique()))
        state  = col6.selectbox("State",  sorted(data_vals["state"].dropna().astype(str).str.lower().unique()))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">Financial Details</div>', unsafe_allow_html=True)
        col7, col8, col9, col10 = st.columns(4)
        annual_income = col7.number_input("Annual Income (LPA)",  min_value=0.0, max_value=1000.0, value=10.0, step=0.5)
        family_income = col8.number_input("Family Income (LPA)",  min_value=0.0, max_value=2000.0, value=15.0, step=0.5)
        num_policies  = col9.number_input("Insurance Policies",   min_value=0,   max_value=50,     value=1)
        num_vehicles  = col10.number_input("Vehicles Owned",      min_value=0,   max_value=50,     value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        # Show selected company if any
        if st.session_state.selected_insurance_type and st.session_state.selected_company:
            ins_data = INSURANCE_DATA[st.session_state.selected_insurance_type]
            sel_c    = next(
                (c for c in ins_data["companies"] if c["name"] == st.session_state.selected_company),
                None,
            )
            if sel_c:
                st.markdown(f"""
                <div style='background:var(--frost);border:1px solid var(--fog);border-radius:12px;
                            padding:14px 18px;display:flex;align-items:center;gap:12px;margin-bottom:16px;'>
                    <span style='font-size:1.4rem;'>{ins_data["icon"]}</span>
                    <div>
                        <div style='font-size:0.70rem;color:var(--mist);letter-spacing:0.08em;
                                    text-transform:uppercase;'>Selected from Explorer</div>
                        <div style='font-family:Syne,sans-serif;font-weight:700;font-size:0.88rem;
                                    color:var(--ink);'>
                            {sel_c["name"]} · {sel_c["plan"]} · {ins_data["title"]}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔮  Calculate Insurance Cost"):
            input_data = {
                "age": age, "bmi": bmi, "sex": sex, "children": children,
                "smoker": smoker, "state": state,
                "annual_income_lpa": annual_income, "family_income_lpa": family_income,
                "num_insurance_policies": num_policies, "num_vehicles": num_vehicles,
            }
            try:
                result = p.predict_one(input_data)
                st.session_state.prediction_result = result
                st.session_state.prediction_error  = None
            except Exception as e:
                st.session_state.prediction_result = None
                st.session_state.prediction_error  = str(e)

        if st.session_state.prediction_result is not None:
            r            = st.session_state.prediction_result
            company_note = ""
            if st.session_state.selected_company:
                ins_data = INSURANCE_DATA.get(st.session_state.selected_insurance_type, {})
                sel_c    = next(
                    (c for c in ins_data.get("companies", []) if c["name"] == st.session_state.selected_company),
                    None,
                )
                if sel_c:
                    company_note = (
                        f'<div class="company-rec">'
                        f'Recommended Provider · <strong>{sel_c["name"]}</strong> · {sel_c["plan"]}'
                        f'</div>'
                    )

            st.markdown(f"""
            <div class="result-card">
                <div class="rlabel">Estimated Annual Insurance Cost</div>
                <div class="ramount">₹ {r:,.2f}</div>
                <div class="rsub">XGBoost · Log-target regression · Trained on your dataset</div>
                {company_note}
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.prediction_error:
            st.error(f"Prediction failed: {st.session_state.prediction_error}")


# ══════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Dataset":

    st.markdown("""
    <div class="page-hero">
        <div class="tag">Data Explorer</div>
        <h1>Dataset Overview</h1>
        <p>Inspect the loaded dataset, review summary statistics, and understand the feature distribution.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.data is None:
        st.info("Upload a CSV from the sidebar to view dataset details.")
    else:
        df = st.session_state.data
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",       f"{len(df):,}")
        c2.metric("Features",         len(EXPECTED_COLS) - 1)
        c3.metric("Missing Handling", "Auto")

        st.markdown('<div class="sec-title" style="margin-top:28px;">Dataset Preview</div>',    unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown('<div class="sec-title" style="margin-top:28px;">Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "About":

    st.markdown("""
    <div class="page-hero">
        <div class="tag">About InsureIQ</div>
        <h1>Smart Insurance Platform</h1>
        <p>Built for India — combining AI prediction with real-world insurance company data
           to help you make better decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Platform Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">What InsureIQ Offers</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧭 Insurance Explorer</div>
            <div class="card-body">
                Browse and compare Health, Car, and Life insurance providers across India.
                Filter by claim ratio, premium, network size, and ratings. Select your preferred
                plan and carry it into the prediction flow.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r1c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔮 AI Cost Predictor</div>
            <div class="card-body">
                Upload your own dataset, train an XGBoost regression model in seconds, and get
                precise annual insurance cost estimates. The model uses log-target training for
                robustness on skewed cost data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Model Details</div>
            <div class="card-body">
                XGBoost regressor with 500 estimators, learning rate 0.05, max depth 6.
                Preprocessing includes median imputation for numerics and one-hot encoding for
                categoricals. Target is log-transformed for stability.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🗂️ Input Features</div>
            <div class="card-body">
                Age, BMI, Sex, Children, Smoker status, State, Annual &amp; Family Income (LPA),
                Number of Policies, Vehicles Owned. Engineered features include Age×BMI,
                Income-per-Child, and Coverage Load.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Built With</div>', unsafe_allow_html=True)

    tech_tags = ["streamlit", "xgboost", "scikit-learn", "pandas", "numpy"]
    tags_html = "".join(
        f"<span style='display:inline-block;background:var(--frost);border:1px solid var(--fog);"
        f"border-radius:6px;font-size:0.76rem;font-weight:700;padding:5px 13px;"
        f"margin:3px 4px 3px 0;color:var(--ink-3);'>{t}</span>"
        for t in tech_tags
    )
    st.markdown(f'<div class="card"><div class="card-body">{tags_html}</div></div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>InsureIQ</strong> · Smart Insurance Platform · Powered by XGBoost &amp; Streamlit · © 2025
</div>
""", unsafe_allow_html=True)