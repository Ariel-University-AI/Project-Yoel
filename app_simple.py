"""
נדלניסט — גרסה מפושטת
Run:  streamlit run app_simple.py
"""
import pathlib
import datetime
import re
import json
import requests
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import streamlit as st

BASE          = pathlib.Path(__file__).parent
MODEL_PATH    = BASE / "model.pkl"
APT_ML_PATH   = BASE / "DATA_FILES" / "apartments_ml_ready.csv"
APT_DISP_PATH = BASE / "DATA_FILES" / "apartments_display.csv"
POI_PATH      = BASE / "DATA_FILES" / "ISRAEL_POINTS_FILTERED_GEO.csv"

_DISTRICT_MAP = {
    "מחוז ירושלים": [
        "ירושלים", "בית שמש", "מעלה אדומים", "גבעת זאב", "ביתר עילית",
        "מודיעין עילית", "אפרת", "קרית ארבע", "מבשרת ציון", "בית לחם הגלילית",
        "עפרה", "קדומים", "אריאל", "מעלה שומרון",
    ],
    "מחוז תל אביב": [
        "תל אביב-יפו", "תל אביב יפו", "בת ים", "חולון", "בני ברק",
        "גבעתיים", "רמת גן", "אור יהודה", "גבעת שמואל", "קרית אונו",
        "יהוד-מונוסון", "יהוד מונוסון", "אלעד",
    ],
    "מחוז המרכז": [
        "ראשון לציון", "פתח תקווה", "רחובות", "נס ציונה", "לוד", "רמלה",
        "מודיעין-מכבים-רעות", "מודיעין מכבים רעות", "רעננה", "כפר סבא",
        "הוד השרון", "נתניה", "הרצליה", "רמת השרון", "ראש העין", "יבנה",
        "גדרה", "באר יעקב", "שוהם", "טייבה", "קלנסווה", "כפר קאסם",
        "ג'לג'וליה", "כפר יונה", "עמנואל", "בית אריה",
    ],
    "מחוז חיפה": [
        "חיפה", "קרית אתא", "קרית ביאליק", "קרית מוצקין", "קרית ים",
        "נשר", "טירת כרמל", "קרית טבעון", "זכרון יעקב",
        "פרדס חנה-כרכור", "פרדס חנה כרכור", "חדרה", "אור עקיבא",
        "בנימינה-גבעת עדה", "בנימינה גבעת עדה", "עכו", "נהריה",
        "שפרעם", "בקה אל-גרבייה", "אום אל-פחם",
    ],
    "מחוז הצפון": [
        "נצרת", "נוף הגליל", "נצרת עילית", "עפולה", "קרית שמונה",
        "צפת", "טבריה", "כרמיאל", "מגדל העמק", "בית שאן",
        "מעלות-תרשיחא", "מעלות תרשיחא", "יוקנעם עילית", "סח'נין",
        "ירכא", "כפר יאסיף", "מגדל", "רמת ישי", "קרית טבעון",
        "בית שאן", "אפולה", "מגדל העמק",
    ],
    "מחוז הדרום": [
        "באר שבע", "אשדוד", "אשקלון", "דימונה", "קרית מלאכי",
        "נתיבות", "שדרות", "אופקים", "ירוחם", "מצפה רמון",
        "רהט", "קרית גת", "גן יבנה", "קרית עקרון", "יבנה",
        "גדרות", "לכיש", "קרית ארבע", "ערד",
    ],
}

st.set_page_config(
    page_title='נדלניסט',
    layout="wide",
    page_icon="🏠",
)

st.markdown("""
<style>
  /* ── Google Fonts — Rubik (RTL Hebrew) ──────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;700&display=swap');

  /* ── Madlan Design Tokens ────────────────────────────────────────────── */
  :root {
    --primary:       #35BA85;
    --primary-dark:  #1A6B5A;
    --primary-light: #E0F5EC;
    --bg:            #F5F7F6;
    --white:         #FFFFFF;
    --border:        #D4EBE2;
    --text-dark:     #1A2E2A;
    --text-mid:      #4A5E58;
    --text-light:    #8A9E98;
    --red:           #E8334A;
    --orange:        #F5A623;
    --shadow:        0 2px 12px rgba(26,107,90,0.08);
    --shadow-hover:  0 8px 32px rgba(26,107,90,0.18);
  }

  /* ── Hide Streamlit chrome ───────────────────────────────────────────── */
  #MainMenu,
  header[data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stSidebarCollapsedControl"],
  footer { display: none !important; }
  /* Hide the << collapse button inside the sidebar */
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
  section[data-testid="stSidebar"] button[kind="header"],
  section[data-testid="stSidebar"] > div > div > button { display: none !important; }
  .block-container { padding-top: 0.75rem !important; padding-bottom: 2rem; }

  /* ── Font & Base ─────────────────────────────────────────────────────── */
  .stApp, body {
    background-color: var(--bg) !important;
    font-family: 'Rubik', 'Assistant', -apple-system, sans-serif !important;
    color: var(--text-dark) !important;
  }

  /* ── Metrics ─────────────────────────────────────────────────────────── */
  [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--primary-dark) !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: var(--text-light) !important;
    font-weight: 500 !important;
  }
  [data-testid="metric-container"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: var(--shadow) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
  }
  [data-testid="metric-container"]:hover {
    box-shadow: var(--shadow-hover) !important;
    transform: translateY(-2px);
  }

  /* ── Labels ──────────────────────────────────────────────────────────── */
  label { font-weight: 600 !important; color: var(--text-dark) !important; }

  /* ── Verdict boxes ───────────────────────────────────────────────────── */
  .verdict-good {
    background: var(--primary-light);
    border: 2px solid var(--primary);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(53,186,133,0.12);
  }
  .verdict-ok {
    background: #FFF8ED;
    border: 2px solid var(--orange);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(245,166,35,0.10);
  }
  .verdict-bad {
    background: #FEF0F2;
    border: 2px solid var(--red);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(232,51,74,0.10);
  }

  /* ── Cards / bordered containers ─────────────────────────────────────── */
  [data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 16px !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow) !important;
    background: var(--white) !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    transform: translateY(-4px) !important;
    box-shadow: var(--shadow-hover) !important;
    border-color: var(--primary) !important;
  }

  /* ── Equal height for card rows ──────────────────────────────────────── */
  [data-testid="stHorizontalBlock"] { align-items: stretch !important; }
  [data-testid="stVerticalBlockBorderWrapper"] { height: 100% !important; }
  [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] {
    margin-top: auto !important;
    padding-top: 12px !important;
  }

  /* ── Buttons ─────────────────────────────────────────────────────────── */
  .stButton > button {
    background: var(--primary-dark) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Rubik', sans-serif !important;
    transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(26,107,90,0.20) !important;
  }
  .stButton > button:hover {
    background: #134d41 !important;
    box-shadow: var(--shadow-hover) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Expanders ───────────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow) !important;
  }
  [data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-dark) !important;
  }

  /* ── DataFrames ──────────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
  }

  /* ── Inputs & Selects ────────────────────────────────────────────────── */
  .stTextInput input, .stNumberInput input {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'Rubik', sans-serif !important;
    background: var(--white) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(53,186,133,0.15) !important;
  }
  [data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--white) !important;
    font-family: 'Rubik', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  [data-baseweb="select"] > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(53,186,133,0.15) !important;
  }

  /* ── Slider ──────────────────────────────────────────────────────────── */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--primary-dark) !important;
    border-color: var(--primary-dark) !important;
  }

  /* ── Layout: sidebar fixed RIGHT ────────────────────────────────────────
     Sidebar section is removed from normal flow (position:fixed; right:0).
     No flex/grid ordering needed — the sidebar just pins to the right.
     Main content gets margin-right:288px to avoid sliding under sidebar. */

  [data-testid="stAppViewContainer"] {
    direction: ltr !important;
  }

  /* Collapse sidebar wrapper so it takes no flex-space */
  [data-testid="stAppViewContainer"] > div:has([data-testid="stSidebar"]) {
    flex: 0 0 0 !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    overflow: visible !important;
  }

  /* Main wrapper: fill all available space */
  [data-testid="stAppViewContainer"] > div:has([data-testid="stMain"]) {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
    transform: none !important;
  }

  /* Sidebar section: pinned to RIGHT of viewport */
  section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    right: 0 !important;
    left: auto !important;
    width: 288px !important;
    min-width: 288px !important;
    max-width: 288px !important;
    height: 100dvh !important;
    background: var(--white) !important;
    border-right: none !important;
    border-left: 1px solid var(--border) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    transform: none !important;
    margin: 0 !important;
    z-index: 999 !important;
    display: block !important;
    visibility: visible !important;
  }
  section[data-testid="stSidebar"] > div {
    transform: none !important;
    visibility: visible !important;
  }

  /* Main section: offset from right to clear the fixed sidebar */
  section[data-testid="stMain"] {
    margin-left: 0 !important;
    margin-right: 288px !important;
    width: calc(100% - 288px) !important;
    max-width: calc(100% - 288px) !important;
    min-width: 0 !important;
    direction: ltr !important;
    align-items: stretch !important;
    transform: none !important;
  }

  [data-testid="stMainBlockContainer"],
  .block-container {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1rem !important;
    margin: 0 !important;
    direction: rtl !important;
  }
  /* Columns themselves stay LTR so visual order is preserved (col1=left, col2=right) */
  section[data-testid="stMain"] [data-testid="stHorizontalBlock"] {
    direction: ltr !important;
  }
  section[data-testid="stMain"] [data-testid="column"] {
    direction: rtl !important;
  }
  [data-testid="stSidebarContent"] { padding: 0 !important; }

  /* Logo area */
  .sidebar-logo {
    padding: 24px 20px 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
    font-family: 'Rubik', sans-serif;
    direction: rtl;
  }

  /* Hide widget label above the radio group */
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    display: none !important;
  }

  /* Hide the radio dot/circle */
  section[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
  }

  /* Nav items */
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px !important;
    padding: 0 12px !important;
  }
  section[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 13px 16px !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: var(--text-mid) !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    border: none !important;
    margin-bottom: 2px !important;
  }
  section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--primary-light) !important;
    color: var(--primary-dark) !important;
  }
  /* Selected nav item */
  section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: var(--primary-light) !important;
    color: var(--primary-dark) !important;
    font-weight: 700 !important;
    border-left: 3px solid var(--primary) !important;
  }

  /* ── Section header with green accent bar ────────────────────────────── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
    font-family: 'Rubik', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-dark);
    direction: rtl;
  }
  .section-header::after {
    content: '';
    display: block;
    width: 4px;
    height: 24px;
    background: var(--primary);
    border-radius: 2px;
    flex-shrink: 0;
    order: -1;
  }

  /* ── RTL: Hebrew text right-aligned everywhere ───────────────────────── */

  /* Main: all text right-aligned */
  section[data-testid="stMain"],
  section[data-testid="stMain"] * {
    text-align: right !important;
  }

  /* direction:rtl on every Streamlit widget container + text nodes */
  section[data-testid="stMain"] p,
  section[data-testid="stMain"] span,
  section[data-testid="stMain"] h1,
  section[data-testid="stMain"] h2,
  section[data-testid="stMain"] h3,
  section[data-testid="stMain"] h4,
  section[data-testid="stMain"] h5,
  section[data-testid="stMain"] h6,
  section[data-testid="stMain"] label,
  section[data-testid="stMain"] li,
  section[data-testid="stMain"] td,
  section[data-testid="stMain"] th,
  section[data-testid="stMain"] [data-testid="stMarkdownContainer"],
  section[data-testid="stMain"] [data-testid="stText"],
  section[data-testid="stMain"] [data-testid="stMetricLabel"],
  section[data-testid="stMain"] [data-testid="stMetricValue"],
  section[data-testid="stMain"] [data-testid="stMetricDelta"],
  section[data-testid="stMain"] [data-testid="stWidgetLabel"],
  section[data-testid="stMain"] [data-testid="stRadio"],
  section[data-testid="stMain"] [data-testid="stSelectbox"],
  section[data-testid="stMain"] [data-testid="stNumberInput"],
  section[data-testid="stMain"] [data-testid="stTextInput"],
  section[data-testid="stMain"] [data-testid="stSlider"],
  section[data-testid="stMain"] [data-testid="stMultiSelect"],
  section[data-testid="stMain"] [data-testid="stCheckbox"],
  section[data-testid="stMain"] [data-testid="stExpander"],
  section[data-testid="stMain"] [data-testid="stVerticalBlock"],
  section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
    direction: rtl !important;
  }

  /* Inputs / dataframes: keep LTR */
  section[data-testid="stMain"] input,
  section[data-testid="stMain"] textarea {
    direction: ltr !important;
    text-align: left !important;
  }
  section[data-testid="stMain"] [data-testid="stDataFrameContainer"],
  section[data-testid="stMain"] [data-testid="stDataFrameContainer"] * {
    text-align: left !important;
  }

  /* Sidebar: RTL text */
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] * {
    text-align: right !important;
    direction: rtl !important;
  }
  /* Sidebar inputs stay LTR */
  section[data-testid="stSidebar"] input {
    direction: ltr !important;
    text-align: left !important;
  }
</style>
""", unsafe_allow_html=True)



# ── helper: render a block of Hebrew HTML right-to-left ───────────────────────
def rtl(html: str, extra_style: str = "") -> None:
    st.markdown(
        f'<div dir="rtl" style="text-align:right;line-height:1.8;font-size:0.95rem;{extra_style}">'
        f'{html}</div>',
        unsafe_allow_html=True,
    )


# ─── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df_ml = pd.read_csv(APT_ML_PATH,   encoding="utf-8-sig")
    df_d  = pd.read_csv(APT_DISP_PATH, encoding="utf-8-sig")
    return df_ml, df_d


@st.cache_data(show_spinner=False)
def compute_predictions():
    mdl   = joblib.load(str(MODEL_PATH))
    df_ml = pd.read_csv(str(APT_ML_PATH),   encoding="utf-8-sig")
    df_d  = pd.read_csv(str(APT_DISP_PATH), encoding="utf-8-sig")
    X     = df_ml.drop(columns=["dealAmount"])
    df_d  = df_d.copy()
    df_d["predicted"]       = mdl.predict(X)
    df_d["gap_pct"]         = (df_d["predicted"] - df_d["dealAmount"]) / df_d["dealAmount"] * 100
    df_d["viability_score"] = ((50 + df_d["gap_pct"] * 1.5) / 10).clip(0, 10).round(1)
    return df_d


@st.cache_data(show_spinner=False)
def compute_area_stats():
    mdl   = joblib.load(str(MODEL_PATH))
    df_ml = pd.read_csv(str(APT_ML_PATH),   encoding="utf-8-sig")
    df_d  = pd.read_csv(str(APT_DISP_PATH), encoding="utf-8-sig")
    X     = df_ml.drop(columns=["dealAmount"])
    df_d  = df_d.copy()
    df_d["predicted"] = mdl.predict(X)
    df_d["gap_pct"]   = (df_d["predicted"] - df_d["dealAmount"]) / df_d["dealAmount"] * 100

    def _trend(g):
        if g["deal_year"].nunique() < 2:
            return 0.0
        slope = np.polyfit(g["deal_year"].values, g["dealAmount"].values, 1)[0]
        return float(round(slope / g["dealAmount"].mean() * 100, 2))

    try:
        trend_s = df_d.groupby("settlementNameHeb").apply(_trend, include_groups=False)
    except TypeError:
        trend_s = df_d.groupby("settlementNameHeb").apply(_trend)
    trend_s = trend_s.rename("trend_pct_yr")

    stats = df_d.groupby("settlementNameHeb").agg(
        avg_price  = ("dealAmount",      "mean"),
        avg_gap    = ("gap_pct",         "mean"),
        deal_count = ("dealAmount",      "count"),
        avg_socio  = ("socio_index_avg", "mean"),
    ).join(trend_s).reset_index()
    return stats


@st.cache_data(show_spinner=False)
def get_baselines():
    df_ml = pd.read_csv(str(APT_ML_PATH),   encoding="utf-8-sig")
    df_d  = pd.read_csv(str(APT_DISP_PATH), encoding="utf-8-sig")
    df_ml = df_ml.copy()
    df_ml["settlementNameHeb"] = df_d["settlementNameHeb"]
    return df_ml.groupby("settlementNameHeb").median()


# ─── URL scraping helpers (ported from app.py) ────────────────────────────────

_YAD2_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _get_num(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(str(v).replace(",", "").replace(" ", "").replace("\xa0", ""))
            except (ValueError, TypeError):
                pass
    return None


def _get_str(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v and isinstance(v, str):
            return v.strip()
    return None


def _meta(html: str, prop: str):
    m = re.search(
        rf'<meta[^>]+(?:property|name)=["\'](?:og:)?{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    ) or re.search(
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:og:)?{re.escape(prop)}["\']',
        html, re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _parse_yad2_html(html: str) -> dict:
    """Parse Yad2 listing HTML → data dict (used by scraper + manual paste)."""
    m = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        raw = html.strip()
        if raw.startswith("{"):
            try:
                json.loads(raw)
                m = type("_M", (), {"group": lambda self, i: raw})()
            except json.JSONDecodeError:
                pass
    if not m:
        # Fallback: extract from meta tags
        title = _meta(html, "title") or ""
        desc  = _meta(html, "description") or ""
        combined = title + " " + desc
        rm = re.search(r'(\d+(?:\.\d)?)\s*חדרים', combined)
        rooms = float(rm.group(1)) if rm else None
        parts = [p.strip() for p in title.split(",")]
        city = hood = street = house_num = None
        if len(parts) >= 5:
            street_raw = parts[1]
            hood = parts[3]
            city = parts[4].split("|")[0].strip()
            sm = re.match(r'^(.+?)\s+(\d+)$', street_raw)
            street   = sm.group(1).strip() if sm else street_raw.strip()
            house_num = sm.group(2) if sm else None
        elif len(parts) >= 4:
            street_raw = parts[1]
            hood = parts[2]
            city = parts[3].split("|")[0].strip()
            sm = re.match(r'^(.+?)\s+(\d+)$', street_raw)
            street   = sm.group(1).strip() if sm else street_raw.strip()
            house_num = sm.group(2) if sm else None
        if rooms or city:
            return {"price": None, "rooms": rooms, "area": None, "floor": None,
                    "city": city, "neighborhood": hood, "street": street,
                    "house_num": house_num, "lat": None, "lon": None,
                    "error": "נמצאו פרטים חלקיים — חסר מחיר, השלם ידנית.", "needs_manual": True}
        return {"error": "לא נמצא __NEXT_DATA__ — ודא שהדבקת את קוד המקור המלא (Ctrl+U → Ctrl+A → Ctrl+C).", "needs_manual": True}

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"error": "שגיאה בפענוח JSON.", "needs_manual": True}

    pp      = data.get("props", {}).get("pageProps", {})
    listing = None
    try:
        listing = pp["dehydratedState"]["queries"][0]["state"]["data"]
    except (KeyError, IndexError, TypeError):
        pass
    if not listing:
        for path in [["listing"], ["item"], ["itemData"], ["listingData"], ["ad"]]:
            try:
                obj = pp
                for k in path:
                    obj = obj[k]
                if isinstance(obj, dict) and ("price" in obj or "priceOnly" in obj):
                    listing = obj; break
            except (KeyError, TypeError):
                continue
    if not listing:
        return {"error": "מבנה הנתונים לא מוכר.", "needs_manual": True}

    price = _get_num(listing, "price", "priceOnly", "priceFormatted")
    addr  = listing.get("address") or {}
    add_d = listing.get("additionalDetails") or {}
    inp   = listing.get("inProperty") or {}

    def _txt(d, *keys):
        for k in keys:
            v = d.get(k)
            if isinstance(v, dict):
                t = v.get("text") or v.get("textHeb")
                if t and isinstance(t, str): return t.strip()
            if v and isinstance(v, str): return v.strip()
        return None

    city   = _txt(addr, "city")   or _get_str(listing, "city", "cityHeb")
    street = _txt(addr, "street") or _get_str(listing, "street", "streetHeb")
    hood   = _txt(addr, "neighborhood") or _get_str(listing, "neighborhood")
    house  = addr.get("house") or {}
    floor  = (_get_num(house, "floor") or _get_num(add_d, "floor", "floorFormatted") or _get_num(listing, "floor"))
    rooms  = (_get_num(add_d, "roomsCount", "rooms", "roomNum") or _get_num(inp, "rooms") or _get_num(listing, "rooms", "roomNum"))
    area   = (_get_num(add_d, "squareMeter", "area", "meter") or _get_num(inp, "squareMeter", "area") or _get_num(listing, "squareMeter", "area", "meter"))
    coords_d = addr.get("coords") or {}
    lat = coords_d.get("lat")
    lon = coords_d.get("lon")
    # House number — try address.house.number / address.houseNum / listing.houseNum
    _hn = (_get_num(house, "number") or _get_num(addr, "houseNum", "houseNumber")
           or _get_num(listing, "houseNum", "houseNumber"))
    house_num = str(int(_hn)) if _hn is not None else (_get_str(house, "number") or _get_str(addr, "houseNum"))
    if not price:
        return {"error": "לא נמצא מחיר במודעה.", "needs_manual": True}
    return {"price": price, "rooms": rooms, "area": area, "floor": floor,
            "city": city, "neighborhood": hood, "street": street,
            "house_num": house_num, "lat": lat, "lon": lon}


def scrape_yad2_listing(url: str) -> dict:
    """Fetch a YAD2 item page using curl_cffi Chrome impersonation."""
    if not re.search(r"yad2\.co\.il/.*item/", url):
        return {"error": "הקישור אינו תקין — חייב להיות קישור לנכס ב-yad2.co.il", "needs_manual": False}
    import time as _time
    _extra = {
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site", "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    html = None
    try:
        from curl_cffi import requests as _cr
        for _attempt, _ver in enumerate(["chrome133", "chrome131", "chrome124", "chrome120", "chrome116"]):
            try:
                if _attempt: _time.sleep(1.5)
                s = _cr.Session(impersonate=_ver)
                s.headers.update(_extra)
                resp = s.get(url, timeout=25)
                if resp.status_code == 200 and len(resp.text) > 5000:
                    html = resp.text; break
            except Exception:
                continue
    except ImportError:
        pass
    if html is None:
        try:
            resp = requests.get(url, headers={**_YAD2_HEADERS, **_extra}, timeout=15)
            html = resp.text
        except Exception:
            return {"error": "שגיאת חיבור — בדוק חיבור לאינטרנט.", "needs_manual": True}
    return _parse_yad2_html(html)


_HEB_FLOOR_ORDINALS = {
    "ראשונה": 1, "ראשון": 1, "שנייה": 2, "שני": 2, "שלישית": 3, "שלישי": 3,
    "רביעית": 4, "רביעי": 4, "חמישית": 5, "חמישי": 5, "שישית": 6, "שישי": 6,
    "שביעית": 7, "שביעי": 7, "שמינית": 8, "שמיני": 8, "קרקע": 0,
}

_HEB_CITIES = [
    "תל אביב", "ירושלים", "חיפה", "ראשון לציון", "פתח תקווה", "אשדוד",
    "נתניה", "באר שבע", "בני ברק", "רמת גן", "בת ים", "חולון", "אשקלון",
    "רחובות", "הרצליה", "כפר סבא", "מודיעין", "לוד", "רמלה", "נהריה",
    "רעננה", "גבעתיים", "עכו", "אילת", "טבריה", "צפת", "חדרה",
    "יהוד", "אור יהודה", "גבעת שמואל", "רמת השרון", "הוד השרון", "גדרה",
    "ראש העין", "קריית גת", "קריית שמונה", "אריאל", "מעלה אדומים",
]


def _parse_realestate_text(text: str) -> dict:
    """Extract price/rooms/area/floor/city from free-text Israeli real-estate description."""
    text = (text
        .replace("״", '"').replace("׳", "'").replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"').replace(" ", " ")
        .replace("–", "-").replace("—", "-")
    )
    price = None
    for pat in [
        r'([\d]{1,3}(?:,[\d]{3})+)\s*₪', r'₪\s*([\d]{1,3}(?:,[\d]{3})+)',
        r'([\d]{4,7})\s*₪',               r'₪\s*([\d]{4,7})',
        r'([\d]{1,3}(?:,[\d]{3})+)\s*שקל', r'מחיר[:\s]*([\d,]+)',
        r'(?:^|[-—|\s])([\d]{1,3}(?:,[\d]{3})+)(?=\s|$|[.,\-])',
    ]:
        mm = re.search(pat, text, re.MULTILINE)
        if mm:
            try:
                v = float(mm.group(1).replace(",", ""))
                if v >= 50_000:
                    price = v
                    break
            except (ValueError, TypeError):
                pass
    if not price:
        mm = re.search(r'([\d]+(?:[.,][\d]+)?)\s*מיליון', text)
        if mm:
            try: price = float(mm.group(1).replace(",", ".")) * 1_000_000
            except (ValueError, TypeError): pass

    rooms = None
    for pat in [r'(\d+[.,]\d)\s*חדרים', r'(\d+[.,]\d)\s*חד', r'(\d+)\s*חדרים',
                r"(\d+)\s*חד\'", r'חדרים[:\s]*(\d+[.,]?\d?)', r'rooms?[:\s]*(\d+(?:\.\d)?)']:
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try: rooms = float(mm.group(1).replace(",", ".")); break
            except (ValueError, TypeError): pass

    area = None
    for pat in [r'(\d+)\s*מ"ר', r"(\d+)\s*מ''", r"(\d+)\s*מ'",
                r'(\d+)\s*מטר\s*(?:רבוע)?', r'שטח[:\s]*(?:כ-?\s*)?(\d+)',
                r'(?:כ-?\s*)(\d+)\s*מ', r'(\d+)\s*sqm']:
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try:
                v = float(mm.group(1))
                if 10 <= v <= 1000:
                    area = v; break
            except (ValueError, TypeError): pass

    floor = None
    for pat in [r'קומה\s*(\d+)', r'(\d+)\s*קומה', r'floor[:\s]*(\d+)']:
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            try: floor = float(mm.group(1)); break
            except (ValueError, TypeError): pass
    if floor is None:
        for word, num in _HEB_FLOOR_ORDINALS.items():
            if re.search(rf'קומה\s+{word}', text):
                floor = float(num); break

    city = None
    for c in _HEB_CITIES:
        if c in text:
            city = c; break

    return {"price": price, "rooms": rooms, "area": area, "floor": floor,
            "city": city, "neighborhood": None, "street": None, "lat": None, "lon": None}


def scrape_facebook_listing(url: str) -> dict:
    """Attempt to scrape a Facebook Marketplace real-estate listing."""
    import time as _time
    html = None
    fb_headers = {
        **_YAD2_HEADERS,
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Mobile Safari/537.36"
        ),
        "Accept-Language": "he-IL,he;q=0.9",
    }
    try:
        from curl_cffi import requests as _cr
        for _attempt, _ver in enumerate(["chrome133", "chrome124", "chrome120"]):
            try:
                if _attempt: _time.sleep(1)
                resp = _cr.Session(impersonate=_ver).get(url, timeout=25)
                html = resp.text; break
            except Exception: continue
    except ImportError:
        pass
    if html is None:
        try:
            resp = requests.get(url, headers=fb_headers, timeout=15)
            html = resp.text
        except Exception:
            return {"error": "לא ניתן להגיע ל-Facebook.", "needs_manual": True, "needs_paste": True}

    _FB_BLOCKED = ["log in to facebook", "create new account", "you must log in",
                   "login_form", "sorry, something went wrong", "error facebook"]
    if any(kw in html[:5000].lower() for kw in _FB_BLOCKED):
        return {"error": "Facebook דורש התחברות לצפייה במודעה זו.", "needs_manual": True, "needs_paste": True}

    og_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']*)["\']', html)
    og_desc  = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']*)["\']', html)
    json_desc = None
    for pat in [r'"body"\s*:\s*\{"text"\s*:\s*"([^"]{20,})"',
                r'"description"\s*:\s*"([^"]{20,})"',
                r'"listing_description"\s*:\s*"([^"]{20,})"']:
        jm = re.search(pat, html)
        if jm:
            json_desc = jm.group(1).encode().decode("unicode_escape", errors="ignore"); break
    json_price = None
    for pat in [r'"listing_price"[^}]*"amount"\s*:\s*"?([\d.]+)"?',
                r'"price_value"\s*:\s*"?([\d,]+)"?', r'"formatted_amount"\s*:\s*"([\d,]+)"']:
        jm = re.search(pat, html)
        if jm:
            try:
                v = float(jm.group(1).replace(",", ""))
                if v >= 50_000: json_price = v
            except (ValueError, TypeError): pass
            if json_price: break

    text = " ".join(filter(None, [
        og_title.group(1) if og_title else "",
        og_desc.group(1)  if og_desc  else "",
        json_desc or "",
    ]))
    if not text.strip() and not json_price:
        return {"error": "לא ניתן לחלץ פרטים מ-Facebook.", "needs_manual": True, "needs_paste": True}

    result = _parse_realestate_text(text)
    if json_price:
        result["price"] = json_price
    if not result["price"]:
        return {"error": "לא נמצא מחיר במודעה.", "needs_manual": True, "needs_paste": True}
    return result


def scrape_listing(url: str) -> dict:
    """Route to the correct scraper based on the listing URL."""
    if re.search(r"yad2\.co\.il/.*item/", url):
        return scrape_yad2_listing(url)
    if re.search(r"facebook\.com/marketplace/item/|fb\.com/marketplace/item/", url):
        return scrape_facebook_listing(url)
    return {"error": "הקישור אינו תקין — יש להזין קישור מ-yad2.co.il או Facebook Marketplace", "needs_manual": False}


def _match_settlement(city, settlements: list):
    """Fuzzy-match a YAD2 city name against our settlement list."""
    import difflib
    if not city:
        return None

    def _norm(s):
        s = s.strip().replace("\xa0", " ").replace("-", " ").replace("–", " ")
        return " ".join(s.split()).lower()

    def _norm_he(s):
        s = _norm(s)
        s = s.replace("יי", "י").replace("וו", "ו").replace("'", "").replace('"', "")
        return s

    c_norm    = _norm(city)
    c_norm_he = _norm_he(city)
    for s in settlements:
        if _norm(s) == c_norm: return s
    for s in settlements:
        if _norm_he(s) == c_norm_he: return s
    for s in settlements:
        s_n = _norm(s)
        if c_norm in s_n or s_n in c_norm: return s
    norm_he_map = {_norm_he(s): s for s in settlements}
    close = difflib.get_close_matches(c_norm_he, norm_he_map.keys(), n=1, cutoff=0.82)
    if close:
        return norm_he_map[close[0]]
    return None


# ─── Yad2 real-time city listing scraper ─────────────────────────────────────

# Yad2 city codes — verified by probing www.yad2.co.il/realestate/forsale?city=<id>
# and checking the fullTitleText / city.text returned in __NEXT_DATA__.
_YAD2_CITY_IDS = {
    # ── Confirmed correct ─────────────────────────────────────────────────────
    "תל אביב יפו": 5000, "תל אביב": 5000,
    "ירושלים": 3000,
    "נתניה": 7400,
    "פתח תקווה": 7900,
    "ראשון לציון": 8300,
    "בת ים": 6200,
    "רחובות": 8400,
    "הרצליה": 6400,
    "כפר סבא": 6900,
    "בני ברק": 6100,
    "רעננה": 8700,
    "מודיעין-מכבים-רעות": 4400, "מודיעין": 4400,
    "עכו": 7600,
    "נצרת": 7300,
    "אילת": 2600,
    "רמת השרון": 800,
    # ── Likely correct (not blocked during testing, but verified via listing city) ─
    "לוד": 7000,
    "רמלה": 8800,
    "קריית גת": 7680,
}


def _fetch_yad2_search_page(city_id, page=1, timeout=20):
    """Fetch one page of Yad2 forsale search results as HTML. Returns (html, error)."""
    url = "https://www.yad2.co.il/realestate/forsale"
    params = {
        "city": city_id,
        "propertyGroup": "apartments",
        "propertyType": "1",
        "page": page,
    }
    page_hdrs = {
        **_YAD2_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.yad2.co.il/realestate/forsale",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    }
    try:
        from curl_cffi import requests as _cr
        import time as _t
        # chrome116 consistently works; try it first, then older fallbacks
        for i, ver in enumerate(["chrome116", "chrome110", "chrome120", "chrome124"]):
            try:
                if i:
                    _t.sleep(1.0)
                s = _cr.Session(impersonate=ver)
                s.headers.update(page_hdrs)
                r = s.get(url, params=params, timeout=timeout)
                if r.status_code == 200 and len(r.text) > 3000:
                    return r.text, None
            except Exception:
                continue
    except ImportError:
        pass
    try:
        r = requests.get(url, params=params, headers=page_hdrs, timeout=timeout)
        if r.status_code == 200:
            return r.text, None
        return None, f"קוד שגיאה {r.status_code}"
    except Exception as exc:
        return None, str(exc)


def _parse_yad2_search_html(html):
    """Extract listing dicts from __NEXT_DATA__ in a Yad2 forsale search page.
    Returns (items_list, error_str).  As of mid-2026, listings live in
    pageProps.feed.{private,agency,platinum,trio}.
    """
    m = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        return None, "לא נמצאו נתונים בעמוד — יד2 ייתכן שחסמה את הגישה."
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, "שגיאה בפענוח נתוני העמוד."

    pp = data.get("props", {}).get("pageProps", {})

    def _collect_from_feed(feed_dict):
        """Collect all real listings from a feed dict keyed by listing type."""
        result = []
        for key in ("private", "agency", "platinum", "trio", "booster"):
            bucket = feed_dict.get(key) or []
            if isinstance(bucket, list):
                result.extend(b for b in bucket if isinstance(b, dict) and b.get("price"))
        return result

    # Primary path: feed is a direct pageProps key (current Yad2 structure)
    top_feed = pp.get("feed")
    if isinstance(top_feed, dict):
        items = _collect_from_feed(top_feed)
        if items:
            return items, None

    # Fallback: buried inside dehydratedState queries
    for q in pp.get("dehydratedState", {}).get("queries", []):
        sd = q.get("state", {}).get("data", {})
        if not isinstance(sd, dict):
            continue
        items = _collect_from_feed(sd)
        if items:
            return items, None
        # Older feed_items structure
        for key in ("feed", "feedData"):
            sub = sd.get(key) or {}
            if isinstance(sub, dict):
                fi = sub.get("feed_items") or sub.get("feedItems") or []
                if fi:
                    return fi, None

    return None, "לא נמצאו מודעות בעמוד. ייתכן שיד2 שינתה את מבנה הנתונים."


def fetch_yad2_city_listings(city_heb, max_pages=3):
    """Fetch real-time apartment listings from Yad2 for a Hebrew city name.
    Returns (DataFrame, error_str). On success, error_str == ''.
    Scrapes www.yad2.co.il/realestate/forsale?city=<id> and parses __NEXT_DATA__.
    """
    import difflib

    # ── 1. Resolve city name → Yad2 city ID ──────────────────────────────────
    city_id = _YAD2_CITY_IDS.get(city_heb)
    if city_id is None:
        norm = city_heb.strip().replace("-", " ").replace("–", " ")
        close = difflib.get_close_matches(norm, list(_YAD2_CITY_IDS.keys()), n=1, cutoff=0.78)
        if close:
            city_id = _YAD2_CITY_IDS[close[0]]
        else:
            return pd.DataFrame(), (
                f"עיר '{city_heb}' אינה ברשימת הערים הנתמכות לחיפוש יד2 כרגע. "
                "נסה עיר גדולה יותר, או השתמש בנתונים ההיסטוריים."
            )

    # ── 2. Fetch and parse pages ──────────────────────────────────────────────
    all_items = []
    for page in range(1, max_pages + 1):
        html, err = _fetch_yad2_search_page(city_id, page=page)
        if err or not html:
            if page == 1:
                return pd.DataFrame(), (
                    f"שגיאת חיבור ליד2: {err or 'תגובה ריקה'}. "
                    "בדוק חיבור לאינטרנט או נסה שוב מאוחר יותר."
                )
            break

        items, parse_err = _parse_yad2_search_html(html)
        if parse_err:
            if page == 1:
                return pd.DataFrame(), parse_err
            break

        real_ads = [
            i for i in items
            if isinstance(i, dict)
            and i.get("type") not in {
                "commercial_promoted", "promoted_native",
                "banner", "promoted", "lead_gen",
            }
        ]
        all_items.extend(real_ads)
        if len(items) < 25:
            break

    if not all_items:
        return pd.DataFrame(), (
            "לא נמצאו מודעות. יד2 ייתכן וחסמה את הגישה — נסה שוב בעוד מספר דקות."
        )

    # ── 3. Parse into DataFrame ───────────────────────────────────────────────
    # Field paths confirmed from live __NEXT_DATA__ (June 2026):
    #   address.area.text        → neighborhood / area label
    #   address.street.text      → street name (may be absent)
    #   address.house.number     → house number
    #   address.house.floor      → floor
    #   additionalDetails.squareMeter → area m²
    #   additionalDetails.roomsCount  → rooms
    #   price                    → asking price (int)
    #   token                    → listing ID for URL /item/<token>
    rows = []
    for item in all_items:
        price = item.get("price")
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if price < 100_000:
            continue

        addr  = item.get("address") or {}
        add_d = item.get("additionalDetails") or {}
        house = addr.get("house") or {}

        def _txt_field(d, key):
            v = d.get(key)
            if isinstance(v, dict):
                return (v.get("text") or v.get("textHeb") or "").strip()
            return (v or "").strip() if isinstance(v, str) else ""

        hood    = _txt_field(addr, "area") or _txt_field(addr, "neighborhood")
        street  = _txt_field(addr, "street")
        hn      = house.get("number")
        house_n = str(int(hn)) if hn is not None else ""
        floor_v = house.get("floor")
        area    = add_d.get("squareMeter") or add_d.get("squareMeters")
        rooms   = add_d.get("roomsCount")  or add_d.get("rooms")
        item_id = str(item.get("token") or item.get("orderId") or item.get("id") or "")

        # Extract coordinates — try several known Yad2 JSON paths
        _coords = (
            addr.get("coords") or addr.get("coordinate") or
            item.get("coordinate") or item.get("geoLocation") or
            item.get("location") or {}
        )
        _lat_v = _coords.get("lat") or _coords.get("latitude")
        _lon_v = _coords.get("lon") or _coords.get("lng") or _coords.get("longitude")
        try:
            _lat_f = float(_lat_v) if _lat_v is not None else np.nan
            _lon_f = float(_lon_v) if _lon_v is not None else np.nan
            # Sanity-check: must be within Israel bounding box
            if not (29.0 <= _lat_f <= 34.0 and 34.0 <= _lon_f <= 36.5):
                _lat_f = _lon_f = np.nan
        except (TypeError, ValueError):
            _lat_f = _lon_f = np.nan

        rows.append({
            "שכונה":           hood,
            "רחוב":            street,
            "מס' בית":         house_n,
            'שטח (מ"ר)':       float(area)    if area    is not None else np.nan,
            "חדרים":           float(rooms)   if rooms   is not None else np.nan,
            "קומה":            float(floor_v) if floor_v is not None else np.nan,
            "מחיר מבוקש (₪)": int(price),
            "_yad2_id":        item_id,
            "_lat":            _lat_f,
            "_lon":            _lon_f,
        })

    if not rows:
        return pd.DataFrame(), "לא נמצאו מודעות עם פרטים מלאים."

    return pd.DataFrame(rows), ""


# ─── POI / Map helpers ────────────────────────────────────────────────────────

_CAT_COLORS = {
    "transport":  [255, 140,   0, 160],
    "education":  [138,  43, 226, 160],
    "health":     [  0, 180,  60, 160],
    "park":       [ 34, 139,  34, 160],
    "retail":     [220, 180,   0, 160],
    "food":       [220,  20,  60, 160],
    "service":    [ 70, 130, 180, 160],
    "leisure":    [255, 105, 180, 160],
    "tourism":    [  0, 190, 200, 160],
    "community":  [200, 120,  30, 160],
    "nature":     [ 34, 100,  34, 160],
    "historic":   [139,  90,  43, 160],
    "employment": [110, 110, 110, 160],
}

_CAT_HEB = {
    "transport": "תחבורה", "education": "חינוך", "health": "בריאות",
    "park": "פארקים", "retail": "קניות", "food": "מזון",
    "service": "שירותים", "leisure": "פנאי", "tourism": "תיירות",
    "community": "קהילה", "nature": "טבע", "historic": "היסטוריה",
    "employment": "תעסוקה",
}

_CAT_EMOJI = {
    "transport": "🚌", "education": "🎓", "health": "🏥",
    "park": "🌳", "retail": "🛒", "food": "🍽️",
    "service": "🏛️", "leisure": "🎭", "tourism": "🏛️",
    "community": "⛪", "nature": "🌿", "historic": "🏰",
    "employment": "🏢",
}

_FOLIUM_COLORS = {
    "transport": "#FF8C00", "education": "#8A2BE2",
    "health":    "#00B43C", "park":      "#228B22",
    "retail":    "#DAA520", "food":      "#DC143C",
    "service":   "#4682B4", "leisure":   "#FF69B4",
    "tourism":   "#00CED1", "community": "#FFA500",
    "nature":    "#2E6B2E", "historic":  "#8B5A2B",
    "employment": "#696969",
}


def itm_to_wgs84(x_ser: pd.Series, y_ser: pd.Series):
    """Convert ITM (EPSG:2039) easting/northing to WGS84 lat/lon arrays."""
    a   = 6_378_137.0
    f   = 1.0 / 298.257_222_101
    e2  = 2*f - f**2
    k0  = 1.000_006_7
    lam0 = np.radians(35.204_516_944)
    phi0 = np.radians(31.734_393_611)
    FE, FN = 219_529.584, 626_907.390

    x0 = x_ser.values - FE
    y0 = y_ser.values - FN

    A0 = 1 - e2/4 - 3*e2**2/64 - 5*e2**3/256
    B0 = 3/8   * (e2 + e2**2/4  + 15*e2**3/128)
    C0 = 15/256 * (e2**2 + 3*e2**3/4)
    D0 = 35*e2**3/3072
    M0 = a * (A0*phi0 - B0*np.sin(2*phi0) + C0*np.sin(4*phi0) - D0*np.sin(6*phi0))

    M1  = M0 + y0 / k0
    mu1 = M1 / (a * A0)
    e1  = (1 - np.sqrt(1-e2)) / (1 + np.sqrt(1-e2))

    phi1 = (mu1
            + (3*e1/2   - 27*e1**3/32) * np.sin(2*mu1)
            + (21*e1**2/16)             * np.sin(4*mu1)
            + (151*e1**3/96)            * np.sin(6*mu1))

    N1 = a / np.sqrt(1 - e2 * np.sin(phi1)**2)
    T1 = np.tan(phi1)**2
    C1 = e2 * np.cos(phi1)**2 / (1 - e2)
    R1 = a * (1-e2) / (1 - e2*np.sin(phi1)**2)**1.5
    D1 = x0 / (N1 * k0)

    lat = phi1 - (N1*np.tan(phi1)/R1) * (
        D1**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2/(1-e2)) * D1**4/24
    )
    lon = lam0 + (D1 - (1 + 2*T1 + C1)*D1**3/6) / np.cos(phi1)
    return np.degrees(lat), np.degrees(lon)


@st.cache_data
def _get_df_with_latlon() -> pd.DataFrame:
    """Returns the full predictions DataFrame enriched with _lat/_lon columns (WGS84)."""
    df = compute_predictions()
    valid = df.dropna(subset=["X", "Y"])
    lats, lons = itm_to_wgs84(valid["X"], valid["Y"])
    df = df.copy()
    df["_lat"] = np.nan
    df["_lon"] = np.nan
    df.loc[valid.index, "_lat"] = lats
    df.loc[valid.index, "_lon"] = lons
    return df[df["_lat"].between(29, 34) & df["_lon"].between(34, 36)].copy()


def _pts_in_polygon(lats: np.ndarray, lons: np.ndarray, polygon_lonlat: list) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon. polygon_lonlat is a GeoJSON ring: [[lon,lat],...]."""
    py = np.array([p[1] for p in polygon_lonlat], dtype=float)
    px = np.array([p[0] for p in polygon_lonlat], dtype=float)
    n = len(py)
    inside = np.zeros(len(lats), dtype=bool)
    j = n - 1
    for i in range(n):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        cross = ((yi > lats) != (yj > lats)) & (
            lons < (xj - xi) * (lats - yi) / np.where(yj != yi, yj - yi, 1e-15) + xi
        )
        inside ^= cross
        j = i
    return inside


@st.cache_data
def load_poi_data(poi_path: str) -> pd.DataFrame:
    df = pd.read_csv(poi_path, encoding="utf-8-sig", low_memory=False,
                     usecols=["lat", "lon", "name", "category"])
    return df.dropna(subset=["lat", "lon", "category"]).reset_index(drop=True)


def get_local_pois(poi_df: pd.DataFrame, lat: float, lon: float,
                   radius_m: float = 1000) -> pd.DataFrame:
    """Return rows from poi_df within radius_m metres of (lat, lon), with distance."""
    dlat = radius_m / 111_000
    dlon = radius_m / (111_000 * np.cos(np.radians(lat)))
    nearby = poi_df[
        poi_df["lat"].between(lat - dlat, lat + dlat) &
        poi_df["lon"].between(lon - dlon, lon + dlon)
    ].copy()
    if nearby.empty:
        return nearby
    R = 6_371_000.0
    dlat_r = np.radians(nearby["lat"].values - lat)
    dlon_r = np.radians(nearby["lon"].values - lon)
    a = (np.sin(dlat_r / 2) ** 2
         + np.cos(np.radians(lat)) * np.cos(np.radians(nearby["lat"].values))
         * np.sin(dlon_r / 2) ** 2)
    dist = R * 2 * np.arcsin(np.sqrt(a))
    mask = dist <= radius_m
    nearby = nearby[mask].copy()
    nearby["dist_m"]  = dist[mask].round(0).astype(int)
    nearby["prefix"]  = nearby["category"].apply(lambda c: c.split("_")[0])
    nearby["cat_heb"] = nearby["prefix"].map(_CAT_HEB).fillna(nearby["prefix"])
    nearby["color"]   = nearby["prefix"].apply(
        lambda p: _CAT_COLORS.get(p, [128, 128, 128, 160])
    )
    return nearby.sort_values("dist_m").reset_index(drop=True)


@st.cache_data(ttl=86_400, show_spinner=False)
def geocode_address(query: str) -> tuple:
    """Geocode an Israeli address/city via Nominatim. Results cached 24 h."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{query}, ישראל", "format": "json", "limit": 1,
                    "accept-language": "he"},
            headers={"User-Agent": "AG_RealEstate_Advisor/1.0 (academic)"},
            timeout=6,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None


# ─── Profile defaults ──────────────────────────────────────────────────────────
if "profile_budget" not in st.session_state:
    st.session_state["profile_budget"] = 2_000_000
if "profile_goal" not in st.session_state:
    st.session_state["profile_goal"] = "💵 תשואה שוטפת (השכרה)"
if "_profile_goal_val" not in st.session_state:
    st.session_state["_profile_goal_val"] = st.session_state["profile_goal"]
if "profile_configured" not in st.session_state:
    st.session_state["profile_configured"] = False

# ─── Sidebar navigation ────────────────────────────────────────────────────────

# Apply any pending navigation request BEFORE the radio widget is created
if "_nav_request" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("_nav_request")

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div style="font-size:1.4rem;font-weight:700;color:#1A6B5A;">🏠 נדלניסט</div>
      <div style="font-size:0.72rem;font-weight:400;color:#8A9E98;margin-top:3px;">AI · נדל"ן ישראל</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "ניווט:",
        [
            "🏠 עמוד הבית",
            "👤 הפרופיל שלי",
            "🔍 איפה להשקיע?",
            "🏡 בדוק עסקה ספציפית",
            "📊 חפש עסקאות פעילות",
        ],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("""
    <div dir="rtl" style="
      font-size:0.83rem; color:#4A5E58; text-align:right; line-height:1.7;
      padding: 16px 16px 20px; margin-top: 12px;
      border-top: 1px solid #D4EBE2;
    ">
    המודל אומן על <strong>6,609 עסקאות</strong> נדל"ן אמיתיות בישראל.<br><br>
    הכלי מיועד לסיוע בקבלת החלטות בלבד — אינו תחליף לייעוץ מקצועי.
    </div>
    """, unsafe_allow_html=True)


# ─── Pre-warm caches so page navigation is instant ────────────────────────────
# These run once on first load (silently, show_spinner=False) and return
# cached results on every subsequent rerun — eliminating ghost-render artifacts
# where the previous page's content bleeds through while a heavy function runs.
load_model()
compute_predictions()
compute_area_stats()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 עמוד הבית":

    st.markdown("""
    <div dir="rtl" style="
      background:linear-gradient(135deg,#35BA85 0%,#1A6B5A 100%);
      color:white; padding:44px 40px; border-radius:20px; margin-bottom:28px; text-align:center;
      box-shadow:0 6px 24px rgba(26,107,90,0.22); font-family:'Rubik',sans-serif;
    ">
      <div style="font-size:2.2rem;font-weight:700;letter-spacing:-0.5px;">🏠 נדלניסט</div>
      <div style="margin:12px 0 0;opacity:0.90;font-size:1.05rem;font-weight:400;line-height:1.6;">
        מדלן. לדעת. למצוא השקעה. · מבוסס נתוני רשות המיסים ובינה מלאכותית
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Profile CTA banner ────────────────────────────────────────────────────
    if not st.session_state["profile_configured"]:
        st.markdown("""
        <div dir="rtl" style="
          background:#FFF8E1; border:2px solid #FFC107; border-radius:10px;
          padding:18px 24px; margin-bottom:20px; text-align:right;
        ">
          <b style="font-size:1.05rem;">👤 טרם הגדרת פרופיל משקיע אישי</b><br>
          <span style="font-size:0.92rem; color:#555;">
            הגדר תקציב ומטרת השקעה כדי שהכלי יוכל להתאים את ההמלצות בדיוק עבורך.
          </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👤 הגדר פרופיל עכשיו ←", key="home_goto_profile", type="primary"):
            st.session_state["_nav_request"] = "👤 הפרופיל שלי"
            st.rerun()
        st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
    else:
        _goal_short = "תשואה שוטפת" if "תשואה" in st.session_state.get("profile_goal", "💵 תשואה שוטפת (השכרה)") else "עליית ערך"
        st.success(f"👤 פרופיל פעיל: תקציב **{st.session_state.get('profile_budget', 2_000_000):,} ₪** · מטרה: **{_goal_short}**")

    # ── Tool cards ────────────────────────────────────────────────────────────
    rtl('<h3>🚀 מה תוכל לעשות כאן?</h3>')

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            rtl("""
            <h4>🔍 איפה להשקיע?</h4>
            <span style="display:inline-block;background:#E0F5EC;color:#1A6B5A;
              border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:700;
              margin-bottom:10px;">32+ אזורים מנותחים</span>
            <p style="color:#555; font-size:0.9rem;">מתי: לא יודע איפה לחפש</p>
            <p>קבל רשימת ערים מומלצות לפי תקציב ומטרה — ממוינות לפי ציון כדאיות.</p>
            """)
            if st.button("התחל לחפש ←", key="go_find", use_container_width=True):
                st.session_state["_nav_request"] = "🔍 איפה להשקיע?"
                st.rerun()

    with c2:
        with st.container(border=True):
            rtl("""
            <h4>🏡 בדוק עסקה ספציפית</h4>
            <span style="display:inline-block;background:#E0F5EC;color:#1A6B5A;
              border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:700;
              margin-bottom:10px;">6,609 עסקאות · R²=0.74</span>
            <p style="color:#555; font-size:0.9rem;">מתי: מצאת דירה ורוצה לדעת אם המחיר הוגן</p>
            <p>הכנס פרטי הנכס — המודל יגיד אם המחיר מעל או מתחת לשוק.</p>
            """)
            if st.button("בדוק נכס ←", key="go_check", use_container_width=True):
                st.session_state["_nav_request"] = "🏡 בדוק עסקה ספציפית"
                st.rerun()

    with c3:
        with st.container(border=True):
            rtl("""
            <h4>📊 חפש עסקאות פעילות</h4>
            <span style="display:inline-block;background:#E0F5EC;color:#1A6B5A;
              border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:700;
              margin-bottom:10px;">נתוני יד2 בזמן אמת</span>
            <p style="color:#555; font-size:0.9rem;">מתי: רוצה לראות מה נמכר בפועל בעיר מסוימת</p>
            <p>סנן עסקאות לפי גודל, חדרים ושנה — כולל מחיר חזוי וציון לכל עסקה.</p>
            """)
            if st.button("עיין ביישוב ←", key="go_browse", use_container_width=True):
                st.session_state["_nav_request"] = "📊 חפש עסקאות פעילות"
                st.rerun()

    st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)

    # ── Quick reference legend ─────────────────────────────────────────────────
    with st.expander("📌 מה כל עמוד עושה — טבלת עזר מהירה"):
        rtl("""
        <table style="width:100%; border-collapse:collapse; font-size:0.93rem;">
          <tr style="background:#F5F5F5;">
            <th style="padding:10px; border:1px solid #E0E0E0; text-align:right;">עמוד</th>
            <th style="padding:10px; border:1px solid #E0E0E0; text-align:right;">שאלה שהוא עונה</th>
            <th style="padding:10px; border:1px solid #E0E0E0; text-align:right;">מה מכניסים</th>
            <th style="padding:10px; border:1px solid #E0E0E0; text-align:right;">מה מקבלים</th>
          </tr>
          <tr>
            <td style="padding:10px; border:1px solid #E0E0E0;"><strong>🔍 איפה להשקיע?</strong></td>
            <td style="padding:10px; border:1px solid #E0E0E0;">באיזו עיר כדאי לחפש?</td>
            <td style="padding:10px; border:1px solid #E0E0E0;">תקציב + מטרה (שכ"ד / ערך)</td>
            <td style="padding:10px; border:1px solid #E0E0E0;">רשימת ערים עם ציון כדאיות + גרף</td>
          </tr>
          <tr style="background:#fafafa;">
            <td style="padding:10px; border:1px solid #E0E0E0;"><strong>🏡 בדוק עסקה</strong></td>
            <td style="padding:10px; border:1px solid #E0E0E0;">האם המחיר של הדירה הזו הוגן?</td>
            <td style="padding:10px; border:1px solid #E0E0E0;">עיר + מחיר + שטח + חדרים + קומה</td>
            <td style="padding:10px; border:1px solid #E0E0E0;">ציון 0–100 + פירוט + עסקאות דומות</td>
          </tr>
          <tr>
            <td style="padding:10px; border:1px solid #E0E0E0;"><strong>📊 חפש עסקאות</strong></td>
            <td style="padding:10px; border:1px solid #E0E0E0;">בכמה נמכרו דירות בעיר הזו?</td>
            <td style="padding:10px; border:1px solid #E0E0E0;">עיר + פילטרים (גודל / שנה / חדרים)</td>
            <td style="padding:10px; border:1px solid #E0E0E0;">טבלת עסקאות אמיתיות + גרף מחירים</td>
          </tr>
        </table>
        """)

    st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
    rtl('<h2>📚 מילון מושגים — הסבר על מונחי נדל"ן</h2>')
    rtl('<p style="color:#555">לחץ על כל מושג כדי לקרוא הסבר פשוט</p>')

    with st.expander("🎯 ציון כדאיות — מה זה ולמה חשוב?", expanded=True):
        rtl("""
        <p><strong>ציון כדאיות</strong> הוא מספר בין <strong>0 ל-10</strong> שמסכם
        <strong>כמה כדאי לקנות נכס מסוים</strong>.</p>
        <table style="width:100%; border-collapse:collapse; margin:10px 0;">
          <tr style="background:#F5F5F5;">
            <th style="padding:8px; border:1px solid #E0E0E0; text-align:right;">ציון</th>
            <th style="padding:8px; border:1px solid #E0E0E0; text-align:right;">צבע</th>
            <th style="padding:8px; border:1px solid #E0E0E0; text-align:right;">משמעות</th>
            <th style="padding:8px; border:1px solid #E0E0E0; text-align:right;">מה לעשות?</th>
          </tr>
          <tr>
            <td style="padding:8px; border:1px solid #E0E0E0;">7–10</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">🟢 ירוק</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">כדאי מאוד</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">מומלץ לבחון ברצינות</td>
          </tr>
          <tr style="background:#fafafa;">
            <td style="padding:8px; border:1px solid #E0E0E0;">4.5–7</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">🟡 צהוב</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">סביר, דורש בדיקה</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">בדוק לעומק לפני החלטה</td>
          </tr>
          <tr>
            <td style="padding:8px; border:1px solid #E0E0E0;">0–4.5</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">🔴 אדום</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">לא מומלץ</td>
            <td style="padding:8px; border:1px solid #E0E0E0;">המחיר כנראה גבוה מדי</td>
          </tr>
        </table>
        <p><strong>איך מחשבים את הציון?</strong><br>
        💰 <strong>60%</strong> — האם המחיר הוגן? (הכי חשוב)<br>
        🏙️ <strong>25%</strong> — איכות האזור (מדד סוציו-אקונומי)<br>
        💧 <strong>15%</strong> — קל לקנות ולמכור באזור? (נזילות)</p>
        """)

    with st.expander("💰 מחיר חזוי ופער — מה ההבדל?"):
        rtl("""
        <p><strong>מחיר חזוי</strong> = מה מחיר ה"שוק ההוגן" לפי המודל שלנו.<br>
        המודל למד מ-6,609 עסקאות אמיתיות שנמכרו בישראל.</p>
        <hr style="border-color:#eee; margin:10px 0;">
        <p><strong>פער</strong> = ההפרש בין מה שאתה משלם לבין מה שהמודל חושב שזה שווה:</p>
        <ul>
          <li>✅ <strong>פער חיובי (+)</strong> → אתה משלם <strong>פחות</strong> מהשוק → <strong>הזדמנות!</strong></li>
          <li>⚠️ <strong>פער קרוב לאפס</strong> → מחיר הוגן</li>
          <li>❌ <strong>פער שלילי (−)</strong> → אתה משלם <strong>יותר</strong> מהשוק → מחיר גבוה</li>
        </ul>
        <p><strong>דוגמה:</strong> מחיר חזוי 2,200,000 ₪, מחיר מבוקש 2,000,000 ₪ → פער +10% → עסקה טובה!</p>
        """)

    with st.expander("📈 מגמה שנתית — מה זה אומר?"):
        rtl("""
        <p><strong>מגמה שנתית</strong> = כמה אחוזים השתנו המחירים בעיר הזו בכל שנה (בממוצע).</p>
        <ul>
          <li>📈 <strong>+5% בשנה</strong> → המחירים עלו — טוב אם אתה מחפש שהנכס יעלה בערך</li>
          <li>📉 <strong>−2% בשנה</strong> → המחירים ירדו — כדאי לשאול למה</li>
          <li>➡️ <strong>0%</strong> → מחירים יציבים — פחות סיכון, פחות פוטנציאל</li>
        </ul>
        <p>מגמה מהעבר לא מבטיחה את העתיד, אבל היא אינדיקטור חשוב.</p>
        """)

    with st.expander("🏙️ מדד סוציו-אקונומי — למה זה משנה?"):
        rtl("""
        <p><strong>מדד סוציו-אקונומי</strong> = מדד שמשקף את <strong>רמת החיים</strong> ביישוב:
        הכנסות ממוצעות, רמת השכלה, שיעור תעסוקה ועוד. (מקור: למ"ס)</p>
        <table style="width:100%; border-collapse:collapse; margin:10px 0;">
          <tr style="background:#F5F5F5;">
            <th style="padding:8px; border:1px solid #E0E0E0; text-align:right;">מדד</th>
            <th style="padding:8px; border:1px solid #E0E0E0; text-align:right;">משמעות לנדל"ן</th>
          </tr>
          <tr>
            <td style="padding:8px; border:1px solid #E0E0E0;"><strong>גבוה</strong></td>
            <td style="padding:8px; border:1px solid #E0E0E0;">אזור חזק כלכלית → מחירים יציבים → פחות סיכון</td>
          </tr>
          <tr style="background:#fafafa;">
            <td style="padding:8px; border:1px solid #E0E0E0;"><strong>בינוני</strong></td>
            <td style="padding:8px; border:1px solid #E0E0E0;">איזון בין יציבות לפוטנציאל</td>
          </tr>
          <tr>
            <td style="padding:8px; border:1px solid #E0E0E0;"><strong>נמוך</strong></td>
            <td style="padding:8px; border:1px solid #E0E0E0;">פוטנציאל לעלייה → אבל גם יותר סיכון</td>
          </tr>
        </table>
        """)

    with st.expander("💧 נזילות שוק — מה זה ולמה חשוב?"):
        rtl("""
        <p><strong>נזילות שוק</strong> = כמה קל <strong>לקנות ולמכור</strong> נכסים באזור.</p>
        <ul>
          <li>הרבה עסקאות → שוק נזיל → קל למצוא קונים כשתרצה למכור</li>
          <li>מעט עסקאות → קשה יותר למכור → הכסף יכול להיות "תקוע"</li>
        </ul>
        <p>אם תצטרך למכור בדחיפות, בשוק לא נזיל אולי תצטרך להוריד מחיר משמעותית.</p>
        """)

    with st.expander("🤖 המודל — איך הוא עובד?"):
        rtl("""
        <p><strong>המודל:</strong> XGBoost — סוג של בינה מלאכותית.</p>
        <p><strong>מה הוא למד?</strong> מ-6,609 עסקאות דירות אמיתיות מרשות המיסים:
        שטח, חדרים, קומה, עיר, שכונה, רחוב, מדד סוציו ועוד.</p>
        <p><strong>כמה הוא מדויק?</strong><br>
        R² = 0.741 → מסביר 74% מהשינויים במחיר<br>
        שגיאה ממוצעת: ~607,000 ₪</p>
        <p>המודל טוב לזהות אם נכס <strong>מאוד יקר</strong> או <strong>מאוד זול</strong>,
        אבל לא מדויק לסכומים קטנים. השתמש בו כאינדיקטור.</p>
        """)

    st.info("💡 טיפ: אם אתה חדש בנדל\"ן, התחל עם \"איפה להשקיע?\" — הכנס תקציב וקבל המלצות מיידיות.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 הפרופיל שלי":

    st.session_state["profile_configured"] = True

    rtl('<h1>👤 הפרופיל שלי</h1>')
    rtl('<p style="color:#555;">הגדרות אלו משמשות את כל הכלים באפליקציה. שנה אותן בכל עת.</p>')

    with st.container(border=True):
        _budget_mode = st.radio(
            "💰 שיטת הגדרת תקציב",
            ["תקציב כולל ישיר", "חישוב לפי הון עצמי + מימון בנק"],
            horizontal=True,
            key="profile_budget_mode",
        )

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        if _budget_mode == "תקציב כולל ישיר":
            col_a, col_b = st.columns(2)
            with col_a:
                _direct = st.number_input(
                    "💰 תקציב מקסימלי (₪)",
                    min_value=300_000, max_value=10_000_000,
                    value=st.session_state.get("profile_budget", 2_000_000),
                    step=100_000, format="%d",
                    key="profile_budget_direct",
                    help="המחיר המקסימלי שאתה מוכן לשלם. אזורים שמחירם הממוצע גבוה יותר יסוננו.",
                )
                st.session_state["profile_budget"] = int(_direct)
            with col_b:
                st.radio(
                    "🎯 מה המטרה שלך?",
                    ["💵 תשואה שוטפת (השכרה)", "📈 עליית ערך (מכירה ברווח)"],
                    key="profile_goal",
                    help="תשואה שוטפת = שכ\"ד חודשי. עליית ערך = קנה בזול, מכור ביוקר.",
                )

        else:
            col_a, col_b, col_c = st.columns([5, 3, 5])
            with col_a:
                _equity = st.number_input(
                    "🏦 הון עצמי (₪)",
                    min_value=0, max_value=5_000_000,
                    value=st.session_state.get("profile_equity", 0),
                    step=50_000, format="%d",
                    key="profile_equity",
                    help="הסכום שיש לך להשקיע. ניתן להשאיר 0 אם עדיין לא ידוע.",
                )
            with col_b:
                _fin_pct = st.number_input(
                    "🏛️ מימון בנקאי (%)",
                    min_value=0, max_value=90,
                    value=st.session_state.get("profile_financing_pct", 75),
                    step=5, format="%d",
                    key="profile_financing_pct",
                    help="אחוז מהמחיר הכולל שהבנק מממן. ברירת מחדל: 75%.",
                )
            with col_c:
                if _equity > 0:
                    _computed = int(_equity / (1 - _fin_pct / 100)) if _fin_pct < 100 else _equity
                    _computed = max(300_000, min(10_000_000, _computed))
                    _loan     = _computed - _equity
                    st.session_state["profile_budget"] = _computed
                    st.markdown(
                        f'<div dir="rtl" style="background:#EAF7EE;border:2px solid #1A9E3F;'
                        f'border-radius:10px;padding:14px 16px;text-align:center;margin-top:26px;">'
                        f'<div style="font-size:.8rem;color:#555;">תקציב מחושב</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:#1A9E3F;">₪{_computed:,}</div>'
                        f'<div style="font-size:.75rem;color:#777;">'
                        f'הון ₪{_equity:,} + הלוואה ₪{_loan:,}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div dir="rtl" style="background:#F5F5F5;border:2px dashed #CCC;'
                        'border-radius:10px;padding:14px 16px;text-align:center;margin-top:26px;">'
                        '<div style="font-size:.85rem;color:#AAA;">הזן הון עצמי לחישוב</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            st.radio(
                "🎯 מה המטרה שלך?",
                ["💵 תשואה שוטפת (השכרה)", "📈 עליית ערך (מכירה ברווח)"],
                key="profile_goal",
                help="תשואה שוטפת = שכ\"ד חודשי. עליית ערך = קנה בזול, מכור ביוקר.",
            )

    # persist goal to a non-widget key so it survives page navigation
    st.session_state["_profile_goal_val"] = st.session_state.get("profile_goal", "💵 תשואה שוטפת (השכרה)")

    st.success(
        f"✅ הפרופיל נשמר אוטומטית. "
        f"תקציב: **{st.session_state.get('profile_budget', 2_000_000):,} ₪**"
    )

    st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
    if st.button("🔍 עבור לאיפה להשקיע? ←", key="profile_goto_find", type="primary"):
        st.session_state["_nav_request"] = "🔍 איפה להשקיע?"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FIND AREA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 איפה להשקיע?":

    rtl('<h1>🔍 איפה להשקיע?</h1>')

    # ── Explanation bar ───────────────────────────────────────────────────────
    with st.expander("📖 הסבר על הכלי — לחץ לפתיחה / סגירה", expanded=True):
        rtl("""
        <h4>🗺️ מה הכלי הזה עושה?</h4>
        <p>הכלי עוזר לך למצוא <strong>אזורים מומלצים להשקעה</strong>
        מבלי שתצטרך לדעת נדל"ן מראש.</p>
        <p>הוא סורק אלפי עסקאות אמיתיות מרשות המיסים, ומחשב לכל אזור
        <strong>ציון כדאיות</strong> (0–10) לפי שלושה גורמים: כמה זול נמכרו
        נכסים שם, האם המחירים עולים, וכמה פעיל השוק.</p>
        <p><strong>מה תקבל בסוף?</strong><br>
        רשימה של אזורים ממוינת מהמומלץ ביותר לפחות — עם מחיר ממוצע,
        מגמה שנתית וציון.</p>
        <p style="color:#555;font-size:0.9rem;">💡 כדי לשנות תקציב או מטרת השקעה — עבור ל-<strong>הפרופיל שלי</strong>.</p>
        """)

    # ── Profile summary ───────────────────────────────────────────────────────
    budget = st.session_state.get("profile_budget", 2_000_000)
    goal   = st.session_state.get("_profile_goal_val", st.session_state.get("profile_goal", "💵 תשואה שוטפת (השכרה)"))

    with st.container(border=True):
        _pc1, _pc2 = st.columns([4, 1])
        with _pc1:
            rtl(f"""
            <p style="margin:0;">
              👤 <strong>הפרופיל שלך:</strong> &nbsp;
              💰 תקציב: <strong>{budget:,} ₪</strong> &nbsp;|&nbsp;
              🎯 מטרה: <strong>{goal}</strong>
            </p>
            """)
        with _pc2:
            if st.button("✏️ ערוך", key="find_edit_profile"):
                st.session_state["_nav_request"] = "👤 הפרופיל שלי"
                st.rerun()

    # ── Compute ───────────────────────────────────────────────────────────────
    stats    = compute_area_stats()
    filtered = stats[(stats["avg_price"] <= budget) & (stats["deal_count"] >= 10)].copy()

    if filtered.empty:
        st.warning("לא נמצאו אזורים בתקציב זה עם מספיק נתונים. נסה להגדיל את התקציב.")
    else:
        def _minmax(s: pd.Series) -> pd.Series:
            lo, hi = s.min(), s.max()
            return pd.Series(50.0, index=s.index) if hi == lo else (s - lo) / (hi - lo) * 100

        gap_sc, trend_sc, liq_sc = _minmax(filtered["avg_gap"]), _minmax(filtered["trend_pct_yr"]), _minmax(filtered["deal_count"])

        if "עליית ערך" in goal:
            filtered["score"] = ((0.30 * gap_sc + 0.50 * trend_sc + 0.20 * liq_sc) / 10).round(1)
            weight_desc = "פער מחיר 30% + מגמה 50% + נזילות 20%"
        else:
            filtered["score"] = ((0.60 * gap_sc + 0.20 * trend_sc + 0.20 * liq_sc) / 10).round(1)
            weight_desc = "פער מחיר 60% + מגמה 20% + נזילות 20%"

        filtered = filtered.sort_values("score", ascending=False)
        top, best = filtered.head(5), filtered.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("אזורים בתקציב",    len(filtered),
                  help="כמה אזורים נמצאו בתוך התקציב שהגדרת (עם לפחות 10 עסקאות בנתונים).")
        m2.metric("האזור המוביל",     best["settlementNameHeb"],
                  help="האזור עם ציון הכדאיות הגבוה ביותר עבורך.")
        m3.metric("ציון מוביל",       f"{best['score']:.1f} / 10",
                  help="ציון בין 0–10. ירוק = מומלץ מאוד, אדום = פחות מומלץ.")
        m4.metric("מחיר ממוצע מוביל", f"{best['avg_price']:,.0f} ₪",
                  help="מחיר ממוצע של דירות באזור המוביל לפי עסקאות אמיתיות.")

        st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div dir="rtl" style="margin:0.5rem 0 0.25rem 0;">'
            '<span style="font-size:1.4rem;font-weight:700;line-height:1.3;">🏆 האזורים המומלצים ביותר עבורך</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        show_future = "עליית ערך" in goal
        _years = 5
        if show_future:
            _years = st.slider(
                "📅 טווח זמן לחיזוי ערך עתידי (שנים)",
                min_value=1, max_value=20, value=5, step=1,
                key="future_years_slider",
                help="שנה את מספר השנים לחישוב הערך העתידי של הנכס לפי מגמת המחירים ההיסטורית.",
            )

        rows = []
        for _, row in top.iterrows():
            _yield_r = max(0.03, min(0.055, 0.04 - float(row["avg_socio"]) * 0.003))
            _rent    = int(row["avg_price"] * _yield_r / 12)
            _trend_c = max(-0.15, min(0.15, float(row["trend_pct_yr"]) / 100))
            _entry   = {
                "יישוב":            row["settlementNameHeb"],
                "ציון כדאיות":      row["score"],
                "מחיר ממוצע (₪)":  int(row["avg_price"]),
                "פער ממחיר שוק":   round(float(row["avg_gap"]), 1),
                "מגמת מחירים/שנה": round(float(row["trend_pct_yr"]), 1),
                "כמות עסקאות":     int(row["deal_count"]),
                "מדד סוציו":       round(row["avg_socio"], 2),
                "שכירות חודשית":   _rent,
            }
            if show_future:
                # Blend historical trend toward 4% long-term average based on horizon.
                # At year 1 the historical rate has ~90% weight; at year 10+ only 4% applies.
                _long_avg  = 0.04
                _blend     = max(0.0, 1.0 - _years / 10.0)
                _eff_rate  = _trend_c * _blend + _long_avg * (1.0 - _blend)
                _entry["ערך עתידי"] = int(row["avg_price"] * (1 + _eff_rate) ** _years)
            rows.append(_entry)
        df_show = pd.DataFrame(rows)

        def _sc_color(s: float) -> str:
            t = max(0.0, min(1.0, s / 10.0))
            if t <= 0.5:
                f = t * 2
                r, g, b = int(217 + f * 28), int(83 + f * 83), int(79 - f * 44)
            else:
                f = (t - 0.5) * 2
                r, g, b = int(245 - f * 219), int(166 - f * 8), int(35 + f * 28)
            return f"#{r:02x}{g:02x}{b:02x}"

        def _mk_badge(tip: str) -> str:
            return (
                '<span style="display:inline-flex;align-items:center;justify-content:center;'
                'width:14px;height:14px;min-width:14px;border-radius:50%;flex-shrink:0;'
                'background:#9e9e9e;color:#fff;font-size:10px;font-weight:700;cursor:help;'
                f'margin-right:4px;" title="{tip}">?</span>'
            )

        _trs = []
        for _, _r in df_show.iterrows():
            _sc  = _r["ציון כדאיות"]
            _cl  = _sc_color(_sc)
            _pct = _sc / 10 * 100
            _gap = _r["פער ממחיר שוק"]
            _trd = _r["מגמת מחירים/שנה"]
            _gc  = "#1A9E3F" if _gap >= 0 else "#D9534F"
            _tc  = "#1A9E3F" if _trd >= 0 else "#D9534F"
            _row = (
                f'<tr style="border-bottom:1px solid #f0f0f0;">'
                f'<td style="padding:8px 12px;text-align:right;font-weight:500;">{_r["יישוב"]}</td>'
                f'<td style="padding:8px 12px;min-width:140px;">'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="flex:1;background:#eee;border-radius:4px;height:8px;overflow:hidden;">'
                f'<div style="width:{_pct:.0f}%;background:{_cl};height:100%;border-radius:4px;"></div></div>'
                f'<span style="font-size:.8rem;font-weight:700;color:{_cl};">{_sc:.1f}</span>'
                f'</div></td>'
                f'<td style="padding:8px 12px;text-align:right;direction:ltr;">₪{int(_r["מחיר ממוצע (₪)"]):,}</td>'
                f'<td style="padding:8px 12px;text-align:center;color:{_tc};">{_trd:+.1f}%</td>'
                f'<td style="padding:8px 12px;text-align:center;">{_r["מדד סוציו"]:.2f}</td>'
                f'<td style="padding:8px 12px;text-align:right;direction:ltr;">₪{int(_r["שכירות חודשית"]):,}</td>'
            )
            if show_future:
                _row += f'<td style="padding:8px 12px;text-align:right;direction:ltr;">₪{int(_r["ערך עתידי"]):,}</td>'
            _row += '</tr>'
            _trs.append(_row)

        _th = "padding:10px 12px;font-weight:600;color:#555;font-size:.8rem;border-bottom:2px solid #e0e0e0;white-space:nowrap;background:#fafafa;"
        _b_score  = _mk_badge("ציון מ-0 עד 10 לפי שלושה גורמים: פער ממחיר שוק, מגמת מחירים ונזילות שוק. ירוק = מומלץ להשקעה, אדום = פחות מומלץ.")
        _b_price  = _mk_badge("ממוצע מחירי הדירות שנרשמו בפועל ברשות המיסים ביישוב זה.")
        _b_gap    = _mk_badge("ההפרש בין מחיר השוק החזוי על-ידי המודל לבין המחיר שנמכר בפועל. חיובי (+) = נמכר מתחת לשווי — הזדמנות! שלילי (−) = נמכר מעל השוק.")
        _b_trend  = _mk_badge("שינוי אחוזי ממוצע במחירים מדי שנה לפי עסקאות היסטוריות. חיובי = מחירים עולים. שלילי = מחירים יורדים.")
        _b_count  = _mk_badge("מספר העסקאות שנרשמו ביישוב בנתונים. יותר עסקאות = ניתוח אמין יותר.")
        _b_socio  = _mk_badge("מדד הלמ'ס לרמת החיים ביישוב (הכנסות, השכלה, תעסוקה). גבוה = יציבות מחירים וסיכון נמוך. נמוך = פוטנציאל עלייה גבוה יותר.")
        _b_rent   = _mk_badge("הערכת שכירות חודשית לפי תשואה שנתית משתנה לפי רמת האזור (3%–5.5%): אזורים חזקים כלכלית = תשואה נמוכה יותר, אזורים חלשים = תשואה גבוהה יותר. מוחל על המחיר הממוצע שבטבלה. אינה מחליפה הערכה מקצועית.")
        _b_future = _mk_badge(f"הערכת שווי הנכס לאחר {_years} שנים. שיעור הצמיחה מוגדר כשילוב של המגמה ההיסטורית (מוגבל ±15%/שנה) וממוצע ארוך-טווח של 4%/שנה. ככל שהאופק ארוך יותר, המגמה ההיסטורית מקבלת פחות משקל. ב-10 שנים ומעלה — 4% בלבד.")
        _tbl = (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;font-size:.875rem;">'
            '<thead><tr>'
            f'<th style="{_th}text-align:right;">יישוב</th>'
            f'<th style="{_th}text-align:right;min-width:150px;">ציון כדאיות (0-10) {_b_score}</th>'
            f'<th style="{_th}text-align:right;">מחיר ממוצע (₪) {_b_price}</th>'
            f'<th style="{_th}text-align:center;">מגמת מחירים/שנה (%) {_b_trend}</th>'
            f'<th style="{_th}text-align:center;">מדד סוציו-אקונומי {_b_socio}</th>'
            f'<th style="{_th}text-align:right;">שכירות חודשית צפויה {_b_rent}</th>'
        )
        if show_future:
            _tbl += f'<th style="{_th}text-align:right;">ערך עתידי ({_years} שנ׳) {_b_future}</th>'
        _tbl += (
            '</tr></thead>'
            f'<tbody>{"".join(_trs)}</tbody>'
            '</table></div>'
        )

        st.markdown(_tbl, unsafe_allow_html=True)

        with st.expander("📖 איך לקרוא את הטבלה?"):
            rtl("""
            <table style="width:100%;border-collapse:collapse;">
              <tr style="background:#F5F5F5;">
                <th style="padding:8px;border:1px solid #E0E0E0;text-align:right;">עמודה</th>
                <th style="padding:8px;border:1px solid #E0E0E0;text-align:right;">הסבר</th>
              </tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>ציון כדאיות</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">0–10. ירוק = מומלץ, אדום = פחות</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מחיר ממוצע</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">ממוצע עסקאות בפועל באזור</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>פער ממחיר שוק (%)</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">חיובי (+) = נמכר מתחת לשוק = הזדמנות · שלילי (−) = נמכר מעל השוק</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מגמת מחירים/שנה (%)</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">כמה % עלו / ירדו המחירים בשנה בממוצע</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>כמות עסקאות</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">יותר עסקאות = נתון אמין יותר</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מדד סוציו-אקונומי</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">מדד למ"ס לרמת החיים ביישוב. גבוה = אזור חזק → יציבות. נמוך = פוטנציאל עלייה + סיכון</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>שכירות חודשית צפויה</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">הערכה לפי תשואת שכירות אופיינית (3%–5.5% בשנה לפי רמת האזור). לדוגמה: דירה ב-2,000,000 ₪ עם תשואה 4% ≈ 6,700 ₪/חודש.</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>ערך עתידי (5 שנ׳)</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">הערכת שווי הנכס לאחר 5 שנים לפי מגמת המחירים ההיסטורית. מגמה חיובית = ציפייה לעלייה. מוגבל ל-±15%/שנה.</td></tr>
            </table>
            """)

        rtl(f'<p style="color:#555;font-size:0.85rem;">ציון כדאיות חושב לפי: {weight_desc}</p>')

        rtl('<h3>📊 השוואה ויזואלית — ציון כדאיות לפי אזור</h3>')
        fig = px.bar(
            df_show.head(12), x="יישוב", y="ציון כדאיות",
            color="ציון כדאיות", color_continuous_scale=[[0,"#E8334A"],[0.5,"#F5A623"],[1.0,"#35BA85"]],
            range_color=[0, 10], height=370, text="ציון כדאיות",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30, margin=dict(t=20, b=60))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📈 גרף — מחיר ממוצע מול מגמת מחירים"):
            fig2 = px.scatter(
                filtered.head(30), x="avg_price", y="trend_pct_yr",
                size="deal_count", color="score",
                color_continuous_scale=[[0,"#E8334A"],[0.5,"#F5A623"],[1.0,"#35BA85"]], range_color=[0, 10],
                hover_name="settlementNameHeb",
                labels={"avg_price": "מחיר ממוצע (₪)", "trend_pct_yr": "מגמה (%/שנה)",
                        "deal_count": "עסקאות", "score": "ציון כדאיות"},
                title="מחיר מול מגמה — גודל הנקודה = כמות עסקאות", height=400,
            )
            fig2.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
            rtl('<p style="color:#555;font-size:0.85rem;">ציר X = מחיר ממוצע · ציר Y = מגמה שנתית · גודל = נזילות · צבע = ציון כדאיות</p>')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHECK PROPERTY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏡 בדוק עסקה ספציפית":

    rtl('<h1>🏡 בדוק עסקה ספציפית</h1>')

    # ── Explanation bar ───────────────────────────────────────────────────────
    with st.expander("📖 הסבר על הכלי — לחץ לפתיחה / סגירה", expanded=True):
        about_col, inputs_col = st.columns(2)

        with about_col:
            rtl("""
            <h4>🏡 מה הכלי הזה עושה?</h4>
            <p>הכלי בודק אם <strong>מחיר נכס ספציפי</strong> שמצאת הוא הוגן —
            או שיקר / זול מהשוק.</p>
            <p>לאחר שתכניס את הפרטים, המודל יחפש עסקאות דומות שכבר נמכרו
            בעיר הזו, ויחשב מה הנכס "שווה" לפי השוק. לאחר מכן הוא ישווה את
            הסכום הזה למחיר שנדרש ממך — וייתן לך:</p>
            <ul>
              <li><strong>ציון כדאיות</strong> (0–100) — האם כדאי לקנות?</li>
              <li><strong>פירוט</strong> — למה הנכס קיבל את הציון הזה</li>
              <li><strong>עסקאות דומות</strong> — מה שילמו אחרים על נכסים דומים</li>
            </ul>
            """)

        with inputs_col:
            rtl("""
            <h4>📥 מה למלא?</h4>
            <p>🔹 <strong>עיר / יישוב</strong><br>
            העיר שבה הנכס נמצא. הכלי ישווה לעסקאות מאותה עיר בלבד.</p>
            <p>🔹 <strong>מחיר מבוקש (₪)</strong><br>
            הסכום שהמוכר דורש. זה הסכום שנשווה מול מחיר השוק.</p>
            <p>🔹 <strong>שטח (מ"ר)</strong><br>
            גודל הדירה במטרים רבועים כפי שמופיע במודעה.
            דירת 3 חדרים ממוצעת = 70–90 מ"ר.</p>
            <p>🔹 <strong>מספר חדרים</strong><br>
            בישראל נהוג לספור גם את הסלון.
            דירת 3 חדרים = 2 חדרי שינה + סלון. ניתן להכניס חצי חדר (לדוגמה: 2.5).</p>
            <p>🔹 <strong>קומה</strong><br>
            קומת קרקע = 0, קומה ראשונה = 1 וכן הלאה.
            קומות גבוהות נוטות להיות יקרות יותר — המודל לוקח זאת בחשבון.</p>
            """)

    mdl         = load_model()
    df_ml, df_d = load_data()
    feat_cols   = [c for c in df_ml.columns if c != "dealAmount"]
    settlements = sorted(df_d["settlementNameHeb"].dropna().unique().tolist())

    # ── URL auto-fill ─────────────────────────────────────────────────────────
    _FIELD_LABELS = {
        "price":     "💰 מחיר",
        "city":      "🏙️ עיר",
        "street":    "📍 רחוב",
        "house_num": "🔢 מס׳",
        "area":      "📐 שטח",
        "rooms":     "🛏️ חדרים",
        "floor":     "🏢 קומה",
    }

    with st.container(border=True):
        rtl('<h4>📎 יבוא אוטומטי מקישור — יד2</h4>')
        rtl('<p style="color:#555;font-size:0.88rem;">הדבק קישור למודעה ולחץ "חילוץ מידע" — הכלי ינסה למלא את הפרטים אוטומטית. אם לא הצליח, מלא ידנית בטופס למטה.</p>')

        col_url, col_btn = st.columns([5, 1])
        with col_url:
            paste_url = st.text_input(
                "🔗 קישור למודעה",
                key="cp_url",
                placeholder="https://www.yad2.co.il/item/...",
                label_visibility="collapsed",
            )
        with col_btn:
            st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
            fetch_btn = st.button("📥 חילוץ מידע", key="cp_fetch_btn", use_container_width=True, type="primary")

        if fetch_btn:
            if not paste_url or not paste_url.strip():
                st.warning("הדבק קישור תחילה.")
            else:
                _inp = paste_url.strip()
                _is_html = _inp.startswith("<!") or _inp.lower().startswith("<html")
                if _is_html:
                    result = _parse_yad2_html(_inp)
                    # Restore canonical URL from og:url if present
                    _og = re.search(r'property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']', _inp) \
                       or re.search(r'content=["\']([^"\']+)["\'][^>]+property=["\']og:url["\']', _inp)
                    if _og:
                        st.session_state["cp_url"] = _og.group(1)
                else:
                    with st.spinner("מחלץ נתונים מהקישור..."):
                        result = scrape_listing(_inp)

                # Apply whatever fields were found (works for full scrape AND partial HTML)
                filled = []
                if result.get("price"):
                    st.session_state["cp_price"] = max(100_000, min(20_000_000, int(result["price"])))
                    filled.append("price")
                if result.get("area"):
                    st.session_state["cp_area"]  = max(20, min(500, int(result["area"])))
                    filled.append("area")
                if result.get("rooms"):
                    st.session_state["cp_rooms"] = max(1.0, min(10.0, float(result["rooms"])))
                    filled.append("rooms")
                if result.get("floor") is not None:
                    st.session_state["cp_floor"] = max(0, min(50, int(result["floor"])))
                    filled.append("floor")
                if result.get("city"):
                    matched = _match_settlement(result["city"], settlements)
                    if matched:
                        st.session_state["cp_city"] = matched
                        filled.append("city")
                if result.get("lat") is not None:
                    st.session_state["cp_lat"] = result["lat"]
                if result.get("lon") is not None:
                    st.session_state["cp_lon"] = result["lon"]
                if result.get("street"):
                    st.session_state["cp_street"]       = result["street"]
                    st.session_state["cp_street_input"] = result["street"]
                    filled.append("street")
                if result.get("house_num"):
                    st.session_state["cp_house_num"] = str(result["house_num"])
                    filled.append("house_num")

                if filled:
                    st.session_state["cp_auto_fields"] = filled
                    if result.get("error"):
                        # Partial extraction — show what was found + what's missing
                        missing = [_FIELD_LABELS[f] for f in _FIELD_LABELS if f not in filled]
                        st.warning(f"⚠️ חולץ חלקית: {', '.join(_FIELD_LABELS[f] for f in filled if f in _FIELD_LABELS)}"
                                   + (f" | חסר: {', '.join(missing)} — השלם ידנית." if missing else ""))
                    st.rerun()
                else:
                    # Nothing found at all
                    if result.get("needs_paste"):
                        st.warning(f"⚠️ {result['error']} — הדבק את תיאור המודעה ידנית בטופס למטה.")
                    elif result.get("needs_manual"):
                        st.warning(
                            f"⚠️ {result.get('error','')}  \n"
                            "**פתרון:** פתח המודעה בדפדפן → **Ctrl+U** → **Ctrl+A** → **Ctrl+C** → הדבק כאן ולחץ שוב."
                        )
                    else:
                        st.error(f"❌ {result.get('error','שגיאה לא ידועה')}")

        # ── Auto-fill checklist ───────────────────────────────────────────────
        auto_fields = st.session_state.get("cp_auto_fields", [])
        if auto_fields:
            n_total  = len(_FIELD_LABELS)
            n_filled = sum(1 for f in _FIELD_LABELS if f in auto_fields)
            bar_color = "#1A9E3F" if n_filled == n_total else ("#F5A623" if n_filled >= 3 else "#D9534F")
            items_html = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:4px;margin:3px 6px;font-size:0.88rem;">'
                f'{"✅" if f in auto_fields else "❌"} '
                f'<span style="color:{"#1A9E3F" if f in auto_fields else "#D9534F"};font-weight:{"600" if f in auto_fields else "400"};">'
                f'{lbl}</span></span>'
                for f, lbl in _FIELD_LABELS.items()
            )
            st.markdown(
                f'<div dir="rtl" style="background:#F8F9FA;border:1px solid #E0E0E0;border-radius:8px;padding:10px 14px;margin:8px 0;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'<strong style="font-size:0.92rem;">אחזור אוטומטי:</strong>'
                f'<span style="background:{bar_color};color:white;border-radius:20px;padding:2px 10px;font-size:0.85rem;font-weight:700;">'
                f'{n_filled}/{n_total}</span></div>'
                f'<div style="display:flex;flex-wrap:wrap;">{items_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("🗑️ נקה נתונים שאוחזרו", key="cp_clear_auto"):
                for k in ["cp_auto_fields", "cp_lat", "cp_lon", "cp_street", "cp_street_input"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.container(border=True):
        rtl('<h4>פרטי הנכס</h4>')
        # Row 1 — Address
        r1c1, r1c2, r1c3 = st.columns([2, 2, 1])
        with r1c1:
            city = st.selectbox(
                "🏙️ עיר / יישוב", settlements,
                index=settlements.index("בת ים") if "בת ים" in settlements else 0,
                key="cp_city",
                help="בחר את העיר שבה הנכס נמצא.",
            )
        with r1c2:
            if "cp_street_input" not in st.session_state:
                st.session_state["cp_street_input"] = st.session_state.get("cp_street", "")
            street_input = st.text_input(
                "📍 שם רחוב",
                placeholder="לדוגמה: הרצל",
                key="cp_street_input",
                help="שם הרחוב של הנכס.",
            )
        with r1c3:
            house_num = st.text_input(
                "🔢 מס׳", value=st.session_state.get("cp_house_num", ""),
                placeholder="12",
                key="cp_house_num",
                help="מספר הבית.",
            )
        # Row 2 — Price + Area
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            price = st.number_input(
                "💰 מחיר מבוקש (₪)", min_value=100_000, max_value=20_000_000,
                value=1_500_000, step=50_000, format="%d", key="cp_price",
                help="הסכום שהמוכר דורש. זהו הסכום שנשווה מול מחיר השוק.",
            )
        with r2c2:
            area = st.number_input(
                '📐 שטח (מ"ר)', min_value=20, max_value=500, value=80, step=5, key="cp_area",
                help="שטח הדירה במ\"ר. דירת 3 חדרים ממוצעת = 70–90 מ\"ר.",
            )
        # Row 3 — Rooms + Floor
        r3c1, r3c2, _ = st.columns(3)
        with r3c1:
            rooms = st.number_input(
                "🛏️ מספר חדרים", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="cp_rooms",
                help="בישראל נהוג לספור גם את הסלון. דירת 3 חדרים = 2 שינות + סלון.",
            )
        with r3c2:
            floor = st.number_input(
                "🏢 קומה", min_value=0, max_value=50, value=2, step=1, key="cp_floor",
                help="קומת קרקע = 0. קומה ראשונה = 1. קומות גבוהות = בדרך כלל יקרות יותר.",
            )

    calc_btn = st.button("🔍 בדוק את הנכס", type="primary", use_container_width=True)

    if calc_btn:
        _full_addr = " ".join(filter(None, [street_input.strip(), house_num.strip()]))
        st.session_state["cp_result"] = {
            "city": city, "price": price, "area": area, "rooms": rooms, "floor": floor,
            "lat":    st.session_state.get("cp_lat"),
            "lon":    st.session_state.get("cp_lon"),
            "street": _full_addr or st.session_state.get("cp_street", ""),
        }
        st.session_state.pop("cp_show_map", None)

    if st.session_state.get("cp_result"):
        r     = st.session_state["cp_result"]
        city  = r["city"]
        price = r["price"]
        area  = r["area"]
        rooms = r["rooms"]
        floor = r["floor"]

        mask   = df_d["settlementNameHeb"] == city
        sub_d  = df_d[mask]
        sub_ml = df_ml[mask]

        fvec = sub_ml[feat_cols].median().copy()
        fvec["assetArea"]    = float(area)
        fvec["assetRoomNum"] = float(rooms)
        fvec["floor_num"]    = float(floor)
        today = datetime.date.today()
        fvec["deal_year"]    = float(today.year)
        fvec["deal_month"]   = float(today.month)

        predicted  = float(mdl.predict(fvec[feat_cols].values.reshape(1, -1))[0])
        gap_pct    = (predicted - price) / price * 100
        gap_amount = predicted - price

        avg_area  = float(sub_d["dealAmount"].mean()) if not sub_d.empty else price
        n_deals   = len(sub_d)

        s_min = float(df_d["socio_index_avg"].min())
        s_max = float(df_d["socio_index_avg"].max())
        s_avg = float(sub_d["socio_index_avg"].mean()) if not sub_d.empty else (s_min + s_max) / 2
        socio_score  = (s_avg - s_min) / (s_max - s_min) * 100 if s_max > s_min else 50.0
        price_score  = min(100.0, max(0.0, 50.0 + gap_pct * 2.5))
        liq_score    = min(100.0, n_deals / 50.0 * 100.0)
        viability    = round((0.60 * price_score + 0.25 * socio_score + 0.15 * liq_score) / 10, 1)

        st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
        rtl('<h2>📊 תוצאות הניתוח</h2>')

        if viability >= 7.0:
            verdict_class, verdict_icon, verdict_text = "verdict-good", "🟢", "עסקה טובה!"
            verdict_detail = "המחיר נראה הוגן או אפילו נמוך מהשוק."
        elif viability >= 4.5:
            verdict_class, verdict_icon, verdict_text = "verdict-ok", "🟡", "עסקה סבירה"
            verdict_detail = "המחיר קרוב לשוק. בדוק לעומק לפני החלטה."
        else:
            verdict_class, verdict_icon, verdict_text = "verdict-bad", "🔴", "מחיר גבוה"
            verdict_detail = "המחיר נראה גבוה מהשוק. נסה להתמקח."

        st.markdown(
            f'<div class="{verdict_class}">'
            f'<div style="font-size:3rem;">{verdict_icon} {viability:.1f} / 10</div>'
            f'<div dir="rtl" style="font-size:1.5rem;font-weight:800;margin-top:8px;">{verdict_text}</div>'
            f'<div dir="rtl" style="font-size:1rem;color:#555;margin-top:4px;">{verdict_detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        rtl('<h3>💰 השוואת מחירים</h3>')
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("מחיר חזוי (שוק הוגן)", f"{predicted:,.0f} ₪",
                   help="מה המודל חושב שהנכס שווה לפי עסקאות דומות.")
        mc2.metric("מחיר מבוקש", f"{price:,.0f} ₪",
                   help="הסכום שהמוכר דורש.")
        mc3.metric("הפרש", f"{gap_pct:+.1f}%",
                   delta=f"{gap_amount:+,.0f} ₪",
                   delta_color="normal" if gap_pct >= 0 else "inverse",
                   help="פלוס (+) = הנכס שווה יותר ממה שנדרש = טוב לך!")

        mc4, mc5, mc6 = st.columns(3)
        mc4.metric("מחיר ממוצע בעיר",  f"{avg_area:,.0f} ₪",
                   help="ממוצע כל העסקאות בעיר זו בנתונים.")
        mc5.metric("עסקאות בנתונים",   f"{n_deals}",
                   help="יותר עסקאות = ניתוח אמין יותר.")
        mc6.metric("איכות אזור",        f"{socio_score:.0f} / 100",
                   help="ציון המדד הסוציו-אקונומי (0 נמוך, 100 גבוה).")

        rtl('<h3>🔍 פירוט הציון — למה קיבלת את הציון הזה?</h3>')
        with st.container(border=True):
            reasons = []
            if gap_pct > 15:
                reasons.append(("✅", "מחיר זול משמעותית",
                    f"המודל מעריך שהנכס שווה {gap_pct:.1f}% יותר ממה שנדרש ממך. זו הזדמנות טובה."))
            elif gap_pct > 3:
                reasons.append(("✅", "מחיר מתחת לשוק",
                    f"הנכס מתומחר {gap_pct:.1f}% מתחת להערכת המודל. מרחב מיקוח טוב."))
            elif gap_pct > -5:
                reasons.append(("⚠️", "מחיר בשוק",
                    f"המחיר קרוב לשווי ההוגן (פער {gap_pct:+.1f}%). אין הזדמנות גדולה, אבל גם לא יקר."))
            elif gap_pct > -15:
                reasons.append(("⚠️", "מחיר מעל השוק",
                    f"הנכס מתומחר {abs(gap_pct):.1f}% מעל הערכת המודל. שקול להתמקח."))
            else:
                reasons.append(("❌", "מחיר גבוה מהשוק",
                    f"הנכס מתומחר {abs(gap_pct):.1f}% מעל המודל. ודא שיש סיבה מוצדקת."))

            if socio_score >= 70:
                reasons.append(("✅", f"אזור איכותי — {city}",
                    f"{city} מדורגת גבוה במדד הסוציו-אקונומי. זה מוסיף ביטחון להשקעה."))
            elif socio_score >= 40:
                reasons.append(("⚠️", f"אזור בינוני — {city}",
                    f"{city} במדד סוציו-אקונומי בינוני."))
            else:
                reasons.append(("⚠️", f"אזור עם מדד נמוך — {city}",
                    f"{city} מתחת לממוצע הארצי. סיכון גבוה יותר, אבל גם פוטנציאל עלייה."))

            if n_deals >= 30:
                reasons.append(("✅", "שוק נזיל",
                    f"יש {n_deals} עסקאות — שוק פעיל, קל יותר למכור בעתיד."))
            elif n_deals >= 10:
                reasons.append(("⚠️", "נזילות בינונית",
                    f"יש {n_deals} עסקאות — מספיק לניתוח, אבל לא שוק מאוד פעיל."))
            else:
                reasons.append(("⚠️", "מעט נתונים",
                    f"רק {n_deals} עסקאות. הניתוח פחות אמין — מומלץ לקבל חוות דעת שמאי."))

            for icon, title, desc in reasons:
                rtl(f'<p><strong>{icon} {title}</strong><br>{desc}</p>')

        rtl('<h3>🏘️ עסקאות דומות לאחרונה</h3>')
        rtl(f'<p style="color:#555">עסקאות ב{city} עם שטח דומה (±25%) וחדרים דומים (±0.5)</p>')

        df_all_pred = compute_predictions()
        similar = df_all_pred[
            (df_all_pred["settlementNameHeb"] == city) &
            (df_all_pred["assetArea"].between(area * 0.75, area * 1.25)) &
            (df_all_pred["assetRoomNum"].between(rooms - 0.5, rooms + 0.5))
        ].sort_values("deal_year", ascending=False).head(10)

        if similar.empty:
            st.info("לא נמצאו עסקאות דומות בנתונים. נסה לשנות את הפרמטרים.")
        else:
            avg_similar = similar["dealAmount"].mean()
            diff_vs_avg = price - avg_similar
            direction   = "גבוה" if diff_vs_avg > 0 else "נמוך"
            rtl(f'<p><strong>{len(similar)} עסקאות נמצאו</strong> · מחיר ממוצע: <strong>{avg_similar:,.0f} ₪</strong> · המחיר המבוקש {direction} ב-{abs(diff_vs_avg):,.0f} ₪ מהממוצע</p>')

            show_sim = similar.rename(columns={
                "streetNameHeb": "רחוב", "houseNum": "מס'",
                "assetArea": 'שטח (מ"ר)', "assetRoomNum": "חדרים",
                "floor_num": "קומה", "dealAmount": "מחיר שנמכר (₪)",
                "predicted": "מחיר חזוי (₪)", "viability_score": "ציון כדאיות",
                "deal_year": "שנה",
            }).copy()
            show_sim["מחיר שנמכר (₪)"] = show_sim["מחיר שנמכר (₪)"].round(0).astype(int)
            show_sim["מחיר חזוי (₪)"]  = show_sim["מחיר חזוי (₪)"].round(0).astype(int)
            cols_sim = [c for c in ["רחוב", "מס'", 'שטח (מ"ר)', "חדרים", "קומה",
                                    "מחיר שנמכר (₪)", "מחיר חזוי (₪)", "ציון כדאיות", "שנה"]
                        if c in show_sim.columns]
            st.dataframe(
                show_sim[cols_sim],
                column_config={
                    "ציון כדאיות": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                    "מחיר שנמכר (₪)": st.column_config.NumberColumn(format="₪%,d"),
                    "מחיר חזוי (₪)":  st.column_config.NumberColumn(format="₪%,d"),
                },
                hide_index=True, use_container_width=True,
            )

        # ── Map & POI ─────────────────────────────────────────────────────────
        st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
        _map_open = st.session_state.get("cp_show_map", False)
        _map_label = "🗺️ הסתר מפה ונקודות עניין" if _map_open else "🗺️ הצג מפה ונקודות עניין"
        if st.button(_map_label, key="toggle_map_btn"):
            st.session_state["cp_show_map"] = not _map_open
            st.rerun()

        if st.session_state.get("cp_show_map"):
            has_pin = bool(r.get("lat") and r.get("lon"))
            map_lat = float(r["lat"]) if has_pin else None
            map_lon = float(r["lon"]) if has_pin else None
            geocoded = False
            if not has_pin:
                _gc_q = ", ".join(filter(None, [r.get("street"), r.get("city")]))
                if _gc_q:
                    with st.spinner("מאתר מיקום…"):
                        _gc_lat, _gc_lon = geocode_address(_gc_q)
                    if _gc_lat:
                        map_lat, map_lon = _gc_lat, _gc_lon
                        geocoded = True
            has_map_center = bool(map_lat and map_lon)

            try:
                from streamlit_folium import st_folium
                import folium

                # Similar transactions layer
                sim_df = pd.DataFrame(columns=["lat", "lon", "price", "rooms"])
                sim_sub = sub_d[
                    (sub_d["assetRoomNum"] >= rooms - 0.5) &
                    (sub_d["assetRoomNum"] <= rooms + 0.5)
                ]
                valid_s = sim_sub.dropna(subset=["X", "Y"])
                if len(valid_s):
                    s_lats, s_lons = itm_to_wgs84(valid_s["X"], valid_s["Y"])
                    sim_df = pd.DataFrame({
                        "lat": s_lats, "lon": s_lons,
                        "price": valid_s["dealAmount"].values,
                        "rooms": valid_s["assetRoomNum"].values,
                    })
                    sim_df = sim_df[sim_df["lat"].between(29, 34) & sim_df["lon"].between(34, 36)]

                # Centre + zoom
                if has_map_center:
                    c_lat, c_lon, zoom = map_lat, map_lon, 15 if has_pin else 14
                elif len(sim_df):
                    c_lat, c_lon, zoom = float(sim_df["lat"].mean()), float(sim_df["lon"].mean()), 14
                else:
                    c_lat, c_lon, zoom = 32.0, 34.8, 12

                fmap = folium.Map(location=[c_lat, c_lon], zoom_start=zoom, tiles="OpenStreetMap")

                if len(sim_df):
                    sim_group = folium.FeatureGroup(name=f"נכסים דומים ({len(sim_df)})", show=True)
                    for _, row_m in sim_df.iterrows():
                        folium.CircleMarker(
                            location=[row_m.lat, row_m.lon],
                            radius=6, color="#6495ED", fill=True,
                            fill_color="#6495ED", fill_opacity=0.65, weight=1,
                            tooltip=f"{int(row_m.price):,} ₪ | {row_m.rooms:.1f} חדרים",
                        ).add_to(sim_group)
                    sim_group.add_to(fmap)

                # POI layer
                poi_cache_key = None
                if has_map_center and POI_PATH.exists():
                    poi_cache_key = f"pois_{map_lat:.4f}_{map_lon:.4f}"
                    if poi_cache_key not in st.session_state:
                        all_pois = load_poi_data(str(POI_PATH))
                        st.session_state[poi_cache_key] = get_local_pois(all_pois, map_lat, map_lon)
                    poi_df = st.session_state[poi_cache_key]

                    if len(poi_df):
                        summary = (
                            poi_df.groupby(["prefix", "cat_heb"])
                            .agg(כמות=("dist_m", "count"), קרוב_ביותר=("dist_m", "min"))
                            .reset_index()
                            .sort_values("קרוב_ביותר")
                        )
                        summary["סמל"]    = summary["prefix"].map(_CAT_EMOJI).fillna("📍")
                        summary["קטגוריה"] = summary["סמל"] + " " + summary["cat_heb"]
                        summary["קרוב ביותר (מ')"] = summary["קרוב_ביותר"]
                        with st.expander(f"📊 סיכום נקודות עניין — {len(poi_df)} נקודות ב-1 ק\"מ", expanded=False):
                            st.dataframe(
                                summary[["קטגוריה", "כמות", "קרוב ביותר (מ')"]],
                                hide_index=True, use_container_width=True,
                            )

                        all_prefixes = summary["prefix"].tolist()
                        prefix_labels = {
                            row_s["prefix"]: f"{row_s['סמל']} {row_s['cat_heb']} ({row_s['כמות']})"
                            for _, row_s in summary.iterrows()
                        }
                        selected_prefixes = st.multiselect(
                            "סנן קטגוריות POI",
                            options=all_prefixes,
                            default=all_prefixes,
                            format_func=lambda p: prefix_labels.get(p, p),
                            key="poi_cat_filter_simple",
                        )
                        poi_filtered = poi_df[poi_df["prefix"].isin(selected_prefixes)]

                        for prefix in selected_prefixes:
                            grp_df = poi_filtered[poi_filtered["prefix"] == prefix]
                            clr    = _FOLIUM_COLORS.get(prefix, "#888888")
                            grp    = folium.FeatureGroup(name=f"{prefix_labels[prefix]}", show=True)
                            for _, row_p in grp_df.iterrows():
                                name_str = row_p["name"] if row_p["name"] else row_p["cat_heb"]
                                folium.CircleMarker(
                                    location=[row_p.lat, row_p.lon],
                                    radius=5, color=clr, fill=True,
                                    fill_color=clr, fill_opacity=0.65, weight=1,
                                    tooltip=f"{name_str} | {int(row_p.dist_m)} מ'",
                                ).add_to(grp)
                            grp.add_to(fmap)

                # Property pin
                if has_map_center:
                    addr_parts = [p for p in [r.get("street"), r.get("city")] if p]
                    addr_label = ", ".join(addr_parts) if addr_parts else r.get("city", "הנכס")
                    approx_note = " (מיקום משוער)" if geocoded else ""
                    popup_html = (
                        f"<div dir='rtl' style='font-family:Arial;min-width:180px'>"
                        f"<b>{addr_label}{approx_note}</b><br>"
                        f"{int(r['price']):,} ₪ | {r['rooms']:.1f} חדרים"
                        f"</div>"
                    )
                    folium.Marker(
                        location=[map_lat, map_lon],
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=addr_label + approx_note,
                        icon=folium.Icon(color="orange" if geocoded else "red", icon="home", prefix="fa"),
                    ).add_to(fmap)

                folium.LayerControl(position="topright").add_to(fmap)
                st_folium(fmap, use_container_width=True, height=420, returned_objects=[])

                legend = []
                if has_pin:
                    legend.append("🔴 הנכס (מדויק)")
                elif geocoded:
                    legend.append("🟠 הנכס (מיקום משוער)")
                if len(sim_df):
                    legend.append(f"🔵 נכסים דומים ({len(sim_df)})")
                if poi_cache_key and len(st.session_state.get(poi_cache_key, pd.DataFrame())):
                    legend.append("🟠 תחבורה · 🟣 חינוך · 🟢 בריאות/פארקים · 🟡 קניות · 🔴 מזון")
                if legend:
                    st.caption(" · ".join(legend))

            except Exception as _map_exc:
                st.caption(f"שגיאת מפה: {_map_exc}")
                try:
                    lats, lons = itm_to_wgs84(sub_d["X"].dropna(), sub_d["Y"].dropna())
                    fb = pd.DataFrame({"lat": lats, "lon": lons})
                    fb = fb[fb["lat"].between(29, 34) & fb["lon"].between(34, 36)]
                    if has_pin:
                        fb = pd.concat([pd.DataFrame({"lat": [r["lat"]], "lon": [r["lon"]]}), fb], ignore_index=True)
                    if len(fb):
                        st.map(fb, zoom=14, use_container_width=True)
                except Exception:
                    pass

        if st.button("🔄 בדיקה חדשה", key="reset_cp"):
            for k in ["cp_result", "cp_show_map", "cp_auto_fields",
                      "cp_lat", "cp_lon", "cp_street"]:
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BROWSE CITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 חפש עסקאות פעילות":

    rtl('<h1>📊 חפש עסקאות פעילות</h1>')

    # ── Explanation bar ───────────────────────────────────────────────────────
    with st.expander("📖 הסבר על הכלי — לחץ לפתיחה / סגירה", expanded=True):
        about_col, inputs_col = st.columns(2)

        with about_col:
            rtl("""
            <h4>📊 מה הכלי הזה עושה?</h4>
            <p>הכלי מציג רשימת נכסים לעיר שתבחר —
            מנתונים היסטוריים (<strong>רשות המיסים</strong>) או מודעות
            <strong>בזמן אמת מיד2</strong>.</p>
            <p>כל עסקה/מודעה מקבלת <strong>ציון כדאיות</strong> לפי מודל הבינה המלאכותית שלנו,
            כדי שתוכל להשוות ולמצוא הזדמנויות.</p>
            """)

        with inputs_col:
            rtl("""
            <h4>📥 מה לבחור?</h4>
            <p>🔹 <strong>מקור נתונים</strong><br>
            <em>נתונים היסטוריים</em> — עסקאות שבוצעו בפועל (מחיר שנמכר).<br>
            <em>מחירים בזמן אמת</em> — מודעות פעילות כעת ביד2 (מחיר מבוקש).</p>
            <p>🔹 <strong>בחירת אזור</strong><br>
            עיר בודדת, מחוז שלם, או ציור פוליגון חופשי על המפה.</p>
            <p>🔹 <strong>סינון</strong><br>
            גרור את הסרגלים לפי שטח, חדרים ושנה (בנתונים היסטוריים).</p>
            """)

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "📡 מקור נתונים:",
        ["🔴 מחירים בזמן אמת (יד2)", "📂 נתונים היסטוריים (רשות המיסים)"],
        horizontal=True,
        key="browse_mode",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORICAL MODE
    # ══════════════════════════════════════════════════════════════════════════
    if mode.startswith("📂"):

        df_all = compute_predictions()
        cities = sorted(df_all["settlementNameHeb"].dropna().unique().tolist())

        area_mode = st.radio(
            "🗺️ בחירת אזור:",
            ["🏙️ עיר / יישוב", "🗺️ מחוז", "✏️ פוליגון על מפה"],
            horizontal=True,
            key="browse_area_mode",
        )

        # ── City ──────────────────────────────────────────────────────────────
        if area_mode == "🏙️ עיר / יישוב":
            default_city = "בת ים" if "בת ים" in cities else cities[0]
            selected_city = st.selectbox(
                "🏙️ בחר עיר / יישוב", cities,
                index=cities.index(default_city), key="browse_city",
                help="בחר את העיר שמעניינת אותך לבדיקה.",
            )
            df_city = df_all[df_all["settlementNameHeb"] == selected_city].copy()
            area_label = selected_city

        # ── District ──────────────────────────────────────────────────────────
        elif area_mode == "🗺️ מחוז":
            district = st.selectbox(
                "🗺️ בחר מחוז", list(_DISTRICT_MAP.keys()), key="browse_district",
            )
            district_cities = [c for c in _DISTRICT_MAP[district] if c in cities]
            if not district_cities:
                st.warning(f"לא נמצאו ערים מהמחוז '{district}' בבסיס הנתונים.")
                st.stop()
            df_city = df_all[df_all["settlementNameHeb"].isin(district_cities)].copy()
            area_label = f"{district} ({len(district_cities)} יישובים)"
            st.caption(f"יישובים: {', '.join(sorted(district_cities))}")

        # ── Polygon ───────────────────────────────────────────────────────────
        else:
            try:
                from streamlit_folium import st_folium
                import folium
                from folium.plugins import Draw
            except ImportError:
                st.error("חסרה ספרייה: streamlit-folium. הרץ: pip install streamlit-folium folium")
                st.stop()

            st.info("✏️ צייר פוליגון על המפה כדי לבחור אזור. לחץ על כלי הפוליגון בפאנל השמאלי.")
            fmap_draw = folium.Map(location=[31.8, 34.9], zoom_start=8, tiles="OpenStreetMap")
            Draw(
                draw_options={
                    "polyline": False, "rectangle": True, "polygon": True,
                    "circle": False, "marker": False, "circlemarker": False,
                },
                edit_options={"edit": True, "remove": True},
            ).add_to(fmap_draw)

            draw_out = st_folium(fmap_draw, key="browse_draw_map",
                                 use_container_width=True, height=430)

            polygon_lonlat = None
            if draw_out and draw_out.get("last_active_drawing"):
                geom = draw_out["last_active_drawing"].get("geometry", {})
                if geom.get("type") == "Polygon":
                    polygon_lonlat = geom["coordinates"][0]
                elif geom.get("type") == "Rectangle":
                    polygon_lonlat = geom["coordinates"][0]

            if not polygon_lonlat:
                st.caption("ממתין לציור פוליגון...")
                st.stop()

            df_ll = _get_df_with_latlon()
            mask = _pts_in_polygon(
                df_ll["_lat"].values, df_ll["_lon"].values, polygon_lonlat
            )
            df_city = df_ll[mask].copy()
            area_label = f"פוליגון ({len(df_city)} עסקאות)"
            if df_city.empty:
                st.warning("לא נמצאו עסקאות בתוך הפוליגון. נסה אזור גדול יותר.")
                st.stop()

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("סה\"כ עסקאות",      len(df_city),
                   help="כמה עסקאות יש בבסיס הנתונים עבור עיר זו.")
        sm2.metric("מחיר ממוצע",        f"{df_city['dealAmount'].mean():,.0f} ₪",
                   help="ממוצע מחירי המכירה בפועל — לא מחיר מבוקש!")
        sm3.metric("שטח ממוצע",         f"{df_city['assetArea'].mean():.0f} מ\"ר",
                   help="שטח ממוצע של דירה שנמכרה בעיר.")
        sm4.metric("ציון כדאיות ממוצע", f"{df_city['viability_score'].mean():.1f} / 10",
                   help="מעל 5 = בממוצע נמכרו מתחת לשוק.")

        st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
        rtl('<h3>סינון עסקאות</h3>')

        area_vals  = df_city["assetArea"].dropna()
        rooms_vals = df_city["assetRoomNum"].dropna()
        year_vals  = df_city["deal_year"].dropna()

        # Row 1 — price (₪M min/max) + rooms (min/max)
        hc1, hc2, hc3, hc4 = st.columns(4)
        _prices = df_city["dealAmount"].dropna()
        with hc1:
            h_p_from = st.number_input(
                "💰 מחיר מינימום (₪ מ')",
                min_value=0.0, max_value=200.0,
                value=round(float(_prices.min()) / 1e6, 1) if len(_prices) else 0.0,
                step=0.1, format="%.1f", key="browse_p_from",
            )
        with hc2:
            h_p_to = st.number_input(
                "מחיר מקסימום (₪ מ')",
                min_value=0.0, max_value=200.0,
                value=round(float(_prices.max()) / 1e6 + 0.1, 1) if len(_prices) else 200.0,
                step=0.1, format="%.1f", key="browse_p_to",
            )
        _rv = rooms_vals[rooms_vals.between(1, 10)]
        with hc3:
            h_r_from = st.number_input(
                "🛏️ חדרים מינימום",
                min_value=1.0, max_value=10.0,
                value=float(_rv.min()) if len(_rv) else 1.0,
                step=0.5, format="%.1f", key="browse_r_from",
            )
        with hc4:
            h_r_to = st.number_input(
                "חדרים מקסימום",
                min_value=1.0, max_value=10.0,
                value=float(_rv.max()) if len(_rv) else 10.0,
                step=0.5, format="%.1f", key="browse_r_to",
            )

        # Row 2 — area (min/max) + year (min/max)
        ha1, ha2, ha3, ha4 = st.columns(4)
        _av = area_vals[(area_vals >= 20) & (area_vals <= 600)]
        with ha1:
            h_a_from = st.number_input(
                '📐 שטח מינימום (מ"ר)',
                min_value=0, max_value=1000,
                value=int(_av.min()) if len(_av) else 0,
                step=5, key="browse_a_from",
            )
        with ha2:
            h_a_to = st.number_input(
                'שטח מקסימום (מ"ר)',
                min_value=0, max_value=1000,
                value=int(_av.max()) + 5 if len(_av) else 1000,
                step=5, key="browse_a_to",
            )
        with ha3:
            h_y_from = st.number_input(
                "📅 שנה מינימום",
                min_value=int(year_vals.min()) if len(year_vals) else 2000,
                max_value=int(year_vals.max()) if len(year_vals) else 2030,
                value=int(year_vals.min()) if len(year_vals) else 2000,
                step=1, key="browse_y_from",
            )
        with ha4:
            h_y_to = st.number_input(
                "שנה מקסימום",
                min_value=int(year_vals.min()) if len(year_vals) else 2000,
                max_value=int(year_vals.max()) if len(year_vals) else 2030,
                value=int(year_vals.max()) if len(year_vals) else 2030,
                step=1, key="browse_y_to",
            )

        mask = (
            df_city["dealAmount"].between(h_p_from * 1e6, h_p_to * 1e6) &
            df_city["assetArea"].between(h_a_from, h_a_to) &
            df_city["assetRoomNum"].between(h_r_from, h_r_to) &
            df_city["deal_year"].between(h_y_from, h_y_to)
        )
        df_filtered = df_city[mask].sort_values("viability_score", ascending=False).copy()

        rtl(f'<p><strong>{len(df_filtered)} עסקאות</strong> מוצגות — מדורגות מהכדאית ביותר</p>')

        show = df_filtered.rename(columns={
            "neighborhood": "שכונה", "streetNameHeb": "רחוב", "houseNum": "מס' בית",
            "assetArea": 'שטח (מ"ר)', "assetRoomNum": "חדרים", "floor_num": "קומה",
            "dealAmount": "מחיר שנמכר (₪)", "predicted": "מחיר חזוי (₪)",
            "viability_score": "ציון כדאיות", "deal_year": "שנה",
        }).copy()

        show["מחיר שנמכר (₪)"] = show["מחיר שנמכר (₪)"].round(0).astype(int)
        show["מחיר חזוי (₪)"]  = show["מחיר חזוי (₪)"].round(0).astype(int)
        show['שטח (מ"ר)']       = show['שטח (מ"ר)'].round(1)
        if "קומה" in show.columns:
            try:
                show["קומה"] = show["קומה"].round(0).astype(int)
            except Exception:
                pass

        disp_cols = [c for c in
            ["שכונה", "רחוב", "מס' בית", 'שטח (מ"ר)', "חדרים", "קומה",
             "מחיר שנמכר (₪)", "מחיר חזוי (₪)", "ציון כדאיות", "שנה"]
            if c in show.columns]

        st.dataframe(
            show[disp_cols].head(100),
            column_config={
                "ציון כדאיות": st.column_config.ProgressColumn(
                    "ציון כדאיות (0-10)", min_value=0, max_value=10, format="%.1f",
                ),
                "מחיר שנמכר (₪)": st.column_config.NumberColumn(format="₪%,d"),
                "מחיר חזוי (₪)":  st.column_config.NumberColumn(format="₪%,d"),
            },
            hide_index=True, use_container_width=True,
        )

        with st.expander("📖 איך לקרוא את הטבלה?"):
            rtl("""
            <table style="width:100%;border-collapse:collapse;">
              <tr style="background:#F5F5F5;">
                <th style="padding:8px;border:1px solid #E0E0E0;text-align:right;">עמודה</th>
                <th style="padding:8px;border:1px solid #E0E0E0;text-align:right;">הסבר</th>
              </tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>ציון כדאיות</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">0–100. עמודה ירוקה = נמכר מתחת לשוק</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מחיר שנמכר</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">הסכום שהקונה שילם בפועל (לא מחיר מבוקש)</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מחיר חזוי</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">מה המודל חשב שהנכס שווה</td></tr>
              <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>שנה</strong></td>
                  <td style="padding:8px;border:1px solid #E0E0E0;">מתי בוצעה העסקה</td></tr>
            </table>
            <p>הטבלה ממוינת מציון כדאיות גבוה לנמוך — העסקאות הכי "טובות" מוצגות ראשונות.</p>
            """)

        with st.expander("📊 גרף התפלגות מחירים"):
            if len(df_filtered) >= 5:
                fig3 = px.histogram(
                    df_filtered, x="dealAmount", nbins=30,
                    title=f"התפלגות מחירים — {selected_city}",
                    labels={"dealAmount": "מחיר (₪)", "count": "כמות עסקאות"},
                    height=350, color_discrete_sequence=["#006AFF"],
                )
                fig3.update_layout(margin=dict(t=40, b=20), bargap=0.05)
                st.plotly_chart(fig3, use_container_width=True)
                rtl(f"""
                <p style="color:#555;font-size:0.85rem;">
                מחיר מינימום: {df_filtered['dealAmount'].min():,.0f} ₪ &nbsp;|&nbsp;
                חציון: {df_filtered['dealAmount'].median():,.0f} ₪ &nbsp;|&nbsp;
                מחיר מקסימום: {df_filtered['dealAmount'].max():,.0f} ₪
                </p>
                """)
            else:
                st.info("אין מספיק נתונים להצגת גרף עם הפילטרים הנוכחיים.")

    # ══════════════════════════════════════════════════════════════════════════
    # REAL-TIME YAD2 MODE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        df_all_ref  = compute_predictions()
        _all_cities = df_all_ref["settlementNameHeb"].dropna().unique().tolist()
        # Only show cities that have a confirmed Yad2 city ID
        cities_rt   = sorted(c for c in _all_cities if c in _YAD2_CITY_IDS)
        default_rt  = "בת ים" if "בת ים" in cities_rt else (cities_rt[0] if cities_rt else "")

        rt_area_mode = st.radio(
            "🗺️ בחירת אזור:",
            ["🏙️ עיר / יישוב", "🗺️ מחוז", "✏️ פוליגון על מפה"],
            horizontal=True,
            key="browse_rt_area_mode",
        )

        if rt_area_mode == "🗺️ מחוז":
            district_rt = st.selectbox(
                "🗺️ בחר מחוז", list(_DISTRICT_MAP.keys()), key="browse_rt_district",
            )
            _rt_fetch_cities = sorted(c for c in _DISTRICT_MAP[district_rt] if c in cities_rt)
            if not _rt_fetch_cities:
                st.warning(f"אין ערים נתמכות ביד2 במחוז '{district_rt}'.")
                st.stop()
            st.caption(f"יטען מ-{len(_rt_fetch_cities)} ערים: {', '.join(_rt_fetch_cities)}")
            selected_city_rt = district_rt          # cache key = district name
        elif rt_area_mode == "✏️ פוליגון על מפה":
            try:
                from streamlit_folium import st_folium as _st_folium_sel
                import folium as _folium_sel
                from folium.plugins import Draw as _Draw_sel
            except ImportError:
                st.error("חסרה ספרייה: streamlit-folium. הרץ: pip install streamlit-folium folium")
                st.stop()

            # Build city centroid lookup from historical data
            _df_centroids_src = _get_df_with_latlon()
            _city_centroids_rt = (
                _df_centroids_src[_df_centroids_src["settlementNameHeb"].isin(cities_rt)]
                .groupby("settlementNameHeb")[["_lat", "_lon"]].mean()
                .reset_index()
                .dropna()
            )

            st.caption("✏️ צייר פוליגון או מלבן על המפה — יטענו כל הערים שמרכזן בתוכו.")

            _fmap_sel = _folium_sel.Map(location=[31.8, 34.9], zoom_start=8, tiles="OpenStreetMap")
            for _, _cr in _city_centroids_rt.iterrows():
                _folium_sel.CircleMarker(
                    location=[float(_cr["_lat"]), float(_cr["_lon"])],
                    radius=4, color="#1f77b4", fill=True, fill_opacity=0.7, weight=1,
                    tooltip=_cr["settlementNameHeb"],
                ).add_to(_fmap_sel)
            _Draw_sel(
                draw_options={
                    "polyline": False, "rectangle": True, "polygon": True,
                    "circle": False, "marker": False, "circlemarker": False,
                },
                edit_options={"edit": True, "remove": True},
            ).add_to(_fmap_sel)

            _sel_out = _st_folium_sel(_fmap_sel, key="rt_select_polygon_map",
                                      use_container_width=True, height=380)

            if _sel_out and _sel_out.get("last_active_drawing"):
                _geom_sel = _sel_out["last_active_drawing"].get("geometry", {})
                if _geom_sel.get("type") in ("Polygon", "Rectangle"):
                    _new_sel_poly = _geom_sel["coordinates"][0]
                    if st.session_state.get("_rt_select_polygon") != _new_sel_poly:
                        st.session_state["_rt_select_polygon"] = _new_sel_poly
                        st.session_state.pop("_rt_df", None)
                        st.session_state.pop("_rt_error", None)

            _sel_poly = st.session_state.get("_rt_select_polygon")
            if _sel_poly and not _city_centroids_rt.empty:
                _c_lats = _city_centroids_rt["_lat"].values
                _c_lons = _city_centroids_rt["_lon"].values
                _in_sel = _pts_in_polygon(_c_lats, _c_lons, _sel_poly)
                _rt_fetch_cities = _city_centroids_rt[_in_sel]["settlementNameHeb"].tolist()
                if _rt_fetch_cities:
                    st.caption(f"יטען מ-{len(_rt_fetch_cities)} ערים: {', '.join(_rt_fetch_cities)}")
                    selected_city_rt = "|".join(sorted(_rt_fetch_cities))
                else:
                    st.warning("לא נמצאו ערים בפוליגון. נסה לצייר אזור גדול יותר.")
                    st.stop()
            else:
                st.info("💡 צייר פוליגון להגדרת האזור לטעינה")
                st.stop()
        else:
            selected_city_rt = st.selectbox(
                "🏙️ בחר עיר / יישוב", cities_rt,
                index=cities_rt.index(default_rt) if default_rt in cities_rt else 0,
                key="browse_city_rt",
                help="מציג רק ערים הנתמכות בחיפוש יד2.",
            )
            _rt_fetch_cities = [selected_city_rt]

        # Invalidate cached data when city changes
        if st.session_state.get("_rt_loaded_city") != selected_city_rt:
            st.session_state.pop("_rt_df", None)
            st.session_state.pop("_rt_error", None)

        col_btn, col_hint = st.columns([1, 4])
        with col_btn:
            load_btn = st.button(
                "🔄 טען נתונים מיד2", type="primary",
                use_container_width=True, key="rt_load_btn",
            )
        with col_hint:
            rtl(
                '<p style="color:#696969;font-size:0.85rem;margin-top:8px;">'
                'הנתונים נשלפים בזמן אמת מיד2. '
                'ציון הכדאיות מחושב ע"י מודל הבינה המלאכותית שלנו מול המחיר המבוקש.'
                '</p>'
            )

        if load_btn:
            _fetch_label = selected_city_rt if len(_rt_fetch_cities) == 1 else f"{selected_city_rt} ({len(_rt_fetch_cities)} ערים)"
            with st.spinner(f"טוען מודעות מ{_fetch_label} מיד2..."):
                _frames_rt = []
                rt_err = ""
                for _fc in _rt_fetch_cities:
                    _df_c, _err_c = fetch_yad2_city_listings(_fc)
                    if _err_c and not _frames_rt:
                        rt_err = _err_c   # only fatal if no city succeeded at all
                    elif not _df_c.empty:
                        _df_c["_city"] = _fc
                        _frames_rt.append(_df_c)
                df_rt_raw = pd.concat(_frames_rt, ignore_index=True) if _frames_rt else pd.DataFrame()

            if df_rt_raw.empty:
                st.session_state["_rt_error"] = rt_err or "לא נמצאו מודעות."
                st.session_state.pop("_rt_df", None)
            else:
                # Apply ML model to compute predicted price & viability score
                mdl_rt       = load_model()
                df_ml_rt, _  = load_data()
                feat_cols_rt = [c for c in df_ml_rt.columns if c != "dealAmount"]

                city_mask_rt = df_all_ref["settlementNameHeb"].isin(_rt_fetch_cities)
                sub_ml_rt    = df_ml_rt[city_mask_rt]
                fvec_base_rt = (
                    sub_ml_rt[feat_cols_rt].median()
                    if not sub_ml_rt.empty
                    else df_ml_rt[feat_cols_rt].median()
                )

                today_rt = datetime.date.today()
                preds_rt = []
                for _, row_rt in df_rt_raw.iterrows():
                    fv = fvec_base_rt.copy()
                    if not pd.isna(row_rt.get('שטח (מ"ר)', np.nan)):
                        fv["assetArea"]    = float(row_rt['שטח (מ"ר)'])
                    if not pd.isna(row_rt.get("חדרים", np.nan)):
                        fv["assetRoomNum"] = float(row_rt["חדרים"])
                    if not pd.isna(row_rt.get("קומה", np.nan)):
                        fv["floor_num"]    = float(row_rt["קומה"])
                    fv["deal_year"]  = float(today_rt.year)
                    fv["deal_month"] = float(today_rt.month)
                    try:
                        preds_rt.append(
                            float(mdl_rt.predict(fv[feat_cols_rt].values.reshape(1, -1))[0])
                        )
                    except Exception:
                        preds_rt.append(np.nan)

                df_rt_raw["מחיר חזוי (₪)"] = preds_rt

                def _viab_rt(row):
                    p, pred = row["מחיר מבוקש (₪)"], row["מחיר חזוי (₪)"]
                    if pd.isna(pred) or p <= 0:
                        return 5.0
                    return float(np.clip((50.0 + (pred - p) / p * 100 * 1.5) / 10, 0, 10))

                df_rt_raw["ציון כדאיות"]   = df_rt_raw.apply(_viab_rt, axis=1).round(1)
                df_rt_raw["מחיר חזוי (₪)"] = df_rt_raw["מחיר חזוי (₪)"].round(0)
                df_rt_raw                   = df_rt_raw.sort_values("ציון כדאיות", ascending=False)

                st.session_state["_rt_df"]          = df_rt_raw
                st.session_state["_rt_loaded_city"]  = selected_city_rt
                st.session_state.pop("_rt_error", None)

        # ── Error state ───────────────────────────────────────────────────────
        if st.session_state.get("_rt_error"):
            st.warning(f"⚠️ {st.session_state['_rt_error']}")
            rtl(
                '<p style="color:#696969;font-size:0.85rem;">'
                'טיפ: נסה שוב בעוד כמה דקות, או השתמש ב<strong>נתונים היסטוריים</strong> שתמיד זמינים.'
                '</p>'
            )

        # ── Data loaded — show results ────────────────────────────────────────
        elif "_rt_df" in st.session_state and st.session_state.get("_rt_loaded_city") == selected_city_rt:
            df_rt = st.session_state["_rt_df"].copy()

            # ── Polygon mode: filter by pre-drawn selection polygon ────────────
            if rt_area_mode == "✏️ פוליגון על מפה":
                _poly_rt = st.session_state.get("_rt_select_polygon")
                if _poly_rt and "_lat" in df_rt.columns:
                    _valid_coords = df_rt.dropna(subset=["_lat", "_lon"])
                    _in_poly_rt = _pts_in_polygon(
                        _valid_coords["_lat"].values,
                        _valid_coords["_lon"].values,
                        _poly_rt,
                    )
                    df_rt = _valid_coords[_in_poly_rt].copy()

                    # Show read-only map with polygon outline + listing dots
                    try:
                        from streamlit_folium import st_folium as _st_folium_disp
                        import folium as _folium_disp
                        _rt_has_coords_d = not df_rt.empty and "_lat" in df_rt.columns
                        _ctr_lat = float(df_rt["_lat"].dropna().mean()) if _rt_has_coords_d else 31.8
                        _ctr_lon = float(df_rt["_lon"].dropna().mean()) if _rt_has_coords_d else 34.9
                        _fmap_disp = _folium_disp.Map(location=[_ctr_lat, _ctr_lon],
                                                       zoom_start=13, tiles="OpenStreetMap")
                        _folium_disp.Polygon(
                            locations=[[p[1], p[0]] for p in _poly_rt],
                            color="#ff4b4b", fill=False, weight=2,
                        ).add_to(_fmap_disp)
                        for _, _rw in df_rt.dropna(subset=["_lat", "_lon"]).iterrows():
                            _folium_disp.CircleMarker(
                                location=[_rw["_lat"], _rw["_lon"]],
                                radius=5, color="#006AFF", fill=True,
                                fill_color="#006AFF", fill_opacity=0.6, weight=1,
                                tooltip=f"{int(_rw['מחיר מבוקש (₪)']):,} ₪",
                            ).add_to(_fmap_disp)
                        _st_folium_disp(_fmap_disp, key="rt_filtered_map",
                                         use_container_width=True, height=400)
                    except ImportError:
                        pass
                    st.caption(f"🔍 {len(df_rt)} מודעות בתוך הפוליגון")
                else:
                    st.info("ממתין לציור פוליגון — מוצגות כל המודעות")

                st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("מודעות פעילות", len(df_rt),
                       help="כמה מודעות יד2 פעילות נמצאו בעיר זו.")
            _avg_p = df_rt["מחיר מבוקש (₪)"].mean()
            sm2.metric("מחיר ממוצע מבוקש", f"{_avg_p:,.0f} ₪",
                       help="ממוצע המחירים המבוקשים — טרם עסקה סגורה.")
            _avg_s = df_rt['שטח (מ"ר)'].dropna().mean() if 'שטח (מ"ר)' in df_rt.columns else np.nan
            sm3.metric("שטח ממוצע",
                       f"{_avg_s:.0f} מ\"ר" if not pd.isna(_avg_s) else "—",
                       help="שטח ממוצע של הדירות המוצעות.")
            _avg_v = df_rt["ציון כדאיות"].mean()
            sm4.metric("ציון כדאיות ממוצע", f"{_avg_v:.1f} / 10",
                       help="מעל 5 = בממוצע המחיר מתחת להערכת המודל.")

            # ── Filters ───────────────────────────────────────────────────────
            st.markdown('<div style="height:1px;background:#D4EBE2;margin:20px 0;border-radius:1px;"></div>', unsafe_allow_html=True)
            rtl('<h3>סינון מודעות</h3>')

            _price_col = "מחיר מבוקש (₪)"
            _area_col  = 'שטח (מ"ר)'
            _rooms_col = "חדרים"
            df_rt_filtered = df_rt.copy()

            _pv = df_rt[_price_col].dropna()
            _rv = df_rt[_rooms_col].dropna()
            _rv = _rv[(_rv >= 1) & (_rv <= 10)]
            _av = df_rt[_area_col].dropna()
            _av = _av[(_av >= 20) & (_av <= 600)]

            _filter_mode_rt = st.radio(
                "🔍 מצב סינון:",
                ["✋ ידני", "🤖 AI לפי הפרופיל שלי"],
                horizontal=True, key="rt_filter_mode",
                index=1,
            )

            if _filter_mode_rt == "🤖 AI לפי הפרופיל שלי":
                _prof_budget = st.session_state.get("profile_budget", 2_000_000)
                _prof_goal   = st.session_state.get("profile_goal", "💵 תשואה שוטפת (השכרה)")
                _is_rental   = "תשואה" in _prof_goal

                if not st.session_state.get("profile_configured", False):
                    st.warning("⚠️ הפרופיל טרם הוגדר — עבור לדף **הפרופיל שלי** כדי להגדיר תקציב ומטרה.")
                    df_rt_filtered = df_rt.sort_values("ציון כדאיות", ascending=False).copy()
                else:
                    if _is_rental:
                        _rooms_range  = (1.5, 3.5)
                        _rooms_label  = "1.5–3.5"
                        _rooms_reason = "דירות קטנות-בינוניות — ביקוש גבוה להשכרה ותשואה % גבוהה ביחס למחיר"
                        _goal_label   = "תשואה שוטפת (השכרה)"
                    else:
                        _rooms_range  = (3.0, 6.0)
                        _rooms_label  = "3–6"
                        _rooms_reason = "נכסים גדולים יותר — פוטנציאל עליית ערך גבוה ובסיס קונים רחב יותר במכירה"
                        _goal_label   = "עליית ערך (מכירה ברווח)"

                    # Show current filter settings as read-only summary
                    fm1, fm2, fm3, fm4 = st.columns(4)
                    fm1.metric("💰 מחיר עד", f"{_prof_budget:,} ₪",
                               help="תקציב מקסימלי מהפרופיל שלך.")
                    fm2.metric("🛏️ חדרים", _rooms_label,
                               help=f"מותאם למטרת {_goal_label}.")
                    fm3.metric("📐 שטח", "ללא הגבלה",
                               help="AI לא מסנן לפי שטח — ניתן לעבור למצב ידני לסינון מדויק.")
                    fm4.metric("🎯 מטרה", _goal_label.split("(")[0].strip(),
                               help="עדכן בדף הפרופיל שלי.")

                    with st.expander("💡 איך המסנן הגיע להחלטות האלה?"):
                        rtl(f"""
                        <p>המסנן יישם <strong>3 כללים</strong> מהפרופיל שלך:</p>
                        <ol>
                          <li>
                            <strong>תקציב — מסנן קשיח:</strong>
                            הוסרו כל המודעות מעל <strong>{_prof_budget:,} ₪</strong>.
                            זהו הגבול שהגדרת בפרופיל — אין טעם להציג מה שלא בטווח.
                          </li>
                          <li>
                            <strong>חדרים — מותאם למטרה "{_goal_label}":</strong>
                            נשמרו רק מודעות עם {_rooms_label} חדרים (או ללא מידע על חדרים).
                            ההיגיון: {_rooms_reason}.
                          </li>
                          <li>
                            <strong>מיון — ציון כדאיות יורד:</strong>
                            ציון גבוה = מחיר מבוקש <em>נמוך</em> ממחיר השוק שהמודל חזה.
                            כך המודעות ה"זולות ביחס לשוק" עולות לראש הרשימה.
                          </li>
                        </ol>
                        <p style="color:#696969;font-size:0.85rem;">
                          ניתן לשנות תקציב ומטרה בדף <strong>הפרופיל שלי</strong>,
                          או לעבור ל<strong>סינון ידני</strong> לשליטה מלאה.
                        </p>
                        """)

                    # Apply filters
                    df_rt_filtered = df_rt[df_rt[_price_col] <= _prof_budget].copy()
                    df_rt_filtered = df_rt_filtered[
                        df_rt_filtered[_rooms_col].isna() |
                        df_rt_filtered[_rooms_col].between(*_rooms_range)
                    ]
                    df_rt_filtered = df_rt_filtered.sort_values("ציון כדאיות", ascending=False)

            else:
                # ── Manual filter UI ──────────────────────────────────────────
                # Row 1 — price (min / max) + rooms (min / max)
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    p_from = st.number_input(
                        "💰 מחיר מינימום (₪ מ')",
                        min_value=0.0, max_value=200.0,
                        value=round(float(_pv.min()) / 1e6, 1) if len(_pv) else 0.0,
                        step=0.1, format="%.1f", key="rt_p_from",
                    )
                with fc2:
                    p_to = st.number_input(
                        "מחיר מקסימום (₪ מ')",
                        min_value=0.0, max_value=200.0,
                        value=round(float(_pv.max()) / 1e6 + 0.1, 1) if len(_pv) else 200.0,
                        step=0.1, format="%.1f", key="rt_p_to",
                    )
                with fc3:
                    r_from = st.number_input(
                        "🛏️ חדרים מינימום",
                        min_value=1.0, max_value=10.0,
                        value=float(_rv.min()) if len(_rv) else 1.0,
                        step=0.5, format="%.1f", key="rt_r_from",
                    )
                with fc4:
                    r_to = st.number_input(
                        "חדרים מקסימום",
                        min_value=1.0, max_value=10.0,
                        value=float(_rv.max()) if len(_rv) else 10.0,
                        step=0.5, format="%.1f", key="rt_r_to",
                    )

                # Row 2 — area (min / max)
                fa1, fa2, _, _ = st.columns(4)
                with fa1:
                    a_from = st.number_input(
                        '📐 שטח מינימום (מ"ר)',
                        min_value=0, max_value=1000,
                        value=int(_av.min()) if len(_av) else 0,
                        step=5, key="rt_a_from",
                    )
                with fa2:
                    a_to = st.number_input(
                        'שטח מקסימום (מ"ר)',
                        min_value=0, max_value=1000,
                        value=int(_av.max()) + 5 if len(_av) else 1000,
                        step=5, key="rt_a_to",
                    )

                # Apply manual filters
                if len(_pv):
                    df_rt_filtered = df_rt_filtered[
                        df_rt_filtered[_price_col].between(p_from * 1e6, p_to * 1e6)
                    ]
                if len(_rv):
                    df_rt_filtered = df_rt_filtered[
                        df_rt_filtered[_rooms_col].isna() |
                        df_rt_filtered[_rooms_col].between(r_from, r_to)
                    ]
                if len(_av):
                    df_rt_filtered = df_rt_filtered[
                        df_rt_filtered[_area_col].isna() |
                        df_rt_filtered[_area_col].between(a_from, a_to)
                    ]
                df_rt_filtered = df_rt_filtered.sort_values("ציון כדאיות", ascending=False)
            rtl(f'<p><strong>{len(df_rt_filtered)} מודעות</strong> מוצגות — מדורגות מהכדאית ביותר</p>')

            # Build display table — drop שכונה (Yad2 returns regional area name, same for all rows)
            show_rt = df_rt_filtered.copy()
            if "מחיר חזוי (₪)" in show_rt.columns:
                show_rt["מחיר חזוי (₪)"] = show_rt["מחיר חזוי (₪)"].round(0).astype("Int64")
            if _area_col in show_rt.columns:
                show_rt[_area_col] = show_rt[_area_col].round(0).astype("Int64")
            if "קומה" in show_rt.columns:
                try:
                    show_rt["קומה"] = show_rt["קומה"].round(0).astype("Int64")
                except Exception:
                    pass
            if "_yad2_id" in show_rt.columns:
                show_rt["🔗 קישור"] = show_rt["_yad2_id"].apply(
                    lambda i: f"https://www.yad2.co.il/item/{i}" if i else ""
                )

            _multi_city_rt = rt_area_mode in ("🗺️ מחוז", "✏️ פוליגון על מפה")
            if _multi_city_rt and "_city" in show_rt.columns:
                show_rt = show_rt.rename(columns={"_city": "עיר"})
            _city_col_rt = ["עיר"] if _multi_city_rt and "עיר" in show_rt.columns else []
            rt_disp_cols = [c for c in _city_col_rt + [
                "רחוב", "מס' בית", 'שטח (מ"ר)', "חדרים", "קומה",
                "מחיר מבוקש (₪)", "מחיר חזוי (₪)", "ציון כדאיות", "🔗 קישור",
            ] if c in show_rt.columns]

            st.dataframe(
                show_rt[rt_disp_cols].head(100),
                column_config={
                    "ציון כדאיות": st.column_config.ProgressColumn(
                        "ציון כדאיות (0-10)", min_value=0, max_value=10, format="%.1f",
                    ),
                    "מחיר מבוקש (₪)": st.column_config.NumberColumn(format="₪%,d"),
                    "מחיר חזוי (₪)":  st.column_config.NumberColumn(format="₪%,d"),
                    "🔗 קישור": st.column_config.LinkColumn("🔗 יד2", display_text="פתח מודעה"),
                },
                hide_index=True, use_container_width=True,
            )

            with st.expander("📖 איך לקרוא את הטבלה?"):
                rtl("""
                <table style="width:100%;border-collapse:collapse;">
                  <tr style="background:#F5F5F5;">
                    <th style="padding:8px;border:1px solid #E0E0E0;text-align:right;">עמודה</th>
                    <th style="padding:8px;border:1px solid #E0E0E0;text-align:right;">הסבר</th>
                  </tr>
                  <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>ציון כדאיות</strong></td>
                      <td style="padding:8px;border:1px solid #E0E0E0;">0–100. ירוק = מחיר מתחת להערכת המודל = הזדמנות</td></tr>
                  <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מחיר מבוקש</strong></td>
                      <td style="padding:8px;border:1px solid #E0E0E0;">מה המוכר דורש — עדיין לא עסקה סגורה</td></tr>
                  <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>מחיר חזוי</strong></td>
                      <td style="padding:8px;border:1px solid #E0E0E0;">מה המודל מעריך שהנכס שווה לפי עסקאות היסטוריות</td></tr>
                  <tr><td style="padding:8px;border:1px solid #E0E0E0;"><strong>קישור</strong></td>
                      <td style="padding:8px;border:1px solid #E0E0E0;">לחץ לפתיחת המודעה המלאה ביד2</td></tr>
                </table>
                <p>הטבלה ממוינת מציון כדאיות גבוה לנמוך. שים לב: מחיר מבוקש ≠ מחיר עסקה סופי.</p>
                """)

            with st.expander("📊 גרף התפלגות מחירים"):
                if len(df_rt_filtered) >= 5:
                    fig_rt = px.histogram(
                        df_rt_filtered, x="מחיר מבוקש (₪)", nbins=25,
                        title=f"התפלגות מחירים מבוקשים — {selected_city_rt}",
                        labels={"מחיר מבוקש (₪)": "מחיר מבוקש (₪)", "count": "כמות מודעות"},
                        height=350, color_discrete_sequence=["#E25252"],
                    )
                    fig_rt.update_layout(margin=dict(t=40, b=20), bargap=0.05)
                    st.plotly_chart(fig_rt, use_container_width=True)
                    rtl(f"""
                    <p style="color:#555;font-size:0.85rem;">
                    מחיר מינימום: {df_rt_filtered['מחיר מבוקש (₪)'].min():,.0f} ₪ &nbsp;|&nbsp;
                    חציון: {int(df_rt_filtered['מחיר מבוקש (₪)'].median()):,.0f} ₪ &nbsp;|&nbsp;
                    מחיר מקסימום: {df_rt_filtered['מחיר מבוקש (₪)'].max():,.0f} ₪
                    </p>
                    """)
                else:
                    st.info("אין מספיק נתונים להצגת גרף עם הפילטרים הנוכחיים.")

        # ── Empty state — waiting for user to press the load button ──────────
        else:
            rtl("""
            <div dir="rtl" style="background:#EBF3FF;border:1px solid #006AFF;border-radius:10px;
                 padding:20px 24px;text-align:right;margin:16px 0;">
              <p style="font-size:1.1rem;font-weight:700;color:#006AFF;">📡 מצב בזמן אמת</p>
              <p>לחץ <strong>טען נתונים מיד2</strong> כדי לשלוף את המודעות הפעילות כעת בעיר שבחרת.</p>
              <p style="color:#555;font-size:0.88rem;">
                הנתונים נשלפים ישירות מאתר יד2 ועשויים להשתנות בכל רגע.<br>
                ציון הכדאיות מחושב ע"י מודל הבינה המלאכותית שלנו מול המחיר המבוקש.
              </p>
            </div>
            """)
