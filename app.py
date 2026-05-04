import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import load_data, recommend_areas, InvestorProfile

st.title("יועץ השקעות נדל״ן חכם")

# --- פרופיל משקיע ---
st.header("הגדר פרופיל השקעה")

max_budget = st.slider("תקציב מקסימלי (₪)", 500_000, 5_000_000, 2_000_000, step=100_000)
min_rooms = st.slider("מספר חדרים מינימלי", 1.0, 6.0, 3.0, step=0.5)
min_area = st.slider("שטח מינימלי (מ״ר)", 30, 200, 70)

# --- טעינת נתונים ---
@st.cache_data
def get_data():
    return load_data("nadlan_final.xlsx")

df = get_data()

# --- המלצות ---
if st.button("מצא אזורים מומלצים"):
    profile = InvestorProfile(
        max_budget=max_budget,
        min_rooms=min_rooms,
        min_area=min_area,
    )

    recs = recommend_areas(df, profile, top_n=5)

    st.header("אזורים מומלצים")

    table = pd.DataFrame([{
        "יישוב": r.settlement,
        "מחיר ממוצע (₪)": f"{r.avg_price:,}",
        "שטח ממוצע (מ״ר)": r.avg_area,
        "חדרים": r.avg_rooms,
        "עסקאות": r.num_deals,
        "ציון": f"{r.score}/100",
    } for r in recs])

    st.dataframe(table, use_container_width=True)

    # --- גרף ---
    st.header("שטח מול מחיר")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["assetArea"], df["dealAmount"] / 1_000_000, alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel("Area (sqm)")
    ax.set_ylabel("Price (millions ILS)")
    ax.set_title("Area vs Price")
    st.pyplot(fig)
