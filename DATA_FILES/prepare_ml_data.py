# -*- coding: utf-8 -*-
"""
Prepares all_data_clean.csv for ML model training.
Output: all_data_ml_ready.csv
"""
import pandas as pd
import numpy as np

INPUT  = "DATA_FILES/all_data_clean.csv"
OUTPUT = "DATA_FILES/all_data_ml_ready.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
print(f"Loaded: {len(df):,} rows, {df.shape[1]} columns")

# ── 1. Filter property types ──────────────────────────────────────────────────
# Drop non-residential types by substring matching (avoids encoding issues)
NON_RESIDENTIAL_CONTAINS = [
    "משרד",          # משרד - office
    "חנות",          # חנות - store
    "תעשייה",  # תעשיה - industry
    "מחסנים",  # מחסנים - warehouses
    "מלאכה",    # מלאכה - craft/workshop
    "קומבינציה",  # קומבינציה
    "עסק",                # עסק - business
    "מלונאות",  # מלונאות
    "אופציה",  # אופציה - option
    "בניני ציבור",  # בניני ציבור
    "חקלאי",   # חקלאי - agricultural
    "ניוד זכויות",  # ניוד זכויות
    "לא מעובדת",  # לא מעובדת
    "מסחרי ומשרדים",  # מסחרי + משרדים
]

before = len(df)
mask_drop = df["dealNatureDescription"].apply(
    lambda v: any(kw in str(v) for kw in NON_RESIDENTIAL_CONTAINS)
    if pd.notna(v) else False
)
df = df[~mask_drop]
print(f"After property type filter: {len(df):,} ({before - len(df):,} removed)")

# ── 2. Extract year and month from dealDate ───────────────────────────────────
df["dealDate"] = pd.to_datetime(df["dealDate"], errors="coerce")
df["deal_year"]  = df["dealDate"].dt.year
df["deal_month"] = df["dealDate"].dt.month

# ── 3. Map floorNo Hebrew text to numbers ─────────────────────────────────────
FLOOR_MAP = {
    "מרתף": -1,          # מרתף
    "קרקע": 0,           # קרקע
    "ראשונה": 1,  # ראשונה
    "שנייה": 2,    # שנייה
    "שלישית": 3,  # שלישית
    "רביעית": 4,  # רביעית
    "חמישית": 5,  # חמישית
    "שישית": 6,    # שישית
    "שביעית": 7,  # שביעית
    "שמינית": 8,  # שמינית
    "תשיעית": 9,  # תשיעית
    "עשירית": 10, # עשירית
    "אחת עשרה": 11,  # אחת עשרה
    "שתים עשרה": 12,  # שתים עשרה
    "שלוש עשרה": 13,  # שלוש עשרה
    "ארבע עשרה": 14,  # ארבע עשרה
    "חמש עשרה": 15,  # חמש עשרה
    "שש עשרה": 16,  # שש עשרה
    "קומה 1": 1,        # קומה 1
    "קומה 2": 2,        # קומה 2
    "קומה 3": 3,        # קומה 3
    "קומה 4": 4,        # קומה 4
    "קומה 5": 5,        # קומה 5
    "-": np.nan,
    "קרקע+מרתף": 0,  # קרקע+מרתף
    "קומה+מרתף": 1,  # קומה+מרתף
}

df["floor_num"] = df["floorNo"].map(FLOOR_MAP)
floor_median = df["floor_num"].median()
if np.isnan(floor_median):
    floor_median = 2.0
df["floor_num"] = df["floor_num"].fillna(floor_median)
print(f"floor_num: fill median = {floor_median}")

# ── 4. Handle missing values ──────────────────────────────────────────────────
df["assetRoomNum"]     = df["assetRoomNum"].fillna(df["assetRoomNum"].median())
df["socio_index_avg"]  = df["socio_index_avg"].fillna(df["socio_index_avg"].median())
df["socio_rank_avg"]   = df["socio_rank_avg"].fillna(df["socio_rank_avg"].median())
df["socio_cluster_mode"] = df["socio_cluster_mode"].fillna(df["socio_cluster_mode"].median())

# ── 5. Encode categorical columns ────────────────────────────────────────────
df["settlement_encoded"]  = df["settlementNameHeb"].astype("category").cat.codes
df["deal_nature_encoded"] = df["dealNatureDescription"].astype("category").cat.codes

# ── 6. Drop irrelevant columns ────────────────────────────────────────────────
DROP_COLS = [
    "objectid", "dealId", "gushNum", "parcelNum", "subParcelNum",
    "polygonId", "sourceorder", "streetCode", "streetNameHeb",
    "streetNameEng", "settlementNameEng", "houseNum",
    "E", "N",
    "floorNo", "dealDate",
    "neighborhood",
    "propertyTypeDescription",
    "dealNatureDescription",  # replaced by deal_nature_encoded
    "settlementNameHeb",      # replaced by settlement_encoded
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ── 7. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT}")
print(f"Final rows: {len(df):,}")
print(f"Final columns: {df.shape[1]}")
print(f"\nMissing values per column:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  No missing values!")
else:
    print(missing)