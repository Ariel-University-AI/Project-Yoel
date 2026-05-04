import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from dataclasses import dataclass

# --- Data Structures ---

@dataclass
class InvestorProfile:
    max_budget: float
    min_rooms: float
    min_area: float

@dataclass
class AreaRecommendation:
    settlement: str
    avg_price: float
    avg_area: float
    avg_rooms: float
    num_deals: int
    score: float


# --- Processing Functions ---

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.dropna(subset=["dealAmount", "assetArea", "assetRoomNum", "settlementNameHeb"])
    df = df[df["dealAmount"] > 0]
    return df


def recommend_areas(df: pd.DataFrame, profile: InvestorProfile, top_n: int = 5) -> list[AreaRecommendation]:
    filtered = df[
        (df["dealAmount"] <= profile.max_budget) &
        (df["assetRoomNum"] >= profile.min_rooms) &
        (df["assetArea"] >= profile.min_area)
    ]

    grouped = filtered.groupby("settlementNameHeb").agg(
        avg_price=("dealAmount", "mean"),
        avg_area=("assetArea", "mean"),
        avg_rooms=("assetRoomNum", "mean"),
        num_deals=("dealAmount", "count"),
    ).reset_index()

    max_deals = grouped["num_deals"].max()
    max_price = grouped["avg_price"].max()

    grouped["score"] = (
        (grouped["num_deals"] / max_deals) * 60 +
        (1 - grouped["avg_price"] / max_price) * 40
    )

    grouped = grouped.sort_values("score", ascending=False).head(top_n)

    return [
        AreaRecommendation(
            settlement=row["settlementNameHeb"],
            avg_price=round(row["avg_price"]),
            avg_area=round(row["avg_area"], 1),
            avg_rooms=round(row["avg_rooms"], 1),
            num_deals=int(row["num_deals"]),
            score=round(row["score"], 1),
        )
        for _, row in grouped.iterrows()
    ]


# --- Output ---

def print_recommendations(recs: list[AreaRecommendation]):
    print("=== Recommended Areas for Investment ===\n")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.settlement}")
        print(f"   Avg price: {r.avg_price:,} ILS")
        print(f"   Avg area: {r.avg_area} sqm | Rooms: {r.avg_rooms}")
        print(f"   Deals in data: {r.num_deals} | Score: {r.score}/100")
        print()


# --- Example Run ---

if __name__ == "__main__":
    df = load_data("nadlan_final.xlsx")

    profile = InvestorProfile(
        max_budget=2_000_000,
        min_rooms=3,
        min_area=70,
    )

    recommendations = recommend_areas(df, profile, top_n=5)
    print_recommendations(recommendations)
