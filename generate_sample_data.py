"""
generate_sample_data.py
-----------------------
Run this once from the project root to create fake data for testing the dashboard.

    py generate_sample_data.py

It creates 2 periods of sample data (2024_Q1 and 2024_Q2) in the exact folder
structure and CSV format that the Streamlit app expects.
"""

from pathlib import Path
import random
import pandas as pd

random.seed(42)

# ── Folder paths (same as app.py) ────────────────────────────────────────────
BACKEND_ROOT      = Path(__file__).parent / "backend"
HUB_MP_DIR        = BACKEND_ROOT / "hubMP_folder"
ROUTE_MP_DIR      = BACKEND_ROOT / "routeMP_folder"
HUB_AIRLINE_DIR   = BACKEND_ROOT / "hubxairline_folder"
ROUTE_AIRLINE_DIR = BACKEND_ROOT / "routexairline_folder"

for d in [HUB_MP_DIR, ROUTE_MP_DIR, HUB_AIRLINE_DIR, ROUTE_AIRLINE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sample values ─────────────────────────────────────────────────────────────
CARRIERS = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
AIRPORTS = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "MIA", "SEA", "BOS",
            "LAS", "PHX", "IAH", "MCO", "EWR", "MSP", "DTW", "PHL", "LGA", "CLT"]
STATES   = ["GA", "CA", "IL", "TX", "CO", "NY", "CA", "FL", "WA", "MA",
            "NV", "AZ", "TX", "FL", "NJ", "MN", "MI", "PA", "NY", "NC"]

AIRPORT_STATE = dict(zip(AIRPORTS, STATES))

PERIODS = [
    (2024, 1),
    (2024, 2),
    (2024, 3),
    (2023, 4),
]

def period_tag(year, quarter):
    return f"{year}_Q{quarter}"


def make_route_airline(year, quarter):
    """Mimics routexairline_YEAR_Q#.csv output from capstone_parse.py"""
    rows = []
    for origin in AIRPORTS:
        for dest in AIRPORTS:
            if origin == dest:
                continue
            # Not every carrier flies every route
            carriers_on_route = random.sample(CARRIERS, k=random.randint(1, 5))
            base_dist = random.uniform(300, 2800)
            base_fare = 80 + base_dist * 0.09 + random.uniform(-30, 60)
            for carrier in carriers_on_route:
                pax = random.randint(500, 120_000)
                fare_var = random.uniform(0.8, 1.3)
                rows.append({
                    "Origin":               origin,
                    "Dest":                 dest,
                    "Carrier":              carrier,
                    "avg_fare_weighted":    round(base_fare * fare_var, 2),
                    "avg_distance_weighted": round(base_dist, 2),
                    "total_passengers":     pax,
                    "row_count":            random.randint(10, 500),
                })
    return pd.DataFrame(rows)


def make_hub_airline(year, quarter):
    """Mimics hubxairline_YEAR_Q#.csv output from capstone_parse.py"""
    rows = []
    for origin in AIRPORTS:
        state = AIRPORT_STATE[origin]
        carriers_at_hub = random.sample(CARRIERS, k=random.randint(2, 6))
        for carrier in carriers_at_hub:
            pax = random.randint(5_000, 500_000)
            rows.append({
                "Origin":               origin,
                "OriginState":          state,
                "Carrier":              carrier,
                "avg_fare_weighted":    round(random.uniform(150, 500), 2),
                "avg_distance_weighted": round(random.uniform(400, 2000), 2),
                "total_passengers":     pax,
                "row_count":            random.randint(50, 2000),
            })
    return pd.DataFrame(rows)


def make_route_market_power(route_df, year, quarter):
    """
    Mimics route_market_power_YEAR_Q#.csv output from capstone_analyze.py.
    Computes real HHI and market share from the route_airline data.
    """
    df = route_df.copy()
    df["fare_x_pax"] = df["avg_fare_weighted"] * df["total_passengers"]

    # Route-level totals
    market = df.groupby(["Origin", "Dest"]).agg(
        route_total_passengers_all=("total_passengers", "sum"),
        route_fare_x_pax=("fare_x_pax", "sum"),
        route_min_fare_all=("avg_fare_weighted", "min"),
        carriers_on_route_all=("Carrier", "nunique"),
    ).reset_index()
    market["route_avg_fare_all"] = (market["route_fare_x_pax"] / market["route_total_passengers_all"]).round(2)

    # Valid-only totals (same here since no invalid carriers in sample)
    market_valid = df.groupby(["Origin", "Dest"]).agg(
        route_total_passengers_valid=("total_passengers", "sum"),
        carriers_on_route_valid=("Carrier", "nunique"),
    ).reset_index()

    m = df.merge(market, on=["Origin", "Dest"]).merge(market_valid, on=["Origin", "Dest"])
    m["route_share"] = (m["total_passengers"] / m["route_total_passengers_valid"]).round(4)
    m["route_share_sq"] = m["route_share"] ** 2

    hhi = m.groupby(["Origin", "Dest"])["route_share_sq"].sum().reset_index()
    hhi["route_HHI"] = (hhi["route_share_sq"] * 10000).round(0)
    m = m.merge(hhi[["Origin", "Dest", "route_HHI"]], on=["Origin", "Dest"])

    keep = [
        "Origin", "Dest", "Carrier",
        "total_passengers", "row_count",
        "avg_fare_weighted", "avg_distance_weighted",
        "route_total_passengers_all", "route_total_passengers_valid",
        "carriers_on_route_all", "carriers_on_route_valid",
        "route_share", "route_HHI",
        "route_avg_fare_all", "route_min_fare_all",
    ]
    return m[keep].sort_values(["Origin", "Dest", "Carrier"]).reset_index(drop=True)


def make_hub_market_power(hub_df, year, quarter):
    """
    Mimics hub_market_power_YEAR_Q#.csv output from capstone_analyze.py.
    """
    df = hub_df.copy()
    df["fare_x_pax"] = df["avg_fare_weighted"] * df["total_passengers"]

    market = df.groupby(["Origin", "OriginState"]).agg(
        hub_total_passengers_all=("total_passengers", "sum"),
        hub_fare_x_pax=("fare_x_pax", "sum"),
        hub_min_fare_all=("avg_fare_weighted", "min"),
        carriers_at_hub_all=("Carrier", "nunique"),
    ).reset_index()
    market["hub_avg_fare_all"] = (market["hub_fare_x_pax"] / market["hub_total_passengers_all"]).round(2)

    market_valid = df.groupby(["Origin", "OriginState"]).agg(
        hub_total_passengers_valid=("total_passengers", "sum"),
        carriers_at_hub_valid=("Carrier", "nunique"),
    ).reset_index()

    m = df.merge(market, on=["Origin", "OriginState"]).merge(market_valid, on=["Origin", "OriginState"])
    m["hub_share"] = (m["total_passengers"] / m["hub_total_passengers_valid"]).round(4)
    m["hub_share_sq"] = m["hub_share"] ** 2

    hhi = m.groupby(["Origin", "OriginState"])["hub_share_sq"].sum().reset_index()
    hhi["hub_HHI"] = (hhi["hub_share_sq"] * 10000).round(0)
    m = m.merge(hhi[["Origin", "OriginState", "hub_HHI"]], on=["Origin", "OriginState"])

    keep = [
        "Origin", "OriginState", "Carrier",
        "total_passengers", "row_count",
        "avg_fare_weighted", "avg_distance_weighted",
        "hub_total_passengers_all", "hub_total_passengers_valid",
        "carriers_at_hub_all", "carriers_at_hub_valid",
        "hub_share", "hub_HHI",
        "hub_avg_fare_all", "hub_min_fare_all",
    ]
    return m[keep].sort_values(["Origin", "OriginState", "Carrier"]).reset_index(drop=True)


# ── Generate and save all periods ─────────────────────────────────────────────
for year, quarter in PERIODS:
    tag = period_tag(year, quarter)
    print(f"\n── Generating {tag} ──")

    route_airline = make_route_airline(year, quarter)
    hub_airline   = make_hub_airline(year, quarter)
    route_mp      = make_route_market_power(route_airline, year, quarter)
    hub_mp        = make_hub_market_power(hub_airline, year, quarter)

    # Save to the folders app.py reads from
    route_airline.to_csv(ROUTE_AIRLINE_DIR / f"routexairline_{tag}.csv",   index=False)
    hub_airline.to_csv(  HUB_AIRLINE_DIR   / f"hubxairline_{tag}.csv",     index=False)
    route_mp.to_csv(     ROUTE_MP_DIR      / f"route_market_power_{tag}.csv", index=False)
    hub_mp.to_csv(       HUB_MP_DIR        / f"hub_market_power_{tag}.csv",   index=False)

    print(f"  route_airline:    {len(route_airline):,} rows")
    print(f"  hub_airline:      {len(hub_airline):,} rows")
    print(f"  route_market_power: {len(route_mp):,} rows")
    print(f"  hub_market_power:   {len(hub_mp):,} rows")

print("\n✅ Done! 4 periods of sample data generated.")
print("   Start the app with:  py -m streamlit run frontend/app.py")
print("   Then click Start → Analyze One Period or Analyze Multiple Periods.")