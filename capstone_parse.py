import argparse
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import json
from pathlib import Path

import pandas as pd

COL_YEAR = "Year"
COL_QUARTER = "Quarter"
COL_ORIGIN = "Origin"                 # origin airport code
COL_ORIGIN_STATE = "OriginState"      # state code
COL_DEST = "Dest"                     # dest airport code
COL_DEST_STATE = "DestState"
COL_CARRIER = "TkCarrier"             # which airline (shown as a Two letter or digit code, need an interpreter down the line.)
COL_PASSENGERS = "Passengers"
COL_FARE = "MktFare"
COL_DISTANCE = "NonStopMiles"         # use NonStopMiles for end-to-end distance


#########################################################################################################################

# OPTIONALLY SET PRICE SCALE TO REMOVE 
# CERTAIN OUTLIER PRICES FROM TESTING

fare_upper_bound = 1200
fare_lower_bound = 50

#For Bug-testing (manually source file)
currentData = "Origin_and_Destination_Survey_DB1BMarket_2025_1.csv"

#########################################################################################################################

@dataclass
class Agg:
    passengers_sum: float = 0.0
    fare_x_passengers_sum: float = 0.0
    miles_x_passengers_sum: float = 0.0
    row_count: int = 0


HubAirKey = Tuple[str, str, str]       #(origin, originstate, carrier)
RouteAirKey = Tuple[str, str, str]     #(origin, dest, carrier)


def _wavg(sum_xw: float, sum_w: float) -> float:
    return (sum_xw / sum_w) if sum_w > 0 else float("nan")


def _assert_required_cols(cols) -> None:
    required = {
        COL_YEAR, COL_QUARTER,
        COL_ORIGIN, COL_ORIGIN_STATE,
        COL_DEST,
        COL_CARRIER, COL_PASSENGERS, COL_FARE, COL_DISTANCE,
    }
    missing = sorted([c for c in required if c not in cols])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# Read in year and quarter for naming schema 
def detect_single_period(csv_path: str, chunksize: int = 200_000) -> Tuple[int, int]:
    reader = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
    first = next(reader)
    _assert_required_cols(first.columns)

    periods = set(zip(first[COL_YEAR].dropna().unique(), first[COL_QUARTER].dropna().unique()))
    for chunk in reader:
        ys = chunk[COL_YEAR].dropna().unique()
        qs = chunk[COL_QUARTER].dropna().unique()
        for y in ys:
            for q in qs:
                periods.add((int(y), int(q)))
                if len(periods) > 1:
                    raise ValueError(
                        f"File contains multiple Year/Quarter values: {sorted(periods)}. "
                        f"Pass --year and --quarter to filter, or export a single period."
                    )

    if not periods:
        raise ValueError("Could not detect Year/Quarter (no values found).")

    year, quarter = next(iter(periods))
    return int(year), int(quarter)


def ingest(
    csv_path: str,
    fare_lower_bound: Optional[float],
    fare_upper_bound: Optional[float],
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    chunksize: int = 750_000,
    verbose: int = 1,
) -> Tuple[int, int, Dict[HubAirKey, Agg], Dict[RouteAirKey, Agg]]:
    """
    output:
      - detected/used year, quarter
      - hub_airline_aggs: (origin, originstate, carrier) -> agg
      - route_airline_aggs: (origin, originstate, carrier)  -> agg (later use for HHI, markup proxies, etc.)
    """
    if year is None or quarter is None:
        year, quarter = detect_single_period(csv_path)

    if verbose:
        print(f"[ingest] using Year={year}, Quarter={quarter}")
        if fare_lower_bound is not None or fare_upper_bound is not None:
            print(f"[ingest] fare bounds: lower={fare_lower_bound} upper={fare_upper_bound}")

    hub_airline: Dict[HubAirKey, Agg] = defaultdict(Agg)
    route_airline: Dict[RouteAirKey, Agg] = defaultdict(Agg)

    usecols = [
        COL_YEAR, COL_QUARTER,
        COL_ORIGIN, COL_ORIGIN_STATE,
        COL_DEST,
        COL_CARRIER, COL_PASSENGERS, COL_FARE, COL_DISTANCE,
    ]

    total_seen = 0
    total_kept = 0
    chunk_idx = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, usecols=lambda c: c in usecols):
        chunk_idx += 1
        total_seen += len(chunk)
        if verbose:
            print(f"[chunk {chunk_idx}] read={len(chunk):,} total_seen={total_seen:,}")

        # filter to the single period
        df = chunk[(chunk[COL_YEAR] == year) & (chunk[COL_QUARTER] == quarter)]
        if df.empty:
            continue

        # numeric coercion
        df[COL_PASSENGERS] = pd.to_numeric(df[COL_PASSENGERS], errors="coerce")
        df[COL_FARE] = pd.to_numeric(df[COL_FARE], errors="coerce")
        df[COL_DISTANCE] = pd.to_numeric(df[COL_DISTANCE], errors="coerce")

        # drop invalid essentials
        before = len(df)
        df = df.dropna(subset=[COL_ORIGIN, COL_ORIGIN_STATE, COL_DEST, COL_CARRIER, COL_PASSENGERS, COL_FARE, COL_DISTANCE])
        df = df[(df[COL_PASSENGERS] > 0) & (df[COL_FARE] > 0) & (df[COL_DISTANCE] > 0)]

        # fare bounds filter (this DOES exclude outliers intentionally)
        if fare_lower_bound is not None:
            df = df[df[COL_FARE] >= fare_lower_bound]
        if fare_upper_bound is not None:
            df = df[df[COL_FARE] <= fare_upper_bound]

        total_kept += len(df)
        if verbose >= 2:
            print(f"[chunk {chunk_idx}] period_rows={before:,} kept_after_clean+fare={len(df):,}")

        if df.empty:
            continue

        # normalize keys
        df[COL_ORIGIN] = df[COL_ORIGIN].astype(str).str.strip()
        df[COL_ORIGIN_STATE] = df[COL_ORIGIN_STATE].astype(str).str.strip()
        df[COL_DEST] = df[COL_DEST].astype(str).str.strip()
        df[COL_CARRIER] = df[COL_CARRIER].astype(str).str.strip()

        # derived weighted sums
        df["_fare_x_passengers"] = df[COL_FARE] * df[COL_PASSENGERS]
        df["_miles_x_passengers"] = df[COL_DISTANCE] * df[COL_PASSENGERS]

        ## Hub × Airline ##
        g1 = df.groupby([COL_ORIGIN, COL_ORIGIN_STATE, COL_CARRIER], sort=False)[
            [COL_PASSENGERS, "_fare_x_passengers", "_miles_x_passengers"]
        ].sum()
        c1 = df.groupby([COL_ORIGIN, COL_ORIGIN_STATE, COL_CARRIER], sort=False).size()

        for (origin, state, carrier), row in g1.iterrows():
            a = hub_airline[(origin, state, carrier)]
            a.passengers_sum += float(row[COL_PASSENGERS])
            a.fare_x_passengers_sum += float(row["_fare_x_passengers"])
            a.miles_x_passengers_sum += float(row["_miles_x_passengers"])
            a.row_count += int(c1.loc[(origin, state, carrier)])

        ## Route × Airline ##
        g2 = df.groupby([COL_ORIGIN, COL_DEST, COL_CARRIER], sort=False)[
            [COL_PASSENGERS, "_fare_x_passengers", "_miles_x_passengers"]
        ].sum()
        c2 = df.groupby([COL_ORIGIN, COL_DEST, COL_CARRIER], sort=False).size()

        for (origin, dest, carrier), row in g2.iterrows():
            a = route_airline[(origin, dest, carrier)]
            a.passengers_sum += float(row[COL_PASSENGERS])
            a.fare_x_passengers_sum += float(row["_fare_x_passengers"])
            a.miles_x_passengers_sum += float(row["_miles_x_passengers"])
            a.row_count += int(c2.loc[(origin, dest, carrier)])

        if verbose:
            print(f"[chunk {chunk_idx}] kept_this_chunk={len(df):,} total_kept={total_kept:,} hub_groups={len(hub_airline):,} route_groups={len(route_airline):,}")

    if verbose:
        print(f"[done] total_seen={total_seen:,} total_kept={total_kept:,}")
        print(f"[done] hub×airline groups={len(hub_airline):,} route×airline groups={len(route_airline):,}")

    return year, quarter, hub_airline, route_airline


def hub_airline_table(hub_airline: Dict[HubAirKey, Agg]) -> pd.DataFrame:
    """
    output:
      Origin, OriginState, Carrier
      avg_fare_weighted, avg_distance_weighted
      total_passengers, row_count
    """
    rows = []
    for (origin, state, carrier), a in hub_airline.items():
        rows.append({
            "Origin": origin,
            "OriginState": state,
            "Carrier": carrier,
            "avg_fare_weighted": round(_wavg(a.fare_x_passengers_sum, a.passengers_sum),2),
            "avg_distance_weighted": round(_wavg(a.miles_x_passengers_sum, a.passengers_sum),2),
            "total_passengers": (a.passengers_sum),
            "row_count": a.row_count,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Origin", "OriginState", "Carrier"]).reset_index(drop=True)


def route_airline_table(route_airline: Dict[RouteAirKey, Agg]) -> pd.DataFrame:
    """
    output:
      Origin, Dest, Carrier
      avg_fare_weighted, avg_distance_weighted
      total_passengers, row_count

    for:
      - HHI: shares within (Origin, Dest)
      - compare carrier fares vs route average or vs model-predicted fare
    """
    rows = []
    for (origin, dest, carrier), a in route_airline.items():
        rows.append({
            "Origin": origin,
            "Dest": dest,
            "Carrier": carrier,
            "avg_fare_weighted": round(_wavg(a.fare_x_passengers_sum, a.passengers_sum),2),
            "avg_distance_weighted": round(_wavg(a.miles_x_passengers_sum, a.passengers_sum),2),
            "total_passengers": (a.passengers_sum),
            "row_count": a.row_count,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Origin", "Dest", "Carrier"]).reset_index(drop=True)

def period_tag(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Override default CSV path")
    ap.add_argument("--fare_lower_bound", type=float, default=fare_lower_bound)
    ap.add_argument("--fare_upper_bound", type=float, default=fare_upper_bound)

    # if file contains multiple periods
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--quarter", type=int, default=None)

    ap.add_argument("--chunksize", type=int, default=750_000)
    ap.add_argument("--verbose", type=int, default=1)

    args = ap.parse_args()


    # manual csv input for bug testing or disable to input your own.
    csv_path = args.csv if args.csv is not None else currentData

    print(f"[main] using CSV file: {csv_path}")

    year, quarter, hub_airline, route_airline = ingest(
        csv_path=csv_path,
        fare_lower_bound=args.fare_lower_bound,
        fare_upper_bound=args.fare_upper_bound,
        year=args.year,
        quarter=args.quarter,
        chunksize=args.chunksize,
        verbose=args.verbose,
    )

    tag = period_tag(year, quarter)
    hub_df = hub_airline_table(hub_airline)
    route_df = route_airline_table(route_airline)

    hub_out = f"hubxairline_{tag}.csv"
    route_out = f"routexairline_{tag}.csv"

    print("\n=== HUB × AIRLINE (Origin hub only; no layover hubs) ===")
    print(hub_df.head(50).to_string(index=False))  #bug test preview
    hub_df.to_csv(hub_out, index=False)
    print(f"[saved] {hub_out} ({len(hub_df):,} rows)")

    print("\n=== ROUTE × AIRLINE (for later HHI / markup proxies) ===")
    print(route_df.head(50).to_string(index=False))  #bug test preview
    route_df.to_csv(route_out, index=False)
    print(f"[saved] {route_out} ({len(route_df):,} rows)")

    print(f"\n[info] period used: Year={year}, Quarter={quarter}")


if __name__ == "__main__":
    main()