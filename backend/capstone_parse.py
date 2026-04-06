import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, Optional
import json
import sqlite3
from pathlib import Path

import pandas as pd

from carrier_codes import get_carrier_name

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
min_carrier_total_passengers = 1000
fare_bin_width = 5.0
invalid_carrier_codes_lc = {"99", "00", "", "nan", "none", "null", "unknown", "unk"}

BASE_DIR = Path(__file__).resolve().parent
HUB_AIRLINE_DIR = BASE_DIR / "hubxairline_folder"
ROUTE_AIRLINE_DIR = BASE_DIR / "routexairline_folder"
SPECIFIC_FARE_DISTRIBUTION_DIR = BASE_DIR / "specific_fare_distribution_charts"
UPLOADS_DIR = BASE_DIR / "uploads"

# For manual bug-testing fallback when --csv and --year/--quarter are omitted.
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
RouteFareBinKey = Tuple[str, str, str, float, float]  # (origin, dest, carrier, fare_bin_start, fare_bin_end)


@dataclass
class FareBinAgg:
    passengers_sum: float = 0.0
    row_count: int = 0


def _wavg(sum_xw: float, sum_w: float) -> float:
    return (sum_xw / sum_w) if sum_w > 0 else float("nan")


def is_invalid_carrier(code: str) -> bool:
    return str(code or "").strip().lower() in invalid_carrier_codes_lc

def period_tag(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"


def ensure_output_dirs() -> None:
    HUB_AIRLINE_DIR.mkdir(parents=True, exist_ok=True)
    ROUTE_AIRLINE_DIR.mkdir(parents=True, exist_ok=True)
    SPECIFIC_FARE_DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)


def raw_filename(year: int, quarter: int) -> str:
    return f"Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.csv"


def resolve_csv_path(
    csv_arg: Optional[str],
    year: Optional[int],
    quarter: Optional[int],
    uploads_dir: Path,
) -> Path:
    if csv_arg:
        return Path(csv_arg)

    if year is not None and quarter is not None:
        candidate = uploads_dir / raw_filename(year, quarter)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Could not find raw file for Year={year}, Quarter={quarter} at: {candidate}. "
            f"Either place it in uploads or pass --csv explicitly."
        )

    return Path(currentData)

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
    fare_bin_width: float,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    chunksize: int = 750_000,
    verbose: int = 1,
) -> Tuple[int, int, Dict[HubAirKey, Agg], Dict[RouteAirKey, Agg], Dict[RouteFareBinKey, FareBinAgg], int, int]:
    """
    output:
      - detected/used year, quarter
      - hub_airline_aggs: (origin, originstate, carrier) -> agg
      - route_airline_aggs: (origin, originstate, carrier)  -> agg (later use for HHI, markup proxies, etc.)
      - total_seen: total rows encountered
      - total_kept: total rows after filtering
    """
    if year is None or quarter is None:
        year, quarter = detect_single_period(csv_path)

    if verbose:
        print(f"[ingest] using Year={year}, Quarter={quarter}")
        if fare_lower_bound is not None or fare_upper_bound is not None:
            print(f"[ingest] fare bounds: lower={fare_lower_bound} upper={fare_upper_bound}")

    hub_airline: Dict[HubAirKey, Agg] = defaultdict(Agg)
    route_airline: Dict[RouteAirKey, Agg] = defaultdict(Agg)
    route_fare_distribution: Dict[RouteFareBinKey, FareBinAgg] = defaultdict(FareBinAgg)

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

        # specific fare-distribution cache buckets for route-specific charts
        safe_width = max(float(fare_bin_width), 0.01)
        df["_fare_bin_start"] = (df[COL_FARE] // safe_width) * safe_width
        df["_fare_bin_end"] = df["_fare_bin_start"] + safe_width
        g3 = df.groupby([COL_ORIGIN, COL_DEST, COL_CARRIER, "_fare_bin_start", "_fare_bin_end"], sort=False)[COL_PASSENGERS].sum()
        c3 = df.groupby([COL_ORIGIN, COL_DEST, COL_CARRIER, "_fare_bin_start", "_fare_bin_end"], sort=False).size()

        for (origin, dest, carrier, bin_start, bin_end), passenger_sum in g3.items():
            key: RouteFareBinKey = (
                str(origin),
                str(dest),
                str(carrier),
                float(bin_start),
                float(bin_end),
            )
            a = route_fare_distribution[key]
            a.passengers_sum += float(passenger_sum)
            a.row_count += int(c3.loc[(origin, dest, carrier, bin_start, bin_end)])

        if verbose:
            print(f"[chunk {chunk_idx}] kept_this_chunk={len(df):,} total_kept={total_kept:,} hub_groups={len(hub_airline):,} route_groups={len(route_airline):,}")

    if verbose:
        print(f"[done] total_seen={total_seen:,} total_kept={total_kept:,}")
        print(f"[done] hub×airline groups={len(hub_airline):,} route×airline groups={len(route_airline):,}")

    return year, quarter, hub_airline, route_airline, route_fare_distribution, total_seen, total_kept


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


def route_fare_distribution_table(route_fare_distribution: Dict[RouteFareBinKey, FareBinAgg]) -> pd.DataFrame:
    """
    output:
      Origin, Dest, Carrier
      fare_bin_start, fare_bin_end
      passengers_sum, row_count
    """
    rows = []
    for (origin, dest, carrier, bin_start, bin_end), agg in route_fare_distribution.items():
        rows.append({
            "Origin": origin,
            "Dest": dest,
            "Carrier": carrier,
            "fare_bin_start": round(float(bin_start), 4),
            "fare_bin_end": round(float(bin_end), 4),
            "passengers_sum": float(agg.passengers_sum),
            "row_count": int(agg.row_count),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "Origin", "Dest", "Carrier",
            "fare_bin_start", "fare_bin_end",
            "passengers_sum", "row_count",
        ])
    df = pd.DataFrame(rows)
    return df.sort_values(["Origin", "Dest", "Carrier", "fare_bin_start"]).reset_index(drop=True)

def generate_quality_report(
    year: int,
    quarter: int,
    total_seen: int,
    total_kept: int,
    hub_airline_count: int,
    route_airline_count: int,
    fare_lower: Optional[float],
    fare_upper: Optional[float],
) -> Dict:
    """Generate data quality validation report."""
    report = {
        "period": {"year": year, "quarter": quarter},
        "ingestion": {
            "total_rows_seen": total_seen,
            "total_rows_kept": total_kept,
            "rows_filtered": total_seen - total_kept,
            "retention_rate": round(total_kept / total_seen * 100, 2) if total_seen > 0 else 0.0,
        },
        "fare_filters": {
            "lower_bound": fare_lower,
            "upper_bound": fare_upper,
        },
        "aggregations": {
            "hub_airline_groups": hub_airline_count,
            "route_airline_groups": route_airline_count,
        },
    }
    return report


def save_quality_report(report: Dict, output_path: str):
    """Save quality report as JSON."""
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] Quality report: {output_path}")


def export_to_parquet(df: pd.DataFrame, output_path: str):
    """Export DataFrame to Parquet format for efficient storage."""
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    print(f"[saved] Parquet: {output_path} ({len(df):,} rows)")


def export_to_sqlite(
    hub_df: pd.DataFrame,
    route_df: pd.DataFrame,
    db_path: str,
    year: int,
    quarter: int,
    replace: bool = False,
):
    """
    Export hub and route dataframes to SQLite database.
    
    Args:
        hub_df: Hub×Airline DataFrame
        route_df: Route×Airline DataFrame
        db_path: Path to SQLite database file
        year: Data year
        quarter: Data quarter
        replace: If True, replace existing data; otherwise append
    """
    conn = sqlite3.connect(db_path)
    
    #added period columns for filtering
    hub_df = hub_df.copy()
    route_df = route_df.copy()
    hub_df["Year"] = year
    hub_df["Quarter"] = quarter
    route_df["Year"] = year
    route_df["Quarter"] = quarter
    
    if_exists = "replace" if replace else "append"
    
    hub_df.to_sql("hub_airline", conn, if_exists=if_exists, index=False)
    route_df.to_sql("route_airline", conn, if_exists=if_exists, index=False)
    
    conn.close()
    print(f"[saved] SQLite: {db_path} (hub: {len(hub_df):,} rows, route: {len(route_df):,} rows)")


def period_tag(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Override default CSV path")
    ap.add_argument("--uploads_dir", type=str, default=str(UPLOADS_DIR), help="Folder used to auto-find raw files by --year/--quarter")
    ap.add_argument("--fare_lower_bound", type=float, default=fare_lower_bound)
    ap.add_argument("--fare_upper_bound", type=float, default=fare_upper_bound)
    ap.add_argument("--fare_bin_width", type=float, default=fare_bin_width, help="Bucket width for specific fare-distribution cache exports.")
    ap.add_argument(
        "--min_carrier_total_passengers",
        type=float,
        default=min_carrier_total_passengers,
        help="Drop carriers whose total passengers for the full period are below this threshold.",
    )

    # if file contains multiple periods
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--quarter", type=int, default=None)

    ap.add_argument("--chunksize", type=int, default=750_000)
    ap.add_argument("--verbose", type=int, default=1)
    
    # Export options
    # Parquet is now the default output format for downstream navigation/storage.
    ap.add_argument("--export_parquet", dest="export_parquet", action="store_true", default=True, help="Export outputs to Parquet (default: on)")
    ap.add_argument("--no_export_parquet", dest="export_parquet", action="store_false", help="Disable Parquet export")
    ap.add_argument("--export_csv", action="store_true", help="Also export legacy CSV outputs")
    ap.add_argument("--export_sqlite", type=str, default=None, help="Export to SQLite database (provide path)")
    ap.add_argument("--quality_report", action="store_true", help="Generate data quality report")
    ap.add_argument(
        "--delete_raw_csv",
        action="store_true",
        help="Delete the raw CSV after successful parse/analyze outputs are written (only if it is inside uploads/).",
    )

    args = ap.parse_args()


    # Resolve raw input path. If --csv is omitted, auto-find in uploads using --year/--quarter.
    csv_path = resolve_csv_path(
        csv_arg=args.csv,
        year=args.year,
        quarter=args.quarter,
        uploads_dir=Path(args.uploads_dir),
    )
    ensure_output_dirs()

    print(f"[main] using CSV file: {csv_path}")

    year, quarter, hub_airline, route_airline, route_fare_distribution, total_seen, total_kept = ingest(
        csv_path=str(csv_path),
        fare_lower_bound=args.fare_lower_bound,
        fare_upper_bound=args.fare_upper_bound,
        fare_bin_width=args.fare_bin_width,
        year=args.year,
        quarter=args.quarter,
        chunksize=args.chunksize,
        verbose=args.verbose,
    )

    # Drop carriers using period-level totals, not per-route/per-hub group totals.
    carrier_totals: Dict[str, float] = defaultdict(float)
    for (_, _, carrier), agg in route_airline.items():
        carrier_totals[carrier] += agg.passengers_sum
    keep_carriers = {
        carrier
        for carrier, total in carrier_totals.items()
        if total >= float(args.min_carrier_total_passengers) and not is_invalid_carrier(carrier)
    }
    hub_airline = {k: v for k, v in hub_airline.items() if k[2] in keep_carriers}
    route_airline = {k: v for k, v in route_airline.items() if k[2] in keep_carriers}
    route_fare_distribution = {k: v for k, v in route_fare_distribution.items() if k[2] in keep_carriers}


    tag = period_tag(year, quarter)
    hub_df = hub_airline_table(hub_airline)
    route_df = route_airline_table(route_airline)
    fare_distribution_df = route_fare_distribution_table(route_fare_distribution)

    # Parquet exports (default)
    hub_out = HUB_AIRLINE_DIR / f"hubxairline_{tag}.parquet"
    route_out = ROUTE_AIRLINE_DIR / f"routexairline_{tag}.parquet"
    fare_distribution_out = SPECIFIC_FARE_DISTRIBUTION_DIR / f"specific_fare_distribution_{tag}.parquet"

    print("\n=== HUB × AIRLINE (Origin hub only; no layover hubs) ===")
    print(hub_df.head(50).to_string(index=False))  #bug test preview
    if args.export_parquet:
        export_to_parquet(hub_df, str(hub_out))

    print("\n=== ROUTE × AIRLINE (for later HHI / markup proxies) ===")
    print(route_df.head(50).to_string(index=False))  #bug test preview
    if args.export_parquet:
        export_to_parquet(route_df, str(route_out))

    print("\n=== SPECIFIC FARE DISTRIBUTION CHARTS CACHE ===")
    if args.export_parquet:
        export_to_parquet(fare_distribution_df, str(fare_distribution_out))

    # Optional legacy CSV export
    if args.export_csv:
        legacy_hub_out = HUB_AIRLINE_DIR / f"hubxairline_{tag}.csv"
        legacy_route_out = ROUTE_AIRLINE_DIR / f"routexairline_{tag}.csv"
        legacy_fare_out = SPECIFIC_FARE_DISTRIBUTION_DIR / f"specific_fare_distribution_{tag}.csv"
        hub_df.to_csv(legacy_hub_out, index=False)
        route_df.to_csv(legacy_route_out, index=False)
        fare_distribution_df.to_csv(legacy_fare_out, index=False)
        print(f"[saved] {legacy_hub_out} ({len(hub_df):,} rows)")
        print(f"[saved] {legacy_route_out} ({len(route_df):,} rows)")
        print(f"[saved] {legacy_fare_out} ({len(fare_distribution_df):,} rows)")
    
    # Optional SQLite export
    if args.export_sqlite:
        print("\n=== EXPORTING TO SQLITE ===")
        export_to_sqlite(hub_df, route_df, args.export_sqlite, year, quarter)
    
    # Optional quality report
    if args.quality_report:
        print("\n=== GENERATING QUALITY REPORT ===")
        report = generate_quality_report(
            year=year,
            quarter=quarter,
            total_seen=total_seen,
            total_kept=total_kept,
            hub_airline_count=len(hub_airline),
            route_airline_count=len(route_airline),
            fare_lower=args.fare_lower_bound,
            fare_upper=args.fare_upper_bound,
        )
        report_path = f"quality_report_{tag}.json"
        save_quality_report(report, report_path)
        print(f"\nRetention Rate: {report['ingestion']['retention_rate']}%")

    # Optional: delete raw CSV to save space (only if file is inside uploads/)
    if args.delete_raw_csv:
        try:
            uploads_root = Path(args.uploads_dir).resolve()
            csv_path_resolved = Path(csv_path).resolve()
            if uploads_root in csv_path_resolved.parents:
                csv_path_resolved.unlink(missing_ok=True)
                print(f"[cleanup] deleted raw CSV: {csv_path_resolved}")
            else:
                print(f"[cleanup] skipped (raw CSV is outside uploads/): {csv_path_resolved}")
        except Exception as exc:
            print(f"[cleanup] failed to delete raw CSV: {exc}")

    print(f"\n[info] period used: Year={year}, Quarter={quarter}")


if __name__ == "__main__":
    main()
