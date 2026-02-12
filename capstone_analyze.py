import argparse
import json
import sqlite3
from pathlib import Path
from typing import Tuple, Set

import pandas as pd


# ----------------------------
# Config / tokens
# ----------------------------

INVALID_CARRIERS: Set[str] = {
    "99", "00", "", "nan", "none", "null", "unknown", "unk"
}
INVALID_CARRIERS_LC = {t.lower() for t in INVALID_CARRIERS}

# Columns your parse outputs already produce
ROUTE_REQUIRED = {
    "Origin", "Dest", "Carrier",
    "avg_fare_weighted", "avg_distance_weighted",
    "total_passengers", "row_count"
}
HUB_REQUIRED = {
    "Origin", "OriginState", "Carrier",
    "avg_fare_weighted", "avg_distance_weighted",
    "total_passengers", "row_count"
}


# ----------------------------
# Helpers
# ----------------------------

def period_tag(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"


def find_outputs(year: int, quarter: int, directory: str = ".") -> Tuple[str, str]:
    tag = period_tag(year, quarter)
    hub = Path(directory) / f"hubxairline_{tag}.csv"
    route = Path(directory) / f"routexairline_{tag}.csv"
    if not hub.exists():
        raise FileNotFoundError(f"Missing file: {hub}")
    if not route.exists():
        raise FileNotFoundError(f"Missing file: {route}")
    return str(hub), str(route)


def _assert_cols(df: pd.DataFrame, required: set, label: str) -> None:
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def normalize_carrier_col(df: pd.DataFrame, carrier_col: str = "Carrier") -> pd.DataFrame:
    out = df.copy()
    out[carrier_col] = out[carrier_col].astype(str).str.strip()
    return out


def split_valid_invalid(df: pd.DataFrame, carrier_col: str = "Carrier") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns: (valid_carriers_df, invalid_carriers_df)

    Invalid carriers are retained ONLY for baseline pricing (route/hub avg + min fare),
    but excluded from market power (shares / HHI) and excluded from output rows.
    """
    out = normalize_carrier_col(df, carrier_col=carrier_col)
    carrier_lc = out[carrier_col].astype(str).str.strip().str.lower()
    mask_invalid = carrier_lc.isin(INVALID_CARRIERS_LC)

    invalid = out[mask_invalid].copy()
    valid = out[~mask_invalid].copy()
    return valid, invalid


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: pd.NA})
    return num / den


# ----------------------------
# Core computations (with equations)
# ----------------------------

def compute_route_market_power(route_df_all: pd.DataFrame, min_market_passengers: float = 0.0, verbose: int = 1) -> pd.DataFrame:
    """
    Output rows: one per (Origin, Dest, Carrier) for VALID carriers only.

    Equations (for each route market m = (o,d)):
      Q_m_all   = sum_k Q_km                     (includes invalid)
      Q_m_valid = sum_{i in valid} Q_im          (valid-only)

      share_im = Q_im / Q_m_valid                (valid-only)
      HHI_m    = (sum_{i in valid} share_im^2) * 10000

      Pbar_m_all = (sum_k P_km * Q_km) / Q_m_all (includes invalid)
      Pmin_m_all = min_k P_km                    (includes invalid)

      markup_im = P_im - Pbar_m_all
      lerner_proxy_im = (P_im - Pmin_m_all) / P_im
    """
    _assert_cols(route_df_all, ROUTE_REQUIRED, "route_df")

    df = route_df_all.copy()
    for c in ["avg_fare_weighted", "avg_distance_weighted", "total_passengers", "row_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Origin", "Dest", "Carrier", "avg_fare_weighted", "avg_distance_weighted", "total_passengers"])
    df = df[
        (df["total_passengers"] > 0) &
        (df["avg_fare_weighted"] > 0) &
        (df["avg_distance_weighted"] > 0)
    ].copy()

    valid, invalid = split_valid_invalid(df, carrier_col="Carrier")

    if verbose:
        print(f"[route] rows: all={len(df):,} valid={len(valid):,} invalid={len(invalid):,}")
        print(f"[route] carriers: all={df['Carrier'].nunique():,} valid={valid['Carrier'].nunique():,} invalid={invalid['Carrier'].nunique():,}")

    # Baselines include invalid carriers
    df["fare_x_passengers"] = df["avg_fare_weighted"] * df["total_passengers"]

    market_all = df.groupby(["Origin", "Dest"], as_index=False).agg(
        route_total_passengers_all=("total_passengers", "sum"),
        route_fare_x_passengers_all=("fare_x_passengers", "sum"),
        route_min_fare_all=("avg_fare_weighted", "min"),
        carriers_on_route_all=("Carrier", "nunique"),
    )
    market_all["route_avg_fare_all"] = market_all["route_fare_x_passengers_all"] / market_all["route_total_passengers_all"]

    # Denominator for shares is valid-only
    market_valid = valid.groupby(["Origin", "Dest"], as_index=False).agg(
        route_total_passengers_valid=("total_passengers", "sum"),
        carriers_on_route_valid=("Carrier", "nunique"),
    )

    m = valid.merge(market_all, on=["Origin", "Dest"], how="left").merge(market_valid, on=["Origin", "Dest"], how="left")

    if min_market_passengers and min_market_passengers > 0:
        before = len(m)
        m = m[m["route_total_passengers_all"] >= float(min_market_passengers)].copy()
        if verbose:
            print(f"[route] min_market_passengers={min_market_passengers}: removed {before - len(m):,} rows")

    # share_im = Q_im / Q_m_valid
    m["route_share"] = safe_div(m["total_passengers"], m["route_total_passengers_valid"]).astype(float)

    # HHI_m = (sum_i share_im^2) * 10000 (valid-only)
    m["route_share_sq"] = m["route_share"] ** 2
    hhi = m.groupby(["Origin", "Dest"], as_index=False).agg(route_HHI=("route_share_sq", "sum"))
    hhi["route_HHI"] = hhi["route_HHI"] * 10000.0
    m = m.merge(hhi, on=["Origin", "Dest"], how="left").drop(columns=["route_share_sq"])

    # markup_im = P_im - Pbar_m_all
    m["markup_proxy_vs_route_avg"] = m["avg_fare_weighted"] - m["route_avg_fare_all"]

    # lerner_proxy = (P_im - Pmin_m_all) / P_im
    m["lerner_proxy_vs_route_min"] = safe_div(
        (m["avg_fare_weighted"] - m["route_min_fare_all"]),
        m["avg_fare_weighted"]
    ).astype(float)

    keep = [
        "Origin", "Dest", "Carrier",
        "total_passengers", "row_count",
        "avg_fare_weighted", "avg_distance_weighted",
        "route_total_passengers_all", "route_total_passengers_valid",
        "carriers_on_route_all", "carriers_on_route_valid",
        "route_share", "route_HHI",
        "route_avg_fare_all", "route_min_fare_all",
        "markup_proxy_vs_route_avg",
        "lerner_proxy_vs_route_min",
    ]
    return m[keep].sort_values(["Origin", "Dest", "Carrier"]).reset_index(drop=True)


def compute_hub_market_power(hub_df_all: pd.DataFrame, min_market_passengers: float = 0.0, verbose: int = 1) -> pd.DataFrame:
    """
    Output rows: one per (Origin, OriginState, Carrier) for VALID carriers only.

    Equations (for each hub market m = (Origin, OriginState)):
      Q_m_all   = sum_k Q_km                     (includes invalid)
      Q_m_valid = sum_{i in valid} Q_im          (valid-only)

      share_im = Q_im / Q_m_valid                (valid-only)
      HHI_m    = (sum_{i in valid} share_im^2) * 10000

      Pbar_m_all = (sum_k P_km * Q_km) / Q_m_all (includes invalid)
      Pmin_m_all = min_k P_km                    (includes invalid)

      markup_im = P_im - Pbar_m_all
      lerner_proxy_im = (P_im - Pmin_m_all) / P_im
    """
    _assert_cols(hub_df_all, HUB_REQUIRED, "hub_df")

    df = hub_df_all.copy()
    for c in ["avg_fare_weighted", "avg_distance_weighted", "total_passengers", "row_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Origin", "OriginState", "Carrier", "avg_fare_weighted", "avg_distance_weighted", "total_passengers"])
    df = df[
        (df["total_passengers"] > 0) &
        (df["avg_fare_weighted"] > 0) &
        (df["avg_distance_weighted"] > 0)
    ].copy()

    valid, invalid = split_valid_invalid(df, carrier_col="Carrier")

    if verbose:
        print(f"[hub] rows: all={len(df):,} valid={len(valid):,} invalid={len(invalid):,}")
        print(f"[hub] carriers: all={df['Carrier'].nunique():,} valid={valid['Carrier'].nunique():,} invalid={invalid['Carrier'].nunique():,}")

    # Baselines include invalid carriers
    df["fare_x_passengers"] = df["avg_fare_weighted"] * df["total_passengers"]

    market_all = df.groupby(["Origin", "OriginState"], as_index=False).agg(
        hub_total_passengers_all=("total_passengers", "sum"),
        hub_fare_x_passengers_all=("fare_x_passengers", "sum"),
        hub_min_fare_all=("avg_fare_weighted", "min"),
        carriers_at_hub_all=("Carrier", "nunique"),
    )
    market_all["hub_avg_fare_all"] = market_all["hub_fare_x_passengers_all"] / market_all["hub_total_passengers_all"]

    market_valid = valid.groupby(["Origin", "OriginState"], as_index=False).agg(
        hub_total_passengers_valid=("total_passengers", "sum"),
        carriers_at_hub_valid=("Carrier", "nunique"),
    )

    m = valid.merge(market_all, on=["Origin", "OriginState"], how="left").merge(market_valid, on=["Origin", "OriginState"], how="left")

    if min_market_passengers and min_market_passengers > 0:
        before = len(m)
        m = m[m["hub_total_passengers_all"] >= float(min_market_passengers)].copy()
        if verbose:
            print(f"[hub] min_market_passengers={min_market_passengers}: removed {before - len(m):,} rows")

    # share_im = Q_im / Q_m_valid
    m["hub_share"] = safe_div(m["total_passengers"], m["hub_total_passengers_valid"]).astype(float)

    # HHI_m = sum_i share_im^2 * 10000 (valid-only)
    m["hub_share_sq"] = m["hub_share"] ** 2
    hhi = m.groupby(["Origin", "OriginState"], as_index=False).agg(hub_HHI=("hub_share_sq", "sum"))
    hhi["hub_HHI"] = hhi["hub_HHI"] * 10000.0
    m = m.merge(hhi, on=["Origin", "OriginState"], how="left").drop(columns=["hub_share_sq"])

    # markup_im = P_im - Pbar_m_all
    m["markup_proxy_vs_hub_avg"] = m["avg_fare_weighted"] - m["hub_avg_fare_all"]

    # lerner_proxy = (P_im - Pmin_m_all)/P_im
    m["lerner_proxy_vs_hub_min"] = safe_div(
        (m["avg_fare_weighted"] - m["hub_min_fare_all"]),
        m["avg_fare_weighted"]
    ).astype(float)

    keep = [
        "Origin", "OriginState", "Carrier",
        "total_passengers", "row_count",
        "avg_fare_weighted", "avg_distance_weighted",
        "hub_total_passengers_all", "hub_total_passengers_valid",
        "carriers_at_hub_all", "carriers_at_hub_valid",
        "hub_share", "hub_HHI",
        "hub_avg_fare_all", "hub_min_fare_all",
        "markup_proxy_vs_hub_avg",
        "lerner_proxy_vs_hub_min",
    ]
    return m[keep].sort_values(["Origin", "OriginState", "Carrier"]).reset_index(drop=True)


# ----------------------------
# Export helpers
# ----------------------------

def export_to_parquet(df: pd.DataFrame, output_path: str):
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    print(f"[saved] Parquet: {output_path} ({len(df):,} rows)")


def export_to_sqlite_market_power(
    route_power: pd.DataFrame,
    hub_power: pd.DataFrame,
    db_path: str,
    year: int,
    quarter: int,
    replace: bool = False,
):
    conn = sqlite3.connect(db_path)

    rp = route_power.copy()
    hp = hub_power.copy()
    rp["Year"] = year
    rp["Quarter"] = quarter
    hp["Year"] = year
    hp["Quarter"] = quarter

    if_exists = "replace" if replace else "append"
    rp.to_sql("route_market_power", conn, if_exists=if_exists, index=False)
    hp.to_sql("hub_market_power", conn, if_exists=if_exists, index=False)

    conn.close()
    print(f"[saved] SQLite: {db_path} (route_market_power: {len(rp):,} rows, hub_market_power: {len(hp):,} rows)")


def save_quality_report(report: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] Quality report: {output_path}")


def generate_quality_report(
    year: int,
    quarter: int,
    route_rows_in: int,
    hub_rows_in: int,
    route_rows_out: int,
    hub_rows_out: int,
    invalid_route_rows: int,
    invalid_hub_rows: int,
) -> dict:
    return {
        "period": {"year": year, "quarter": quarter},
        "inputs": {
            "route_rows_in": route_rows_in,
            "hub_rows_in": hub_rows_in,
        },
        "invalid_carrier_rows": {
            "route_invalid_rows": invalid_route_rows,
            "hub_invalid_rows": invalid_hub_rows,
        },
        "outputs": {
            "route_market_power_rows": route_rows_out,
            "hub_market_power_rows": hub_rows_out,
        },
    }


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--quarter", type=int, required=True)
    ap.add_argument("--dir", type=str, default=".")
    ap.add_argument("--min_market_passengers", type=float, default=0.0, help="Optional threshold per market (uses ALL passengers, includes invalid)")
    ap.add_argument("--verbose", type=int, default=1)

    # Base outputs
    ap.add_argument("--export_csv", action="store_true", help="Export route_market_power + hub_market_power CSVs")

    # Optional exports
    ap.add_argument("--export_parquet", action="store_true", help="Export route/hub market power Parquet too")
    ap.add_argument("--export_sqlite", type=str, default=None, help="Export both tables to SQLite db path")
    ap.add_argument("--replace", action="store_true", help="SQLite: replace instead of append")
    ap.add_argument("--quality_report", action="store_true", help="Export a small JSON quality report")

    args = ap.parse_args()

    hub_path, route_path = find_outputs(args.year, args.quarter, args.dir)
    print(f"[analyze] using hub:   {hub_path}")
    print(f"[analyze] using route: {route_path}")

    hub_df = pd.read_csv(hub_path)
    route_df = pd.read_csv(route_path)

    route_rows_in = len(route_df)
    hub_rows_in = len(hub_df)

    # Count invalid rows for reporting (but keep them in baselines)
    _, route_invalid = split_valid_invalid(route_df, carrier_col="Carrier")
    _, hub_invalid = split_valid_invalid(hub_df, carrier_col="Carrier")

    route_power = compute_route_market_power(route_df, min_market_passengers=args.min_market_passengers, verbose=args.verbose)
    hub_power = compute_hub_market_power(hub_df, min_market_passengers=args.min_market_passengers, verbose=args.verbose)

    tag = period_tag(args.year, args.quarter)

    # Preview
    print("\n=== ROUTE MARKET POWER (preview 20) ===")
    print(route_power.head(20).to_string(index=False))
    print("\n=== HUB MARKET POWER (preview 20) ===")
    print(hub_power.head(20).to_string(index=False))

    # CSV outputs
    if args.export_csv:
        out_route = Path(args.dir) / f"route_market_power_{tag}.csv"
        out_hub = Path(args.dir) / f"hub_market_power_{tag}.csv"
        route_power.to_csv(out_route, index=False)
        hub_power.to_csv(out_hub, index=False)
        print(f"\n[saved] {out_route} ({len(route_power):,} rows)")
        print(f"[saved] {out_hub} ({len(hub_power):,} rows)")

    # Parquet outputs
    if args.export_parquet:
        out_route_p = Path(args.dir) / f"route_market_power_{tag}.parquet"
        out_hub_p = Path(args.dir) / f"hub_market_power_{tag}.parquet"
        export_to_parquet(route_power, str(out_route_p))
        export_to_parquet(hub_power, str(out_hub_p))

    # SQLite outputs
    if args.export_sqlite:
        export_to_sqlite_market_power(
            route_power=route_power,
            hub_power=hub_power,
            db_path=args.export_sqlite,
            year=args.year,
            quarter=args.quarter,
            replace=args.replace,
        )

    # Quality report JSON
    if args.quality_report:
        report = generate_quality_report(
            year=args.year,
            quarter=args.quarter,
            route_rows_in=route_rows_in,
            hub_rows_in=hub_rows_in,
            route_rows_out=len(route_power),
            hub_rows_out=len(hub_power),
            invalid_route_rows=len(route_invalid),
            invalid_hub_rows=len(hub_invalid),
        )
        out_json = Path(args.dir) / f"analysis_quality_report_{tag}.json"
        save_quality_report(report, str(out_json))


if __name__ == "__main__":
    main()
