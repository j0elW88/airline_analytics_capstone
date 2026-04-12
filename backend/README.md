# Backend Notes (Airline Analytics)

Hi, this is the backend data pipeline for the app.

Short version: raw DB1B market CSV goes in, cleaned/aggregated market power outputs come out.

## Quick Pipeline

1. "capstone_parse.py" reads one raw DB1B CSV in chunks.
2. It builds:
   - "Hub x Airline" aggregates
   - "Route x Airline" aggregates
   - in-memory/on-demand route fare-bin payloads for route-specific fare variation charts
3. "capstone_analyze.py" reads those parse outputs and computes shares + HHI.
4. Frontend dev server reads analyzer outputs for app screens.

## Current File Format Rules

- Raw import: **CSV** (stays this way)
- Generated navigation/storage outputs: **Parquet by default**
- Legacy CSV export: optional ("--export_csv")

## Main Scripts

### "capstone_parse.py"

Purpose:
- Ingest raw DB1B rows
- Clean/filter data
- Aggregate weighted stats by hub and route
- Build fare bins for route deep-dive charts (served on-demand; not exported to a dedicated folder)

Important defaults:
- fare lower bound: "50"
- fare upper bound: "1200"
- carrier min passengers (period total): "1000"
- fare bin width: "5"
- invalid carrier codes dropped from final carrier outputs: "99", "00", empty/null-like codes

Outputs:
- "hubxairline_folder/hubxairline_<YEAR>_Q<QUARTER>.parquet"
- "routexairline_folder/routexairline_<YEAR>_Q<QUARTER>.parquet"

### "capstone_analyze.py"

Purpose:
- Read parse outputs (Parquet first, CSV fallback)
- Compute market shares + HHI for route and hub markets

Outputs:
- "routeMP_folder/route_market_power_<YEAR>_Q<QUARTER>.parquet"
- "hubMP_folder/hub_market_power_<YEAR>_Q<QUARTER>.parquet"

### "carrier_codes.py"

Purpose:
- Carrier code to name mapping used by backend and frontend display.

## Required Raw DB1B Columns

- "Year"
- "Quarter"
- "Origin"
- "OriginState"
- "Dest"
- "TkCarrier"
- "Passengers"
- "MktFare"
- "NonStopMiles"

## How Core Metrics Are Calculated

Weighted values:
- "avg_fare_weighted = sum(MktFare * Passengers) / sum(Passengers)"
- "avg_distance_weighted = sum(NonStopMiles * Passengers) / sum(Passengers)"

Route market power ("m = Origin, Dest"):
- "Q_m_valid = sum(passengers by valid carriers)"
- "share_im = Q_im / Q_m_valid"
- "HHI_m = sum(share_im^2) * 10000"

Hub market power ("h = Origin, OriginState"):
- "Q_h_valid = sum(passengers by valid carriers)"
- "share_ih = Q_ih / Q_h_valid"
- "HHI_h = sum(share_ih^2) * 10000"

Note:
- Baseline average/min fares use all carriers.
- Invalid carriers are excluded from final share/HHI output rows.

## Folders

- "uploads/" raw DB1B files
- "hubxairline_folder/" parse hub outputs
- "routexairline_folder/" parse route outputs
- "routeMP_folder/" analyzed route market power
- "hubMP_folder/" analyzed hub market power

## Run It

Install core dependencies:

"""powershell
py -m pip install pandas pyarrow
"""

Parse a raw file directly:

"""powershell
py backend/capstone_parse.py --csv backend/uploads/Origin_and_Destination_Survey_DB1BMarket_2025_1.csv --verbose 1
"""

Parse by period from "uploads/":

"""powershell
py backend/capstone_parse.py --year 2025 --quarter 1 --verbose 1
"""

Analyze that period:

"""powershell
py backend/capstone_analyze.py --year 2025 --quarter 1 --verbose 1
"""

If you need legacy CSV outputs too:

"""powershell
py backend/capstone_parse.py --year 2025 --quarter 1 --export_csv
py backend/capstone_analyze.py --year 2025 --quarter 1 --export_csv
"""

## One Important Gotcha

Frontend auto-detect now expects Parquet market-power files.
If an older period only has CSV outputs, rerun parse/analyze for that period to generate Parquet.
