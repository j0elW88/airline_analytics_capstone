# Frontend Notes (React + TypeScript)

This frontend is the React replacement for the old Streamlit flow.

Tone of this app: practical analytics dashboard, fast local workflow, easy to extend.

## What You Can Do In The UI

- Load data from existing backend periods.
- Upload one raw DB1B CSV and let backend parse + analyze automatically.
- Analyze one period or multiple periods.
- Use route/hub filters and see market overview + insights.
- In route-specific scope (`Origin + Dest` selected), load carrier-level fare variation charts.

## Run

```powershell
cd "c:\Users\woest\Desktop\Soleus App\airline_analytics_capstone\frontend"
npm install
npm run dev
```

Build:

```powershell
npm run build
npm run preview
```

## Stack

- React 18
- TypeScript
- Vite

## Navigation Flow

- `Home` -> `History` / `Loaded Data Sets` / `Start`
- `Start` -> `Analyze One` / `Analyze Multiple` / `Load Data Set`
- `Load Data Set`
  - Upload raw DB1B CSV (backend pipeline runs automatically)
  - Or import an already-generated backend period
- `Analyze One` -> `Results (single)`
- `Analyze Multiple` -> `Results (multi)`

## Results Tabs

Current result tabs:
- `Market Overview`
- `Route & Hub Insights`

## Scope Behavior (Important)

Route tab behavior changes by filter scope:

- General scope (no specific route selected):
  - Shows standard fare distribution histogram from route market-power rows.

- Specific route scope (`Origin + Dest` selected):
  - Replaces standard fare distribution with route fare variation charts.
  - Data is loaded on-demand with a button.
  - Shows loading/error states.

- Specific route + specific carrier:
  - Shows the selected carrier's fare variation for that route.

- Specific route with no carrier selected:
  - Shows one chart per carrier on that route.

Also:
- Carrier count in route snapshot includes a tooltip listing contributing carriers.
- Market share by carrier + average fare by carrier include all carriers.
- Carrier shares under 1% are displayed as `< 1%`.

## Data + Local Dev Bridge

`vite.config.ts` provides local API endpoints used by the app:

- `GET /api/local/periods`
  - Returns periods that exist in both backend market-power folders.
- `GET /api/local/carriers`
  - Returns backend carrier lookup map.
- `GET /api/local/dataset?period=YYYY_Q#`
  - Returns route/hub market-power rows for one period.
- `GET /api/local/fare-distribution?period=...&origin=...&dest=...[&carrier=...]`
  - Returns route-specific fare-bin payload.
- `POST /api/local/import-raw?filename=...`
  - Saves uploaded raw CSV to backend `uploads/` and runs parse + analyze.

## Current Storage Conventions

- Raw import remains CSV.
- Backend-generated navigation datasets are Parquet-first:
  - `backend/routeMP_folder/route_market_power_<period>.parquet`
  - `backend/hubMP_folder/hub_market_power_<period>.parquet`
- Route fare distribution for charts is generated on-demand by `capstone_parse.py` and held in runtime memory.

## Folder Guide

- `src/app/`
  - App shell, router, global state.
- `src/pages/`
  - Screen-level pages.
- `src/features/results/`
  - Analytics transforms + results panels.
- `src/components/`
  - Reusable UI components/charts/filters/layout.
- `src/services/`
  - Local backend API bridge wrappers.
- `src/utils/`
  - Carrier formatting, number formatting, helpers.
- `src/types/`
  - Shared TypeScript contracts.
- `src/styles/`
  - App styles and chart/theme variables.

## Dev Notes

- On startup, app attempts to auto-import locally available periods from backend folders.
- If local backend bridge is unavailable, UI still loads but period auto-discovery/import features will fail.
- For old periods that only have CSV outputs, regenerate Parquet backend outputs so they appear in period auto-detection.
