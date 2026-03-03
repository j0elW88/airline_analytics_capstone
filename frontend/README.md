# Frontend (React Migration)

This frontend is now a modular React + TypeScript app (Vite) built to replace the monolithic Streamlit `app.py` workflow.

## Goals of This Structure

- Split pages, reusable widgets, and analytics logic into separate modules.
- Keep UI components reusable (`buttons`, `cards`, `modals`, `tables`, `tabs`, filters).
- Keep workflow behavior consistent while making the codebase easier for multiple developers to maintain.
- Keep backend untouched for this phase.

## Run

```powershell
cd "c:\Users\woest\Desktop\Soleus App\airline_analytics_capstone\frontend"
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

## Build

```powershell
npm run build
npm run preview
```

## Requirements

- Active UI stack:
  - Node.js 18+ (recommended) and npm
  - Frontend packages from `package.json`
- Python requirements:
  - Not required for normal React UI rendering.
  - `frontend/requirements.txt` is kept for optional legacy `frontend/app.py` usage only.
- Local backend bridge (dev convenience):
  - `vite.config.ts` exposes local endpoints used to discover/import backend datasets during development.

## Folder Map

- `src/app/`
  - Global state, navigation stack, screen routing shell.
- `src/pages/`
  - Page modules (`home`, `start`, `load`, `history`, `analyze`, `results`).
- `src/components/ui/`
  - Reusable UI primitives (`AppButton`, `Card`, `MetricCard`, `DataTable`, `Tabs`, `ModalHost`, `EmptyState`).
- `src/components/filters/`
  - Shared filter bars for route/hub contexts.
- `src/components/charts/`
  - Reusable chart display components.
- `src/features/results/`
  - Analytics calculations and result tab composition.
- `src/services/`
  - CSV parsing/loading helpers.
- `src/types/`
  - Shared TypeScript contracts.
- `src/styles/`
  - Theme tokens and app-wide styles.
- `public/assets/`
  - Static assets (including landing-only sky hero SVG).

## Code Documentation

- Every source file in `src/` now includes a file header (`@file`, `@description`) describing its role.
- Core workflow files include additional inline comments for:
  - app bootstrapping and screen routing
  - global state reducer + persistence decisions
  - dataset import bridge and CSV parsing
  - analytics aggregation and results rendering paths
- Goal:
  - make onboarding easier for other developers,
  - clarify where to add or modify behavior without tracing the full app at runtime.

## Current Workflow Behavior

- Home -> History / Loaded Data Sets / Start.
- Start -> Analyze One / Analyze Multiple / Load Data Set.
- On app startup in `npm run dev`, frontend auto-detects complete periods from:
  - `backend/routeMP_folder/route_market_power_<period>.csv`
  - `backend/hubMP_folder/hub_market_power_<period>.csv`
  and imports them into local frontend state.
- Load Data Set:
  - Primary path: upload one raw DB1B CSV and frontend dev bridge runs `capstone_parse.py` + `capstone_analyze.py` automatically.
  - No Year/Quarter input required in UI.
  - Existing generated periods can be selected and imported directly from backend folders.
  - Manual fallback still supports uploading `route_market_power` + `hub_market_power` CSV pair.
  - Success and failure messages use reusable modal system.
- Analyze One:
  - Select one loaded period and open analytics.
- Analyze Multiple:
  - Select one or more loaded periods and open aggregated analytics.
  - Compare prototype guidance is embedded directly in this screen.
- Results tabs (current phase):
  - `Market Overview`
  - `Route & Hub Insights`
- Removed from normal flow for now:
  - Time comparison tab
  - Capacity tab

## Theming Notes

- Neutral palette preserved where possible.
- Non-neutral accents shifted to blue tones per current design direction.
- Sky image treatment is limited to the landing/home dashboard.

## Data Notes

- Frontend expects market power CSV outputs (route + hub) and parses them client-side.
- Carrier display defaults to `carrier_name` when available, with `Carrier` code fallback.
- Auto-detection from backend folders is provided by a Vite dev middleware in `vite.config.ts` (frontend-only change).

## Legacy Reference

- Legacy Streamlit app is still present at `frontend/app.py` for reference during migration.
