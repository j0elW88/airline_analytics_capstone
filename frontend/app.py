"""
Airline Market Analysis Dashboard — frontend/app.py
Drop-in replacement for the existing Streamlit frontend.
Keeps all original data-loading / import logic intact.
Adds fully wired 4-frame analytics dashboard on the results screens.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Paths (unchanged from original) ─────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
BACKEND_ROOT    = PROJECT_ROOT / "backend"

HUB_MP_DIR      = BACKEND_ROOT / "hubMP_folder"
ROUTE_MP_DIR    = BACKEND_ROOT / "routeMP_folder"
HUB_AIRLINE_DIR = BACKEND_ROOT / "hubxairline_folder"
ROUTE_AIRLINE_DIR = BACKEND_ROOT / "routexairline_folder"
UPLOADS_DIR     = BACKEND_ROOT / "uploads"

PERIOD_PATTERN  = re.compile(r"_(\d{4})_Q([1-4])\.csv$", re.IGNORECASE)


# ─── Colour palette (matches Figma wireframe) ─────────────────────────────────
ACCENT      = "#1B3A6B"
ACCENT_MID  = "#2B5AA8"
CHART_COLORS = ["#1B3A6B", "#2B5AA8", "#4A7DC4", "#7BA3D8", "#A8C3E8", "#D0DFF4",
                "#6B3A1B", "#A85A2B"]


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def period_key(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"


def parse_period(filename: str) -> Optional[Tuple[int, int]]:
    match = PERIOD_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def init_dirs() -> None:
    for path in [HUB_MP_DIR, ROUTE_MP_DIR, HUB_AIRLINE_DIR, ROUTE_AIRLINE_DIR, UPLOADS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def run_backend(cmd: list) -> Tuple[bool, str]:
    result = subprocess.run(cmd, cwd=BACKEND_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, (result.stdout or "").strip()


def collect_periods() -> Dict[str, Dict[str, bool]]:
    periods: Dict[str, Dict[str, bool]] = {}
    required = {
        "hub_market_power":   HUB_MP_DIR,
        "route_market_power": ROUTE_MP_DIR,
        "hubxairline":        HUB_AIRLINE_DIR,
        "routexairline":      ROUTE_AIRLINE_DIR,
    }
    for key, folder in required.items():
        for file in folder.glob("*.csv"):
            parsed = parse_period(file.name)
            if not parsed:
                continue
            year, quarter = parsed
            pkey = period_key(year, quarter)
            if pkey not in periods:
                periods[pkey] = {k: False for k in required.keys()}
            periods[pkey][key] = True
    return periods


def is_complete(files: Dict[str, bool]) -> bool:
    return all(files.values())


def add_history(item: str) -> None:
    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:30]


def run_import(raw_csv_path: Path, year: int, quarter: int) -> Tuple[bool, str]:
    parse_cmd = [
        "py", "capstone_parse.py",
        "--csv", str(raw_csv_path),
        "--year", str(year),
        "--quarter", str(quarter),
        "--verbose", "0",
    ]
    ok_parse, msg_parse = run_backend(parse_cmd)
    if not ok_parse:
        return False, f"capstone_parse.py failed:\n{msg_parse}"

    analyze_cmd = [
        "py", "capstone_analyze.py",
        "--year", str(year),
        "--quarter", str(quarter),
        "--dir", ".",
        "--export_csv",
        "--verbose", "0",
    ]
    ok_analyze, msg_analyze = run_backend(analyze_cmd)
    if not ok_analyze:
        return False, f"capstone_analyze.py failed:\n{msg_analyze}"

    tag = period_key(year, quarter)
    map_move = {
        BACKEND_ROOT / f"hubxairline_{tag}.csv":       HUB_AIRLINE_DIR / f"hubxairline_{tag}.csv",
        BACKEND_ROOT / f"routexairline_{tag}.csv":     ROUTE_AIRLINE_DIR / f"routexairline_{tag}.csv",
        BACKEND_ROOT / f"hub_market_power_{tag}.csv":  HUB_MP_DIR / f"hub_market_power_{tag}.csv",
        BACKEND_ROOT / f"route_market_power_{tag}.csv": ROUTE_MP_DIR / f"route_market_power_{tag}.csv",
    }
    for src, dst in map_move.items():
        if src.exists():
            shutil.move(str(src), str(dst))

    return True, "Import and analysis complete."


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_route_mp(period: str) -> pd.DataFrame:
    """Load route_market_power CSV for a given period string like '2025_Q1'."""
    path = HUB_MP_DIR.parent / "routeMP_folder" / f"route_market_power_{period}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_hub_mp(period: str) -> pd.DataFrame:
    """Load hub_market_power CSV for a given period string like '2025_Q1'."""
    path = HUB_MP_DIR / f"hub_market_power_{period}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def apply_filters(df: pd.DataFrame, origin: str, dest: str,
                  carrier: str, region: str) -> pd.DataFrame:
    """Apply sidebar/filter-bar selections to a dataframe."""
    if df.empty:
        return df
    if origin and "Origin" in df.columns:
        df = df[df["Origin"] == origin]
    if dest and "Dest" in df.columns:
        df = df[df["Dest"] == dest]
    if carrier and "Carrier" in df.columns:
        df = df[df["Carrier"] == carrier]
    if region and "OriginState" in df.columns:
        # Region is stored as state code; treat as a loose contains match
        df = df[df["OriginState"].astype(str).str.contains(region, case=False, na=False)]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Shared UI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Force light mode everywhere ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
    .stApp, .main, section.main {
        background-color: #F4F6F8 !important;
        color: #1A1D23 !important;
    }

    /* Top toolbar (Deploy bar) */
    [data-testid="stToolbar"] { background: #FFFFFF !important; }
    header[data-testid="stHeader"] { background: #FFFFFF !important; border-bottom: 1px solid #D8DCE3 !important; }

    /* Remove default top padding, cap width */
    .block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

    /* ── All text forced light ── */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #1A1D23 !important;
    }

    /* ── Buttons — plain bordered style matching Figma ── */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #1A1D23 !important;
        border: 1px solid #D8DCE3 !important;
        border-radius: 5px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        box-shadow: none !important;
        transition: border-color 0.15s, background-color 0.15s !important;
    }
    .stButton > button:hover {
        background-color: #F0F4FF !important;
        border-color: #1B3A6B !important;
        color: #1B3A6B !important;
    }
    /* Primary buttons (type="primary") get navy fill */
    .stButton > button[kind="primary"] {
        background-color: #1B3A6B !important;
        color: #FFFFFF !important;
        border: 1px solid #1B3A6B !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2B5AA8 !important;
        border-color: #2B5AA8 !important;
        color: #FFFFFF !important;
    }

    /* ── Inputs, selectboxes, text inputs ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] > div > div > input,
    [data-testid="stNumberInput"] > div > div > input {
        background-color: #FFFFFF !important;
        color: #1A1D23 !important;
        border: 1px solid #D8DCE3 !important;
        border-radius: 4px !important;
    }
    [data-testid="stSelectbox"] svg { color: #6B7280 !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #D8DCE3 !important;
        border-radius: 6px !important;
        color: #1A1D23 !important;
    }
    [data-testid="stFileUploader"] span { color: #6B7280 !important; }

    /* Checkbox */
    [data-testid="stCheckbox"] span { color: #1A1D23 !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1px solid #D8DCE3 !important;
        border-radius: 6px !important;
        padding: 16px 18px 12px !important;
    }
    [data-testid="metric-container"] label {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: #6B7280 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #1A1D23 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* ── Section header ── */
    .section-header {
        background: #E5E7EB;
        border: 1px solid #D8DCE3;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 14px;
        font-size: 14px;
        font-weight: 700;
        color: #1A1D23;
    }

    /* ── Chart card wrappers ── */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid #D8DCE3;
        border-radius: 6px;
        padding: 16px 18px;
    }

    /* Give every st.container inside columns a card appearance */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background: #FFFFFF;
        border: 1px solid #D8DCE3;
        border-radius: 6px;
        padding: 16px 18px;
    }

    /* ── Caption ── */
    .caption-italic {
        font-size: 12px;
        font-style: italic;
        color: #6B7280;
        margin-top: 10px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF !important;
        border-bottom: 1px solid #D8DCE3 !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #6B7280 !important;
        padding: 10px 20px !important;
        border-radius: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        font-weight: 700 !important;
        color: #1B3A6B !important;
        border-bottom: 2px solid #1B3A6B !important;
        background: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #F4F6F8 !important;
        padding-top: 16px !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #D8DCE3 !important;
        border-radius: 6px !important;
        background: #FFFFFF !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #D8DCE3 !important;
    }

    /* ── Spinner / info / success / error boxes ── */
    [data-testid="stAlert"] { border-radius: 6px !important; }

    /* ── Selectbox / input labels ── */
    .stSelectbox label, .stMultiSelect label,
    .stTextInput label, .stNumberInput label {
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #6B7280 !important;
    }

    /* ── Number input buttons ── */
    [data-testid="stNumberInput"] button {
        background: #F4F6F8 !important;
        border: 1px solid #D8DCE3 !important;
        color: #1A1D23 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str) -> None:
    st.markdown(f"""
    <div style="padding-bottom:14px; border-bottom:1px solid #D8DCE3; margin-bottom:18px;">
        <h1 style="font-size:22px; font-weight:700; color:#1A1D23; margin:0; letter-spacing:-0.02em;">
            {title}
        </h1>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def card_header(title: str) -> None:
    """Bold title rendered inside a container block — replaces the broken chart-card div pattern."""
    st.markdown(
        f'<p style="font-size:13px;font-weight:700;color:#1A1D23;margin:0 0 10px 0">{title}</p>',
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    st.markdown(f'<div class="caption-italic">{text}</div>', unsafe_allow_html=True)


def filter_bar(key_prefix: str, route_df: pd.DataFrame, period: str = "") -> Tuple[str, str, str, str]:
    """
    Renders the 6-column filter bar.
    Year/Quarter show the selected period (read-only).
    Origin, Dest, Carrier, Region are active filters.
    Returns (origin, dest, carrier, region).
    """
    yr, qt = "—", "—"
    if period:
        parts = period.split("_Q")
        if len(parts) == 2:
            yr, qt = parts[0], f"Q{parts[1]}"

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.selectbox("Year", [yr], key=f"{key_prefix}_year", disabled=True)
    with c2:
        st.selectbox("Quarter", [qt], key=f"{key_prefix}_qtr", disabled=True)

    origins  = sorted(route_df["Origin"].dropna().unique().tolist())  if not route_df.empty and "Origin"  in route_df.columns else []
    dests    = sorted(route_df["Dest"].dropna().unique().tolist())    if not route_df.empty and "Dest"    in route_df.columns else []
    carriers = sorted(route_df["Carrier"].dropna().unique().tolist()) if not route_df.empty and "Carrier" in route_df.columns else []

    with c3:
        origin = st.selectbox("Origin Airport", [""] + origins,
                              key=f"{key_prefix}_origin",
                              format_func=lambda x: "Select Origin" if x == "" else x)
    with c4:
        dest = st.selectbox("Destination Airport", [""] + dests,
                            key=f"{key_prefix}_dest",
                            format_func=lambda x: "Select Destination" if x == "" else x)
    with c5:
        carrier = st.selectbox("Carrier (optional)", [""] + carriers,
                               key=f"{key_prefix}_carrier",
                               format_func=lambda x: "Select Carrier" if x == "" else x)
    with c6:
        region = st.text_input("Region (optional)", placeholder="e.g. FL",
                               key=f"{key_prefix}_region")

    st.write("")
    return origin, dest, carrier, region


def plotly_bar(df: pd.DataFrame, x: str, y: str, title: str,
               xlabel: str = "", ylabel: str = "", color: str = ACCENT) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color_discrete_sequence=[color])
    fig.update_layout(
        title=None, paper_bgcolor="#F4F6F8", plot_bgcolor="#F4F6F8",
        margin=dict(l=0, r=0, t=8, b=0), height=260,
        xaxis=dict(title=xlabel, showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(title=ylabel, showgrid=True, gridcolor="#E8ECF0", tickfont=dict(size=10)),
        showlegend=False,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def plotly_line(df: pd.DataFrame, x: str, y: str, color: str = ACCENT) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True, color_discrete_sequence=[color])
    fig.update_layout(
        title=None, paper_bgcolor="#F4F6F8", plot_bgcolor="#F4F6F8",
        margin=dict(l=0, r=0, t=8, b=0), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#E8ECF0", tickfont=dict(size=10)),
        showlegend=False,
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=5))
    return fig


def plotly_grouped_bar(df: pd.DataFrame, x: str, y_cols: List[str],
                       names: List[str]) -> go.Figure:
    fig = go.Figure()
    colors = [ACCENT, "#9CA3AF"]
    for col, name, color in zip(y_cols, names, colors):
        fig.add_trace(go.Bar(x=df[x], y=df[col], name=name,
                             marker_color=color, marker_line_width=0))
    fig.update_layout(
        barmode="group", paper_bgcolor="#F4F6F8", plot_bgcolor="#F4F6F8",
        margin=dict(l=0, r=0, t=8, b=0), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#E8ECF0", tickfont=dict(size=10)),
        legend=dict(font=dict(size=11), orientation="h", y=-0.15),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Frame 1 — Market Overview
# ═══════════════════════════════════════════════════════════════════════════════

def frame_market_overview(route_df: pd.DataFrame, hub_df: pd.DataFrame, period: str = "") -> None:
    origin, dest, carrier, region = filter_bar("f1", route_df, period)

    rdf = apply_filters(route_df.copy(), origin, dest, carrier, region)
    hdf = apply_filters(hub_df.copy(),   origin, "",   carrier, region)

    # ── KPI row ──────────────────────────────────────────────────────────────
    total_pax   = int(rdf["total_passengers"].sum()) if not rdf.empty else 0
    avg_fare    = (rdf["avg_fare_weighted"] * rdf["total_passengers"]).sum() / max(rdf["total_passengers"].sum(), 1) if not rdf.empty else 0
    n_carriers  = rdf["Carrier"].nunique() if not rdf.empty else 0
    avg_hhi     = rdf.groupby(["Origin", "Dest"])["route_HHI"].first().mean() if not rdf.empty and "route_HHI" in rdf.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Passengers",                f"{total_pax:,.0f}")
    k2.metric("Passenger-Weighted Avg Fare",     f"${avg_fare:,.0f}")
    k3.metric("Number of Carriers",              str(n_carriers))
    k4.metric("HHI (Market Concentration)",      f"{avg_hhi:,.0f}")

    st.write("")

    # ── Charts row ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:

        card_header("Market Share by Carrier")
        if not rdf.empty and "route_share" in rdf.columns:
            share_df = (
                rdf.groupby("Carrier")["total_passengers"].sum()
                .reset_index()
                .rename(columns={"total_passengers": "Passengers"})
                .sort_values("Passengers", ascending=False)
                .head(10)
            )
            share_df["Share %"] = (share_df["Passengers"] / share_df["Passengers"].sum() * 100).round(1)
            st.plotly_chart(plotly_bar(share_df, "Carrier", "Share %", "",
                                       ylabel="Share (%)"), use_container_width=True)
        else:
            st.info("No data for selected filters.")


    with col_right:

        card_header("Fare Comparison by Airline")
        if not rdf.empty:
            fare_df = (
                rdf.groupby("Carrier")
                .apply(lambda g: (g["avg_fare_weighted"] * g["total_passengers"]).sum() / g["total_passengers"].sum())
                .reset_index(name="Avg Fare ($)")
                .sort_values("Avg Fare ($)", ascending=False)
                .head(10)
            )
            st.plotly_chart(plotly_bar(fare_df, "Carrier", "Avg Fare ($)", "",
                                       ylabel="Avg Fare ($)", color=ACCENT_MID),
                            use_container_width=True)
        else:
            st.info("No data for selected filters.")


    caption("Filter-driven market overview.")


# ═══════════════════════════════════════════════════════════════════════════════
# Frame 2 — Route & Hub Insights
# ═══════════════════════════════════════════════════════════════════════════════

def frame_route_hub(route_df: pd.DataFrame, hub_df: pd.DataFrame, period: str = "") -> None:
    origin, dest, carrier, region = filter_bar("f2", route_df, period)
    rdf = apply_filters(route_df.copy(), origin, dest, carrier, region)
    hdf = apply_filters(hub_df.copy(),   origin, "",   carrier, region)

    col_left, col_right = st.columns(2)

    with col_left:

        card_header("Route or Hub Demand Rankings")
        if not rdf.empty:
            demand = (
                rdf.groupby(["Origin", "Dest"])["total_passengers"].sum()
                .reset_index()
                .sort_values("total_passengers", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            demand.index += 1
            demand.index.name = "Rank"
            demand["Route"] = demand["Origin"] + " → " + demand["Dest"]
            demand["Passengers"] = demand["total_passengers"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(demand[["Route", "Passengers"]], use_container_width=True)
        else:
            st.info("No route data for selected filters.")


    with col_right:

        card_header("High-Cost Routes Table")
        if not rdf.empty:
            high_cost = (
                rdf.groupby(["Origin", "Dest"])
                .apply(lambda g: (g["avg_fare_weighted"] * g["total_passengers"]).sum() / g["total_passengers"].sum())
                .reset_index(name="avg_fare")
                .sort_values("avg_fare", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            high_cost.index += 1
            high_cost.index.name = "Rank"
            high_cost["Route"]    = high_cost["Origin"] + " → " + high_cost["Dest"]
            high_cost["Avg Fare"] = high_cost["avg_fare"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(high_cost[["Route", "Avg Fare"]], use_container_width=True)
        else:
            st.info("No route data for selected filters.")


    caption("Minimum passenger threshold applied to reduce noise.")


# ═══════════════════════════════════════════════════════════════════════════════
# Frame 3 — Time Comparison  (multi-period)
# ═══════════════════════════════════════════════════════════════════════════════

def frame_time_comparison(periods: List[str]) -> None:
    # Load all selected periods
    all_route = []
    for p in periods:
        df = load_route_mp(p)
        if not df.empty:
            df["period"] = p
            all_route.append(df)

    combined = pd.concat(all_route, ignore_index=True) if all_route else pd.DataFrame()

    # Filter bar — use first period's data for dropdown population
    first_route = load_route_mp(periods[0]) if periods else pd.DataFrame()
    origin, dest, carrier, region = filter_bar("f3", first_route, periods[0] if periods else "")
    if not combined.empty:
        combined = apply_filters(combined, origin, dest, carrier, region)

    # ── Period A vs B KPIs ───────────────────────────────────────────────────
    section_header("Period A vs Period B Comparison")

    period_a = periods[0]  if len(periods) > 0 else None
    period_b = periods[-1] if len(periods) > 1 else None

    def period_stats(p: Optional[str]):
        if p is None or combined.empty:
            return 0.0, 0, 0.0
        sub = combined[combined["period"] == p]
        if sub.empty:
            return 0.0, 0, 0.0
        fare = (sub["avg_fare_weighted"] * sub["total_passengers"]).sum() / max(sub["total_passengers"].sum(), 1)
        pax  = int(sub["total_passengers"].sum())
        hhi  = sub.groupby(["Origin", "Dest"])["route_HHI"].first().mean() if "route_HHI" in sub.columns else 0.0
        return fare, pax, hhi

    fare_a, pax_a, hhi_a = period_stats(period_a)
    fare_b, pax_b, hhi_b = period_stats(period_b)

    k1, k2, k3 = st.columns(3)
    with k1:

        card_header("Avg Fare (A vs B)")
        c_a, vs, c_b = st.columns([2, 1, 2])
        c_a.metric("Period A", f"${fare_a:,.0f}")
        vs.markdown("<div style='text-align:center;padding-top:28px;color:#9CA3AF;font-size:11px'>VS</div>", unsafe_allow_html=True)
        c_b.metric("Period B", f"${fare_b:,.0f}", delta=f"${fare_b - fare_a:+,.0f}")

    with k2:

        card_header("Total Passengers (A vs B)")
        c_a, vs, c_b = st.columns([2, 1, 2])
        c_a.metric("Period A", f"{pax_a:,}")
        vs.markdown("<div style='text-align:center;padding-top:28px;color:#9CA3AF;font-size:11px'>VS</div>", unsafe_allow_html=True)
        c_b.metric("Period B", f"{pax_b:,}", delta=f"{pax_b - pax_a:+,}")

    with k3:

        card_header("HHI (A vs B)")
        c_a, vs, c_b = st.columns([2, 1, 2])
        c_a.metric("Period A", f"{hhi_a:,.0f}")
        vs.markdown("<div style='text-align:center;padding-top:28px;color:#9CA3AF;font-size:11px'>VS</div>", unsafe_allow_html=True)
        c_b.metric("Period B", f"{hhi_b:,.0f}", delta=f"{hhi_b - hhi_a:+,.0f}")


    st.write("")

    # ── Trend charts ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:

        card_header("Fare Trend Over Time")
        if not combined.empty:
            trend = (
                combined.groupby("period")
                .apply(lambda g: (g["avg_fare_weighted"] * g["total_passengers"]).sum() / max(g["total_passengers"].sum(), 1))
                .reset_index(name="Avg Fare ($)")
                .sort_values("period")
            )
            st.plotly_chart(plotly_line(trend, "period", "Avg Fare ($)"), use_container_width=True)
        else:
            st.info("Load multiple periods to see trend.")


    with col_right:

        card_header("Market Share Change")
        if not combined.empty and period_a and period_b:
            def carrier_share(p):
                sub = combined[combined["period"] == p]
                if sub.empty:
                    return pd.Series(dtype=float)
                total = sub["total_passengers"].sum()
                return sub.groupby("Carrier")["total_passengers"].sum() / total * 100

            sa = carrier_share(period_a).rename("Period A")
            sb = carrier_share(period_b).rename("Period B")
            share_cmp = pd.concat([sa, sb], axis=1).dropna().reset_index()
            share_cmp = share_cmp.sort_values("Period B", ascending=False).head(8)
            fig = plotly_grouped_bar(share_cmp, "Carrier", ["Period A", "Period B"],
                                     [period_a, period_b])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least two periods to compare shares.")


    caption("Cross-period analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# Frame 4 — Capacity Module
# ═══════════════════════════════════════════════════════════════════════════════

def frame_capacity(route_df: pd.DataFrame, hub_df: pd.DataFrame,
                   all_periods_route: Dict[str, pd.DataFrame], period: str = "") -> None:
    origin, dest, carrier, region = filter_bar("f4", route_df, period)
    rdf = apply_filters(route_df.copy(), origin, dest, carrier, region)

    # Capacity is approximated from passenger-miles (avg_distance_weighted × total_passengers)
    if not rdf.empty and "avg_distance_weighted" in rdf.columns:
        rdf["passenger_miles"] = rdf["avg_distance_weighted"] * rdf["total_passengers"]
        total_rpm   = rdf["passenger_miles"].sum()
        # Assume load factor ~82 % to estimate ASM
        load_factor = 0.82
        total_asm   = total_rpm / load_factor
        cap_hhi     = rdf.groupby(["Origin", "Dest"])["route_HHI"].first().mean() if "route_HHI" in rdf.columns else 0
    else:
        total_rpm, total_asm, load_factor, cap_hhi = 0, 0, 0, 0

    # ── KPI row ──────────────────────────────────────────────────────────────
    section_header("Phase 3: Capacity Analytics")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Capacity Supplied",  f"{total_asm / 1e9:,.1f}B" if total_asm else "—")
    k2.metric("ASM",                f"{total_asm / 1e9:,.1f}B" if total_asm else "—", help="Available Seat Miles")
    k3.metric("RPM",                f"{total_rpm / 1e9:,.1f}B" if total_rpm else "—", help="Revenue Passenger Miles")
    k4.metric("Load Factor",        f"{load_factor*100:.1f}%")
    k5.metric("Capacity HHI",       f"{cap_hhi:,.0f}" if cap_hhi else "—")

    st.write("")

    # ── Charts row ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:

        card_header("Capacity Share by Airline")
        if not rdf.empty and "passenger_miles" in rdf.columns:
            cap_share = (
                rdf.groupby("Carrier")["passenger_miles"].sum()
                .reset_index()
                .sort_values("passenger_miles", ascending=False)
                .head(8)
            )
            cap_share["ASM (B)"] = (cap_share["passenger_miles"] / load_factor / 1e9).round(2)
            st.plotly_chart(plotly_bar(cap_share, "Carrier", "ASM (B)", "", ylabel="ASM (B)"),
                            use_container_width=True)
        else:
            st.info("No capacity data for selected filters.")


    with col_right:

        card_header("Capacity Trend")
        if all_periods_route:
            trend_rows = []
            for p, df in sorted(all_periods_route.items()):
                fdf = apply_filters(df.copy(), origin, dest, carrier, region)
                if not fdf.empty and "avg_distance_weighted" in fdf.columns:
                    rpm = (fdf["avg_distance_weighted"] * fdf["total_passengers"]).sum()
                    trend_rows.append({"period": p, "ASM (B)": round(rpm / load_factor / 1e9, 2)})
            if trend_rows:
                trend_df = pd.DataFrame(trend_rows).sort_values("period")
                st.plotly_chart(plotly_line(trend_df, "period", "ASM (B)", color=ACCENT_MID),
                                use_container_width=True)
            else:
                st.info("No trend data available.")
        else:
            st.info("Load multiple periods to see capacity trend.")


    caption("Future capacity analytics module.")


# ═══════════════════════════════════════════════════════════════════════════════
# Navigation helpers (kept from original)
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_state() -> None:
    defaults = {
        "screen":        "home",
        "history":       [],
        "stack":         [],
        "single_period": None,
        "multi_periods": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def nav_to(screen: str) -> None:
    current = st.session_state.screen
    if current != screen:
        st.session_state.stack.append(current)
    st.session_state.screen = screen


def nav_back() -> None:
    if st.session_state.stack:
        st.session_state.screen = st.session_state.stack.pop()
    else:
        st.session_state.screen = "home"


def top_nav() -> None:
    _, col_right = st.columns([9, 1])
    with col_right:
        if st.button("← Back", use_container_width=True):
            nav_back()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Original screens (home, history, loaded, start, load, analyze_one, analyze_multi)
# ═══════════════════════════════════════════════════════════════════════════════

def screen_home() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📋  History", use_container_width=True):
            nav_to("history"); st.rerun()
    with c2:
        if st.button("🗂  Loaded Data Sets", use_container_width=True):
            nav_to("loaded"); st.rerun()
    with c3:
        if st.button("▶  Start", type="primary", use_container_width=True):
            nav_to("start"); st.rerun()


def screen_history() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.subheader("Session History")
    if not st.session_state.history:
        st.info("No datasets viewed yet.")
        return
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    for idx, event in enumerate(st.session_state.history, 1):
        st.write(f"{idx}. {event}")


def screen_loaded() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.subheader("Loaded Data Sets")
    periods = collect_periods()
    if not periods:
        st.warning("No periods available yet. Load a dataset first.")

    for p in sorted(periods.keys()):
        complete = is_complete(periods[p])
        icon = "✅" if complete else "⚠️"
        status = "Ready" if complete else "Missing required file(s) — please re-import"
        color = "#106b21" if complete else "#b00020"
        st.markdown(
            f'<div style="padding:8px 0; color:{color}; font-weight:600">{icon} {p} — {status}</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("+ Add Data Set", type="primary"):
        nav_to("load"); st.rerun()


def screen_start() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.subheader("Start")
    st.write("")
    if st.button("Analyze One Period", type="primary", use_container_width=True):
        nav_to("analyze_one"); st.rerun()
    if st.button("Analyze Multiple Periods", use_container_width=True):
        nav_to("analyze_multi"); st.rerun()
    st.write("")
    _, c2, _ = st.columns([4, 2, 4])
    with c2:
        if st.button("Load Data Set", use_container_width=True):
            nav_to("load"); st.rerun()


def screen_load() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.subheader("Load Data Set")
    st.caption("Drop a DB1BMarket CSV file or provide its file path.")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
    with col2:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)

    uploaded   = st.file_uploader("Drop CSV file", type=["csv"])
    local_path = st.text_input("Or enter file path",
                               placeholder=r"C:\path\to\Origin_and_Destination_Survey_DB1BMarket_2025_1.csv")
    delete_raw = st.checkbox("Delete raw file after processing (recommended)", value=True)

    if st.button("Run Import + Analysis", type="primary"):
        raw_path: Optional[Path] = None
        uploaded_tmp = False

        if uploaded is not None:
            tmp = UPLOADS_DIR / uploaded.name
            with open(tmp, "wb") as handle:
                handle.write(uploaded.getbuffer())
            raw_path, uploaded_tmp = tmp, True
        elif local_path.strip():
            entered = Path(local_path.strip())
            if entered.exists() and entered.is_file():
                raw_path = entered
            else:
                st.error("Provided path does not exist.")
                return
        else:
            st.error("Upload a file or enter a local path.")
            return

        with st.spinner("Running parser and analyzer — this may take a few minutes for large files..."):
            ok, message = run_import(raw_path, int(year), int(quarter))

        if ok:
            st.success(message)
            add_history(f"Imported {period_key(int(year), int(quarter))}")
            if delete_raw and raw_path.exists():
                raw_path.unlink(missing_ok=True)
            # Clear cache so new data is picked up immediately
            st.cache_data.clear()
        else:
            st.error(message)


def screen_analyze_one() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.subheader("Analyze One Period")
    periods = collect_periods()
    complete_list = [p for p in sorted(periods.keys()) if is_complete(periods[p])]
    incomplete    = [p for p in sorted(periods.keys()) if not is_complete(periods[p])]

    for p in incomplete:
        st.markdown(f'<div style="color:#b00020;font-weight:600">⚠️ {p} — Unavailable (re-import required)</div>',
                    unsafe_allow_html=True)

    if not complete_list:
        st.warning("No complete periods available yet.")
        if st.button("+ Add Data Set"):
            nav_to("load"); st.rerun()
        return

    selected = st.selectbox("Available periods", complete_list, index=0)
    if st.button("Open Analytics", type="primary"):
        st.session_state.single_period = selected
        add_history(f"Viewed one period: {selected}")
        nav_to("results_one")
        st.rerun()

    if st.button("+ Add Data Set"):
        nav_to("load"); st.rerun()


def screen_analyze_multi() -> None:
    page_header("Airline Market Analysis Dashboard")
    st.subheader("Analyze Multiple Periods")
    periods = collect_periods()
    complete_list = [p for p in sorted(periods.keys()) if is_complete(periods[p])]

    chosen = st.multiselect("Select periods to compare", complete_list,
                            default=st.session_state.multi_periods)
    st.session_state.multi_periods = chosen

    _, c2 = st.columns([7, 1])
    with c2:
        if st.button("Next →", type="primary", use_container_width=True, disabled=len(chosen) == 0):
            add_history("Viewed multi-periods: " + ", ".join(chosen))
            nav_to("results_multi")
            st.rerun()

    if st.button("+ Add Data Set"):
        nav_to("load"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Results screens — now wired to the 4-frame dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def screen_results_one() -> None:
    page_header("Airline Market Analysis Dashboard")
    period = st.session_state.single_period
    if not period:
        st.info("No period selected.")
        return

    route_df = load_route_mp(period)
    hub_df   = load_hub_mp(period)

    if route_df.empty and hub_df.empty:
        st.error(f"Could not load data for {period}. Try re-importing.")
        return

    # Patch year/quarter display into filter bar cosmetic dropdowns
    parts = period.split("_Q")
    yr, qt = (parts[0], f"Q{parts[1]}") if len(parts) == 2 else ("—", "—")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Frame 1: Market Overview",
        "Frame 2: Route & Hub Insights",
        "Frame 3: Time Comparison",
        "Frame 4: Capacity Module",
    ])

    with tab1:
        frame_market_overview(route_df, hub_df, period)
    with tab2:
        frame_route_hub(route_df, hub_df, period)
    with tab3:
        st.info("Time Comparison requires multiple periods. Use 'Analyze Multiple Periods' from the Start screen.")
    with tab4:
        frame_capacity(route_df, hub_df, {period: route_df}, period)


def screen_results_multi() -> None:
    page_header("Airline Market Analysis Dashboard")
    periods = st.session_state.multi_periods
    if not periods:
        st.info("No periods selected.")
        return

    # Load all selected periods
    all_route = {p: load_route_mp(p) for p in periods}
    all_hub   = {p: load_hub_mp(p)   for p in periods}

    # Use the most recent period as the primary dataframe for filters
    latest = sorted(periods)[-1]
    route_df = all_route.get(latest, pd.DataFrame())
    hub_df   = all_hub.get(latest,   pd.DataFrame())

    # Combine all route data for multi-period frames
    combined_route = pd.concat(
        [df.assign(period=p) for p, df in all_route.items() if not df.empty],
        ignore_index=True
    ) if all_route else pd.DataFrame()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Frame 1: Market Overview",
        "Frame 2: Route & Hub Insights",
        "Frame 3: Time Comparison",
        "Frame 4: Capacity Module",
    ])

    with tab1:
        st.caption(f"Showing combined data across: {', '.join(sorted(periods))}")
        frame_market_overview(combined_route if not combined_route.empty else route_df, hub_df)
    with tab2:
        st.caption(f"Showing combined data across: {', '.join(sorted(periods))}")
        frame_route_hub(combined_route if not combined_route.empty else route_df, hub_df)
    with tab3:
        frame_time_comparison(sorted(periods))
    with tab4:
        frame_capacity(route_df, hub_df, all_route, latest)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Airline Market Analysis Dashboard",
        page_icon="✈️",
        layout="wide",
    )
    init_dirs()
    ensure_state()
    inject_css()

    if st.session_state.screen != "home":
        top_nav()

    screen_map = {
        "home":          screen_home,
        "history":       screen_history,
        "loaded":        screen_loaded,
        "start":         screen_start,
        "load":          screen_load,
        "analyze_one":   screen_analyze_one,
        "analyze_multi": screen_analyze_multi,
        "results_one":   screen_results_one,
        "results_multi": screen_results_multi,
    }

    current = st.session_state.screen
    if current in screen_map:
        screen_map[current]()
    else:
        st.session_state.screen = "home"
        st.rerun()


if __name__ == "__main__":
    main()