import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"

HUB_MP_DIR = BACKEND_ROOT / "hubMP_folder"
ROUTE_MP_DIR = BACKEND_ROOT / "routeMP_folder"
HUB_AIRLINE_DIR = BACKEND_ROOT / "hubxairline_folder"
ROUTE_AIRLINE_DIR = BACKEND_ROOT / "routexairline_folder"
UPLOADS_DIR = BACKEND_ROOT / "uploads"

PERIOD_PATTERN = re.compile(r"_(\d{4})_Q([1-4])\.csv$", re.IGNORECASE)


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


def run_backend(cmd: list[str]) -> Tuple[bool, str]:
    result = subprocess.run(cmd, cwd=BACKEND_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, (result.stdout or "").strip()


def collect_periods() -> Dict[str, Dict[str, bool]]:
    periods: Dict[str, Dict[str, bool]] = {}
    required = {
        "hub_market_power": HUB_MP_DIR,
        "route_market_power": ROUTE_MP_DIR,
        "hubxairline": HUB_AIRLINE_DIR,
        "routexairline": ROUTE_AIRLINE_DIR,
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


def ensure_state() -> None:
    if "screen" not in st.session_state:
        st.session_state.screen = "home"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "stack" not in st.session_state:
        st.session_state.stack = []
    if "single_period" not in st.session_state:
        st.session_state.single_period = None
    if "multi_periods" not in st.session_state:
        st.session_state.multi_periods = []


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


def add_history(item: str) -> None:
    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:30]


def style_page() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 900px; padding-top: 1rem;}
        .error-period {color: #b00020; font-weight: 600;}
        .ok-period {color: #106b21; font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_nav() -> None:
    col_left, col_right = st.columns([9, 1])
    with col_right:
        if st.button("Back", use_container_width=True):
            nav_back()
            st.rerun()


def screen_home() -> None:
    st.title("Airline Analytics")
    st.subheader("Home")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("History", use_container_width=True):
            nav_to("history")
            st.rerun()
    with c2:
        if st.button("Loaded Data Sets", use_container_width=True):
            nav_to("loaded")
            st.rerun()
    with c3:
        if st.button("Start", type="primary", use_container_width=True):
            nav_to("start")
            st.rerun()


def screen_history() -> None:
    st.subheader("History")
    if not st.session_state.history:
        st.info("No datasets viewed yet.")
        return
    for idx, event in enumerate(st.session_state.history, 1):
        st.write(f"{idx}. {event}")


def screen_loaded() -> None:
    st.subheader("Loaded Data Sets")
    periods = collect_periods()
    if not periods:
        st.warning("No periods available yet.")

    warning = "PLEASE RE-IMPORT THIS FILE, THIS FILE HAS BEEN CORRUPTED OR FILE PATH HAS BEEN CHANGED."

    for p in sorted(periods.keys()):
        complete = is_complete(periods[p])
        if complete:
            st.markdown(f"<div class='ok-period'>{p} - Ready</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='error-period' title='{warning}'>{p} - Missing required file(s)</div>",
                unsafe_allow_html=True,
            )

    if st.button("+ Add Data Set"):
        nav_to("load")
        st.rerun()


def screen_start() -> None:
    st.subheader("Start")
    if st.button("Analyze One Period", type="primary", use_container_width=True):
        nav_to("analyze_one")
        st.rerun()
    if st.button("Analyze Multiple Periods", use_container_width=True):
        nav_to("analyze_multi")
        st.rerun()

    st.write("")
    st.write("")
    c1, c2, c3 = st.columns([4, 2, 4])
    with c2:
        if st.button("Load Data Set", use_container_width=True):
            nav_to("load")
            st.rerun()


def run_import(raw_csv_path: Path, year: int, quarter: int) -> Tuple[bool, str]:
    parse_cmd = [
        "py",
        "capstone_parse.py",
        "--csv",
        str(raw_csv_path),
        "--year",
        str(year),
        "--quarter",
        str(quarter),
        "--verbose",
        "0",
    ]
    ok_parse, msg_parse = run_backend(parse_cmd)
    if not ok_parse:
        return False, f"capstone_parse.py failed:\n{msg_parse}"

    analyze_cmd = [
        "py",
        "capstone_analyze.py",
        "--year",
        str(year),
        "--quarter",
        str(quarter),
        "--dir",
        ".",
        "--export_csv",
        "--verbose",
        "0",
    ]
    ok_analyze, msg_analyze = run_backend(analyze_cmd)
    if not ok_analyze:
        return False, f"capstone_analyze.py failed:\n{msg_analyze}"

    tag = period_key(year, quarter)
    map_move = {
        BACKEND_ROOT / f"hubxairline_{tag}.csv": HUB_AIRLINE_DIR / f"hubxairline_{tag}.csv",
        BACKEND_ROOT / f"routexairline_{tag}.csv": ROUTE_AIRLINE_DIR / f"routexairline_{tag}.csv",
        BACKEND_ROOT / f"hub_market_power_{tag}.csv": HUB_MP_DIR / f"hub_market_power_{tag}.csv",
        BACKEND_ROOT / f"route_market_power_{tag}.csv": ROUTE_MP_DIR / f"route_market_power_{tag}.csv",
    }
    for src, dst in map_move.items():
        if src.exists():
            shutil.move(str(src), str(dst))

    return True, "Import and analysis complete."


def screen_load() -> None:
    st.subheader("Load Data Set")
    st.caption("Drop a DB1BMarket CSV file or provide its file path.")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
    with col2:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)

    uploaded = st.file_uploader("Drop CSV file", type=["csv"])
    local_path = st.text_input(
        "Or enter file path",
        placeholder=r"C:\path\to\Origin_and_Destination_Survey_DB1BMarket_2025_1.csv",
    )
    delete_raw = st.checkbox(
        "Delete raw file after successful processing (recommended for uploaded files)",
        value=True,
    )

    if st.button("Run Import + Analysis", type="primary"):
        raw_path: Optional[Path] = None
        uploaded_tmp = False

        if uploaded is not None:
            tmp = UPLOADS_DIR / uploaded.name
            with open(tmp, "wb") as handle:
                handle.write(uploaded.getbuffer())
            raw_path = tmp
            uploaded_tmp = True
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

        with st.spinner("Running parser and analyzer..."):
            ok, message = run_import(raw_path, int(year), int(quarter))

        if ok:
            st.success(message)
            add_history(f"Imported {period_key(int(year), int(quarter))}")
            if delete_raw:
                if uploaded_tmp and raw_path.exists():
                    raw_path.unlink(missing_ok=True)
                if (not uploaded_tmp) and raw_path.exists():
                    raw_path.unlink(missing_ok=True)
        else:
            st.error(message)


def screen_analyze_one() -> None:
    st.subheader("Analyze One Period")
    periods = collect_periods()
    warning = "PLEASE RE-IMPORT THIS FILE, THIS FILE HAS BEEN CORRUPTED OR FILE PATH HAS BEEN CHANGED."

    complete_list = []
    for p in sorted(periods.keys()):
        if is_complete(periods[p]):
            complete_list.append(p)
        else:
            st.markdown(
                f"<div class='error-period' title='{warning}'>{p} - Unavailable</div>",
                unsafe_allow_html=True,
            )

    if not complete_list:
        st.warning("No complete periods available yet.")
    else:
        selected = st.selectbox("Available periods", complete_list, index=0)
        if st.button("Open Analytics", type="primary"):
            st.session_state.single_period = selected
            add_history(f"Viewed one period: {selected}")
            nav_to("results_one")
            st.rerun()

    if st.button("+ Add Data Set"):
        nav_to("load")
        st.rerun()


def screen_analyze_multi() -> None:
    st.subheader("Analyze Multiple Periods")
    periods = collect_periods()
    warning = "PLEASE RE-IMPORT THIS FILE, THIS FILE HAS BEEN CORRUPTED OR FILE PATH HAS BEEN CHANGED."

    complete_list = []
    for p in sorted(periods.keys()):
        if is_complete(periods[p]):
            complete_list.append(p)
        else:
            st.markdown(
                f"<div class='error-period' title='{warning}'>{p} - Unavailable</div>",
                unsafe_allow_html=True,
            )

    chosen = st.multiselect("Select periods", complete_list, default=st.session_state.multi_periods)
    st.session_state.multi_periods = chosen

    c1, c2 = st.columns([7, 1])
    with c2:
        if st.button("Next", type="primary", use_container_width=True, disabled=len(chosen) == 0):
            add_history("Viewed multi-periods: " + ", ".join(chosen))
            nav_to("results_multi")
            st.rerun()

    if st.button("+ Add Data Set"):
        nav_to("load")
        st.rerun()


def screen_results_one() -> None:
    st.subheader("One Period Analytics")
    selected = st.session_state.single_period
    if not selected:
        st.info("No period selected.")
        return
    st.success(f"Selected period: {selected}")
    st.info("Analytics content placeholder. Tell me metrics/charts to add and I will wire this screen next.")


def screen_results_multi() -> None:
    st.subheader("Multiple Period Analytics")
    selected = st.session_state.multi_periods
    if not selected:
        st.info("No periods selected.")
        return
    st.success("Selected periods: " + ", ".join(selected))
    st.info("Comparison content placeholder. Tell me metrics/charts to add and I will wire this screen next.")


def main() -> None:
    st.set_page_config(page_title="Airline Analytics", page_icon="AA")
    init_dirs()
    ensure_state()
    style_page()

    if st.session_state.screen != "home":
        top_nav()

    current = st.session_state.screen
    if current == "home":
        screen_home()
    elif current == "history":
        screen_history()
    elif current == "loaded":
        screen_loaded()
    elif current == "start":
        screen_start()
    elif current == "load":
        screen_load()
    elif current == "analyze_one":
        screen_analyze_one()
    elif current == "analyze_multi":
        screen_analyze_multi()
    elif current == "results_one":
        screen_results_one()
    elif current == "results_multi":
        screen_results_multi()
    else:
        st.session_state.screen = "home"
        st.rerun()


if __name__ == "__main__":
    main()
