import subprocess
from pathlib import Path


def period_tag(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"


def raw_filename(year: int, quarter: int) -> str:
    return f"Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.csv"


def ask_menu_choice() -> str:
    print("\n=== Capstone Runner ===")
    print("1) Process NEW DB1BMarket raw CSV")
    print("2) Run ANALYSIS on existing outputs")
    print("3) Exit")
    while True:
        c = input("Choose 1, 2, or 3: ").strip()
        if c in {"1", "2", "3"}:
            return c
        print("Invalid choice.")


def ask_year_quarter() -> tuple[int, int]:
    while True:
        y = input("Year (e.g., 2025): ").strip()
        q = input("Quarter (1-4): ").strip()
        if y.isdigit() and q.isdigit():
            y = int(y)
            q = int(q)
            if 1 <= q <= 4:
                return y, q
        print("Invalid year/quarter.")


def run_cmd(cmd: list[str]) -> None:
    print("\n[running]", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    while True:
        choice = ask_menu_choice()

        if choice == "3":
            print("Bye.")
            return

        year, quarter = ask_year_quarter()
        tag = period_tag(year, quarter)

        # ======================================================
        # OPTION 1 — PROCESS RAW DB1B FILE (AUTO-FIND)
        # ======================================================
        if choice == "1":

            raw_file = raw_filename(year, quarter)
            raw_path = Path(raw_file)

            if not raw_path.exists():
                print(f"\nRaw file not found: {raw_file}")
                print("Make sure the file is in this directory.")
                continue

            cmd = [
                "py", "capstone_parse.py",
                "--csv", raw_file,
                "--year", str(year),
                "--quarter", str(quarter)
            ]

            run_cmd(cmd)

            print(f"\nCreated:")
            print(f" - hubxairline_{tag}.csv")
            print(f" - routexairline_{tag}.csv")

        # ======================================================
        # OPTION 2 — RUN ANALYSIS
        # ======================================================
        elif choice == "2":

            hub = Path(f"hubxairline_{tag}.csv")
            route = Path(f"routexairline_{tag}.csv")

            if not hub.exists() or not route.exists():
                print("\nMissing required input files:")
                if not hub.exists():
                    print(f" - {hub}")
                if not route.exists():
                    print(f" - {route}")
                print("Run option (1) first.")
                continue

            cmd = [
                "py", "capstone_analyze.py",
                "--year", str(year),
                "--quarter", str(quarter),
                "--dir", ".",
                "--export_csv"
            ]

            run_cmd(cmd)

            print(f"\nCreated:")
            print(f" - analysis_airline_summary_{tag}.csv")

        # loop back


if __name__ == "__main__":
    main()
