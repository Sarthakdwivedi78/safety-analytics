"""Fit and evaluate the capstone from the command line."""
import argparse

import pandas as pd

from data_model import demo, normalize
from research_model import InsufficientEvidence, fit_research


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--csv", help="Telemetry CSV using canonical field names")
    cli.add_argument("--target", type=float, default=.05)
    args = cli.parse_args()
    raw = pd.read_csv(args.csv, dtype="string") if args.csv else demo()
    data, invalid = normalize(raw)
    try:
        result = fit_research(data, args.target)
    except (InsufficientEvidence, ValueError) as exc:
        cli.exit(1, f"Analysis unavailable: {exc}\n")
    print("User-supplied telemetry" if args.csv else "Synthetic demo — seed 24021")
    print(f"Exact cohort: {result['exact']}; excluded: {result['excluded']}; invalid cells: {sum(invalid.values())}")
    print(result["sizes"].to_string(index=False))
    print(result["metrics"].round(4).to_string(index=False))
    print(f"Calibration target {args.target:.1%}; candidate: {result['gate']}")
    if result["gate_result"]:
        print(result["gate_rows"].round(4).to_string(index=False))
        print(result["gate_result"])


if __name__ == "__main__":
    main()
