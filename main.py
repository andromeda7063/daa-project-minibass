"""MiniBass: Optimization and Synthesis of Low-Frequency Sound Fields.

Main entry point for running the 10x12m performance venue dry run and benchmark.
"""

import sys
from minibass.dryrun_10x12 import run_10x12_dryrun


def main() -> None:
    print("MiniBass Acoustic System Optimizer")
    print("Executing 10x12m Venue Dry Run...\n")
    results = run_10x12_dryrun(verbose=True)
    print("\nDry run completed successfully.")


if __name__ == "__main__":
    main()
