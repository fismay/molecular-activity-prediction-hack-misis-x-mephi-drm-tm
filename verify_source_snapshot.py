"""Verify the manually curated source snapshot required by the winning pipeline."""
import argparse
import hashlib
from pathlib import Path

EXPECTED = "2ac50f4859e3d16a201eb97cc3bb4d17c7a973da21e1fa3d0aa183814d000f18"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/external/bornyl_amides_expanded_source.csv"))
    args = parser.parse_args()
    actual = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise SystemExit(f"Source SHA256 mismatch: {actual}; expected {EXPECTED}")
    print(f"Source snapshot verified: {args.source} ({actual})")


if __name__ == "__main__":
    main()
