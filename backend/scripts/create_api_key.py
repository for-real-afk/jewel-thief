"""
Issue a new per-client API key. Prints the raw key exactly once -- it is not
recoverable afterward (only the hash is stored). Store it securely now.

Usage:
    python scripts/create_api_key.py --client-name acme-mobile [--tier standard]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import api_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-name", required=True)
    parser.add_argument("--tier", default=api_keys.DEFAULT_RATE_LIMIT_TIER)
    args = parser.parse_args()

    key_id, raw_key = api_keys.create_key(args.client_name, rate_limit_tier=args.tier)

    print(f"key_id:     {key_id}")
    print(f"client:     {args.client_name}")
    print(f"tier:       {args.tier}")
    print(f"raw key:    {raw_key}")
    print("\nStore this key now -- it cannot be retrieved again, only revoked.")


if __name__ == "__main__":
    main()
