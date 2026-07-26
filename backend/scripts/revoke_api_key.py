"""
Revoke a per-client API key by key_id. Revocation is immediate and
irreversible via this tool -- issue a new key if the client needs access
again.

Usage:
    python scripts/revoke_api_key.py --key-id <uuid>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-id", required=True)
    args = parser.parse_args()

    api_keys.revoke_key(args.key_id)
    print(f"Revoked key_id {args.key_id}.")


if __name__ == "__main__":
    main()
