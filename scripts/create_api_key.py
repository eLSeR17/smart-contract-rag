#!/usr/bin/env python3
"""Create an API key for the REST API.

The plaintext key is printed exactly once — it cannot be recovered later
(keys are stored hashed). Usage:

    SCRAG_AUTH_DB=auth.sqlite3 python scripts/create_api_key.py "my-service" [rate_per_min]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_contract_rag.auth import APIKeyStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Human-readable name for the key (e.g. 'ci-bot')")
    parser.add_argument("rate_limit", nargs="?", type=int, default=60, help="Requests per minute (default 60)")
    args = parser.parse_args()

    db = Path(__file__).resolve().parent.parent / "auth.sqlite3"
    store = APIKeyStore(db_path=db)
    key_id, plaintext = store.create_key(args.name, rate_limit_per_min=args.rate_limit)
    print(f"key_id : {key_id}")
    print(f"secret : {plaintext}")
    print("Store this secret now — it will never be shown again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
