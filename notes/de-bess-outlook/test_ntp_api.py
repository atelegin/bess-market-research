#!/usr/bin/env python3
"""
Quick test: verify Netztransparenz API connection and fetch a sample.

Usage:
  export NTP_CLIENT_ID="cm_app_ntp_id_..."
  export NTP_CLIENT_SECRET="ntp_..."
  python test_ntp_api.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

# Ensure project is on path
sys.path.insert(0, os.path.dirname(__file__))

from data.id_fetcher import _get_token, fetch_id_aep_range, _get_credentials


def main():
    cid, csec = _get_credentials()

    if not cid or not csec:
        print("ERROR: Set NTP_CLIENT_ID and NTP_CLIENT_SECRET env vars")
        print("  export NTP_CLIENT_ID='cm_app_ntp_id_...'")
        print("  export NTP_CLIENT_SECRET='ntp_...'")
        sys.exit(1)

    print(f"Client ID: {cid[:20]}...")
    print(f"Client Secret: {csec[:8]}...")

    # Step 1: Get token
    print("\n--- Step 1: OAuth2 token ---")
    try:
        token = _get_token(cid, csec)
        print(f"Token: {token[:20]}...")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # Step 2: Fetch 1 day of ID-AEP
    print("\n--- Step 2: Fetch ID-AEP sample (1 day) ---")
    try:
        df = fetch_id_aep_range("2024-06-01", "2024-06-01", cid, csec)
        if df.empty:
            print("No data returned (check endpoint names in Swagger)")
            print("Try opening https://ds.netztransparenz.de/swagger in browser")
        else:
            print(f"Got {len(df)} rows")
            print(df.head(10))
            print(f"\nStats: mean={df['id_aep_eur_mwh'].mean():.1f}, "
                  f"min={df['id_aep_eur_mwh'].min():.1f}, "
                  f"max={df['id_aep_eur_mwh'].max():.1f}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Step 3: List available endpoints
    print("\n--- Step 3: Probe API endpoints ---")
    import requests
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    for path in [
        "/data/IdAep",
        "/data/NrvSaldo/reBAP/Qualitaetsgesichert",
        "/data/NrvSaldo/AEPModule/Qualitaetsgesichert",
        "/data/NrvSaldo/NRVSaldo/Betrieblich",
        "/data/Spotmarktpreise",
    ]:
        url = f"https://ds.netztransparenz.de/api/v1{path}/2024-06-01/2024-06-02"
        try:
            r = requests.get(
                url,
                headers=headers,
                timeout=15,
            )
            status = r.status_code
            size = len(r.text)
            preview = r.text[:100].replace("\n", " ")
            print(f"  {path:30s}  HTTP {status}  {size:>6} bytes  {preview}")
        except Exception as e:
            print(f"  {path:30s}  ERROR: {e}")


if __name__ == "__main__":
    main()
