#!/usr/bin/env python3
"""Replace a file in an existing Zenodo draft deposit.

This is useful after reserving a DOI, updating local metadata, and rebuilding the
archive. The script deletes any existing draft file with the same filename and
uploads the replacement. It does not publish the record.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests


ZENODO_URL = "https://zenodo.org/api"
SANDBOX_URL = "https://sandbox.zenodo.org/api"


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def fail_for_status(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    try:
        body: Any = response.json()
    except Exception:
        body = response.text
    raise SystemExit(f"{action} failed with HTTP {response.status_code}: {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deposition_id", type=int)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--token-env", default="ZENODO_TOKEN")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing token. Set {args.token_env}=<your Zenodo access token>.")

    archive = args.archive
    if not archive.exists():
        raise FileNotFoundError(archive)

    api_url = SANDBOX_URL if args.sandbox else ZENODO_URL
    deposition_url = f"{api_url}/deposit/depositions/{args.deposition_id}"

    response = requests.get(deposition_url, headers=auth_headers(token), timeout=60)
    fail_for_status(response, "Fetch Zenodo draft")
    draft = response.json()
    bucket_url = draft["links"]["bucket"]

    for file_info in draft.get("files", []):
        if file_info.get("filename") == archive.name:
            delete_url = file_info["links"]["self"]
            delete_response = requests.delete(delete_url, headers=auth_headers(token), timeout=120)
            fail_for_status(delete_response, f"Delete existing draft file {archive.name}")

    with archive.open("rb") as f:
        upload_response = requests.put(
            f"{bucket_url}/{archive.name}",
            data=f,
            headers=auth_headers(token),
            timeout=None,
        )
    fail_for_status(upload_response, "Upload replacement archive")

    print(f"Updated Zenodo draft: {args.deposition_id}")
    print(f"Uploaded replacement: {archive.name}")
    print(f"Review draft: https://{'sandbox.' if args.sandbox else ''}zenodo.org/uploads/{args.deposition_id}")


if __name__ == "__main__":
    main()

