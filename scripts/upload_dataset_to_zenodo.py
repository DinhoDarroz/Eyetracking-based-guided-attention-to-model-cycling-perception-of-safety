#!/usr/bin/env python3
"""Create a Zenodo draft deposit and upload an EG-PCS dataset archive.

By default this script does NOT publish the record. It creates/updates a draft,
uploads the file, and prints the draft URL plus any reserved DOI returned by
Zenodo. Use --publish only after you have reviewed the draft on Zenodo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


ZENODO_URL = "https://zenodo.org/api"
SANDBOX_URL = "https://sandbox.zenodo.org/api"


def headers(token: str, *, json_content: bool = False) -> dict[str, str]:
    out = {"Authorization": f"Bearer {token}"}
    if json_content:
        out["Content-Type"] = "application/json"
    return out


def fail_for_status(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    try:
        body: Any = response.json()
    except Exception:
        body = response.text
    raise SystemExit(f"{action} failed with HTTP {response.status_code}: {body}")


def create_draft(api_url: str, token: str, metadata: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url}/deposit/depositions",
        data=json.dumps(metadata),
        headers=headers(token, json_content=True),
        timeout=60,
    )
    fail_for_status(response, "Create Zenodo draft")
    return response.json()


def upload_file(bucket_url: str, token: str, archive_path: Path) -> dict[str, Any]:
    with archive_path.open("rb") as f:
        response = requests.put(
            f"{bucket_url}/{archive_path.name}",
            data=f,
            headers=headers(token),
            timeout=None,
        )
    fail_for_status(response, "Upload dataset archive")
    return response.json()


def publish(api_url: str, token: str, deposition_id: int) -> dict[str, Any]:
    response = requests.post(
        f"{api_url}/deposit/depositions/{deposition_id}/actions/publish",
        headers=headers(token),
        timeout=120,
    )
    fail_for_status(response, "Publish Zenodo record")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Dataset archive to upload, e.g. EG-PCS-Dataset-v1.0.0.tar.gz")
    parser.add_argument("--metadata", type=Path, default=Path("zenodo_metadata.json"))
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org for a test upload.")
    parser.add_argument("--publish", action="store_true", help="Publish immediately after upload. Use carefully.")
    parser.add_argument("--token-env", default="ZENODO_TOKEN", help="Environment variable containing the Zenodo token.")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing token. Set {args.token_env}=<your Zenodo access token>.")

    archive_path = args.archive
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    api_url = SANDBOX_URL if args.sandbox else ZENODO_URL

    draft = create_draft(api_url, token, metadata)
    deposition_id = int(draft["id"])
    bucket_url = draft["links"]["bucket"]
    upload_file(bucket_url, token, archive_path)

    doi = None
    metadata_out = draft.get("metadata", {})
    prereserve = metadata_out.get("prereserve_doi")
    if isinstance(prereserve, dict):
        doi = prereserve.get("doi")

    print(f"Created Zenodo draft: {deposition_id}")
    print(f"Uploaded: {archive_path.name}")
    if doi:
        print(f"Reserved DOI: {doi}")
    print(f"Review draft: https://{'sandbox.' if args.sandbox else ''}zenodo.org/uploads/{deposition_id}")

    if args.publish:
        published = publish(api_url, token, deposition_id)
        print(f"Published DOI: {published.get('doi')}")
        print(f"Record URL: {published.get('record_url')}")
    else:
        print("Not published. Review the draft on Zenodo, then publish manually or rerun with --publish.")


if __name__ == "__main__":
    main()

