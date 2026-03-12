#!/usr/bin/env python3
"""Sync GitHub repository metadata from Cargo.toml package fields."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARGO_TOML = ROOT / "Cargo.toml"
API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync GitHub repository description/topics from Cargo.toml."
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository in owner/name form. Defaults to GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("REPO_ADMIN_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        help="GitHub token with repository administration permission.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived payloads without calling the GitHub API.",
    )
    return parser.parse_args()


def load_package_metadata() -> dict[str, object]:
    with CARGO_TOML.open("rb") as handle:
        cargo = tomllib.load(handle)

    package = cargo.get("package")
    if not isinstance(package, dict):
        raise ValueError("Cargo.toml is missing a [package] section")

    return package


def normalize_topic(topic: str) -> str:
    normalized = topic.strip().lower()
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized[:50]


def build_topics(package: dict[str, object]) -> list[str]:
    raw_topics = []
    for key in ("keywords", "categories"):
        values = package.get(key, [])
        if isinstance(values, list):
            raw_topics.extend(value for value in values if isinstance(value, str))

    topics = []
    seen = set()
    for raw_topic in raw_topics:
        topic = normalize_topic(raw_topic)
        if not topic or topic in seen:
            continue
        topics.append(topic)
        seen.add(topic)

    return topics[:20]


def build_repo_patch(package: dict[str, object], repository: str) -> dict[str, str]:
    description = str(package.get("description", "")).strip()
    if not description:
        raise ValueError("Cargo.toml package.description is empty")

    patch = {"description": description}

    homepage = str(package.get("homepage", "")).strip()
    repo_url = f"https://github.com/{repository}"
    patch["homepage"] = homepage if homepage and homepage != repo_url else ""

    return patch


def github_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, object],
) -> dict[str, object] | None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "lme-rs-repo-metadata-sync",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API {method} {url} failed with {exc.code}: {details}"
        ) from exc


def main() -> int:
    args = parse_args()

    if not args.repository:
        print("error: missing --repository or GITHUB_REPOSITORY", file=sys.stderr)
        return 1

    package = load_package_metadata()
    patch_payload = build_repo_patch(package, args.repository)
    topics_payload = {"names": build_topics(package)}

    print(f"Repository: {args.repository}")
    print(f"Description: {patch_payload['description']}")
    print(f"Topics: {', '.join(topics_payload['names']) or '(none)'}")
    print(f"Homepage: {patch_payload['homepage'] or '(empty)'}")

    if args.dry_run:
        print("Dry run enabled; no GitHub API calls were made.")
        return 0

    if not args.token:
        print("error: missing --token or REPO_ADMIN_TOKEN/GITHUB_TOKEN", file=sys.stderr)
        return 1

    repo_url = f"{API_BASE}/repos/{args.repository}"
    github_request("PATCH", repo_url, args.token, patch_payload)
    github_request("PUT", f"{repo_url}/topics", args.token, topics_payload)

    print("Repository metadata updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
