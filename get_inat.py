#!/usr/bin/env python3
"""
Download iNaturalist photo metadata (URLs + attribution + license) for observations.

Key features:
- Query taxa by:
    * --taxon_id (single)
    * --taxon_name (single; scientific or common name)
    * --taxa_tree_json (nested taxonomy JSON like your snippet; extracts names from "specie" arrays)
- Filter by place/project/user and photo license
- Output JSON is keyed by TAXON NAME (your query string), not taxon_id
- Update mode: replace entries for taxa queried this run, keep the rest

Examples:
  # Single taxon by name (commercial OK with attribution: CC-BY, CC-BY-SA)
  python inat_fetch_photos.py --taxon_name "Spongilla" --photo_license cc-by,cc-by-sa --update_out

  # From nested taxonomy JSON (extract genus/species from "specie")
  python inat_fetch_photos.py --taxa_tree_json taxa_tree.json --photo_license cc-by --max_obs 500 --update_out

  # Overwrite output (no update): only contains taxa fetched this run
  python inat_fetch_photos.py --taxa_tree_json taxa_tree.json --out photos.json

Notes:
- This script fetches METADATA (URLs + license + attribution), NOT the image binaries.
- iNat v2 uses lowercase license codes like: cc0, cc-by, cc-by-sa, cc-by-nc, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from tqdm import tqdm
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

INAT_V2_OBS = "https://api.inaturalist.org/v2/observations"
INAT_V2_TAXA = "https://api.inaturalist.org/v2/taxa"


# -----------------------------
# Small helpers
# -----------------------------
def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def normalize_photo_license_arg(s: Optional[str]) -> Optional[str]:
    """
    Accepts convenient forms like 'CC-BY,CC-BY-SA' and normalizes to iNat v2:
      'cc-by,cc-by-sa'

    Also allows already-normalized inputs.
    """
    if not s:
        return None
    parts: List[str] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        low = p.lower().replace(" ", "")
        low = low.replace("ccby", "cc-by")
        low = low.replace("cc-by-", "cc-by-")
        low = low.replace("cc-0", "cc0")
        parts.append(low)
    return ",".join(parts) if parts else None


def fetch_json(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code == 422:
        # Useful for debugging invalid params
        print(f"422 Unprocessable Entity for {r.url}", file=sys.stderr)
        print("Response body:", r.text, file=sys.stderr)
    r.raise_for_status()
    return r.json()


def upgrade_photo_url(url: str, size: str) -> str:
    """Replace the size segment in an iNat photo URL."""
    if not url or size == "square":
        return url
    for s in ("square", "small", "medium", "large", "original"):
        url = url.replace(f"/{s}.", f"/{size}.")
    return url


# -----------------------------
# Taxon resolution
# -----------------------------
def resolve_taxon_name(
    session: requests.Session, name: str, timeout: int = 30
) -> Optional[int]:
    """
    Resolve a taxon scientific/common name to a taxon_id using /v2/taxa.

    Heuristic: pick top result ordered by observations_count.
    """
    name = norm(name)
    if not name:
        return None
    params = {
        "q": name,
        "per_page": 1,
        "order_by": "observations_count",
        "order": "desc",
    }
    data = fetch_json(session, INAT_V2_TAXA, params=params, timeout=timeout)
    results = data.get("results") or []
    if not results:
        return None
    return results[0].get("id")


# -----------------------------
# Extract taxa from nested JSON
# -----------------------------
def extract_taxon_queries(tree: Any) -> List[str]:
    """
    Walk a nested taxonomy JSON (like your snippet) and return a list of taxon query strings.

    Extraction rule:
    - If a dict has key 'specie' (list), for each item:
        genus + species (if species present) else genus

    Dedupe (case-insensitive) while preserving order.
    """
    out: List[str] = []

    def add(q: str) -> None:
        q = norm(q)
        if q:
            out.append(q)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "specie" in node and isinstance(node["specie"], list):
                for sp in node["specie"]:
                    if not isinstance(sp, dict):
                        continue
                    genus = norm(sp.get("genus", ""))
                    species = norm(sp.get("species", ""))
                    if genus:
                        add(f"{genus} {species}".strip())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(tree)

    seen = set()
    deduped: List[str] = []
    for q in out:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped


# -----------------------------
# Observations + photos
# -----------------------------
def iter_observations(
    *,
    taxon_id: Optional[int],
    place_id: Optional[int],
    project_id: Optional[int],
    user_id: Optional[str],
    photo_license: Optional[str],
    per_page: int,
    max_obs: int,
    polite_delay: float,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Returns a list of observation dicts with only requested fields.
    Uses page-based pagination.
    """
    fields = (
        "id,uri,photos.id,photos.url,photos.license_code,photos.attribution"
    )
    page = 1
    collected: List[Dict[str, Any]] = []

    with requests.Session() as session:
        while True:
            params: Dict[str, Any] = {
                "per_page": per_page,
                "page": page,
                "order_by": "id",
                "order": "asc",
                "fields": fields,
                "photos": "true",
            }
            if taxon_id is not None:
                params["taxon_id"] = taxon_id
            if place_id is not None:
                params["place_id"] = place_id
            if project_id is not None:
                params["project_id"] = project_id
            if user_id is not None:
                params["user_id"] = user_id
            if photo_license:
                params["photo_license"] = photo_license

            data = fetch_json(
                session, INAT_V2_OBS, params=params, timeout=timeout
            )
            results = data.get("results") or []
            if not results:
                break

            collected.extend(results)
            if len(collected) >= max_obs:
                collected = collected[:max_obs]
                break

            page += 1
            if polite_delay:
                time.sleep(polite_delay)

    return collected


def flatten_photos(
    observations: List[Dict[str, Any]], size: str
) -> List[Dict[str, Any]]:
    """
    Produces one record per photo with:
      - observation links
      - photo links
      - attribution + license_code
    """
    out: List[Dict[str, Any]] = []
    for obs in observations:
        obs_id = obs.get("id")
        obs_uri = obs.get("uri") or (
            f"https://www.inaturalist.org/observations/{obs_id}"
            if obs_id
            else None
        )

        for p in obs.get("photos") or []:
            photo_id = p.get("id")
            square_url = p.get("url")
            chosen_url = upgrade_photo_url(square_url, size)

            out.append(
                {
                    "observation_id": obs_id,
                    "observation_url": obs_uri,
                    "photo_id": photo_id,
                    "photo_page_url": f"https://www.inaturalist.org/photos/{photo_id}"
                    if photo_id
                    else None,
                    "url_square": square_url,
                    "url": chosen_url,
                    "license_code": p.get("license_code"),
                    "attribution": p.get("attribution"),
                    "retrieved_from_api": INAT_V2_OBS,
                }
            )
    return out


# -----------------------------
# Output loading/updating
# -----------------------------
def load_existing_out(path: str) -> Dict[str, Any]:
    """
    Loads an existing output JSON file.
    Ensures a dict with {"taxa": {...}}.
    """
    if not os.path.exists(path):
        return {"taxa": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"taxa": {}}

    if not isinstance(data, dict):
        return {"taxa": {}}
    if "taxa" not in data or not isinstance(data["taxa"], dict):
        data["taxa"] = {}
    return data


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--taxon_id", type=int, default=None, help="Single taxon_id (numeric)"
    )
    ap.add_argument(
        "--taxon_name",
        type=str,
        default=None,
        help="Single taxon name (scientific or common)",
    )
    ap.add_argument(
        "--taxa_tree_json",
        type=str,
        default=None,
        help="Nested taxonomy JSON like your snippet; extracts names from 'specie' lists.",
    )

    ap.add_argument("--place_id", type=int, default=None)
    ap.add_argument("--project_id", type=int, default=None)
    ap.add_argument(
        "--user_id", type=str, default=None, help="login name or numeric ID"
    )

    ap.add_argument(
        "--photo_license",
        type=str,
        default=None,
        help="Comma-separated list like CC-BY,CC-BY-SA (or cc-by,cc-by-sa).",
    )

    ap.add_argument("--per_page", type=int, default=200)
    ap.add_argument("--max_obs", type=int, default=500)
    ap.add_argument(
        "--polite_delay",
        type=float,
        default=0.2,
        help="seconds between requests",
    )
    ap.add_argument(
        "--timeout", type=int, default=30, help="request timeout seconds"
    )

    ap.add_argument(
        "--size",
        type=str,
        default="original",
        choices=["square", "small", "medium", "large", "original"],
    )

    ap.add_argument("--out", type=str, default="inat_photos.json")
    ap.add_argument(
        "--update_out",
        action="store_true",
        help="Update an existing output file by REPLACING entries for taxa queried this run. "
        "If not set, overwrites the entire file with only this run's taxa.",
    )
    args = ap.parse_args()

    if args.per_page < 1 or args.per_page > 200:
        print("per_page must be between 1 and 200", file=sys.stderr)
        return 2

    # Build list of taxon query strings (names)
    taxon_queries: List[str] = []
    if args.taxon_name:
        taxon_queries.append(args.taxon_name)

    if args.taxa_tree_json:
        with open(args.taxa_tree_json, "r", encoding="utf-8") as f:
            tree = json.load(f)
        taxon_queries.extend(extract_taxon_queries(tree))

    # Deduplicate taxon_queries (case-insensitive)
    seen_q = set()
    taxon_queries = [
        q
        for q in taxon_queries
        if not (q.lower() in seen_q or seen_q.add(q.lower()))
    ]

    # Normalize license filter for v2
    photo_license = normalize_photo_license_arg(args.photo_license)

    # Resolve taxon names -> taxon_ids
    resolved_taxa: Dict[str, int] = {}  # key is taxon_name (query string)
    unresolved: List[str] = []

    if taxon_queries:
        with requests.Session() as session:
            for name in tqdm(taxon_queries):
                tid = resolve_taxon_name(session, name, timeout=args.timeout)
                if tid:
                    resolved_taxa[name] = tid
                else:
                    unresolved.append(name)

    # Also allow single --taxon_id without a name:
    # We'll store it under a stable key "taxon_id:<id>" unless you prefer something else.
    if args.taxon_id is not None:
        resolved_taxa[f"taxon_id:{args.taxon_id}"] = args.taxon_id

    if not resolved_taxa:
        print(
            "No taxa specified/resolved. Use --taxon_name, --taxa_tree_json, or --taxon_id.",
            file=sys.stderr,
        )
        return 2

    if unresolved:
        print(
            "Warning: could not resolve these taxa (skipping):", file=sys.stderr
        )
        for name in unresolved:
            print(f"  - {name}", file=sys.stderr)

    # Fetch and build per-taxon results, keyed by taxon_name (query string)
    per_taxon_results: Dict[str, Any] = {}
    for taxon_name, taxon_id in tqdm(resolved_taxa.items()):
        observations = iter_observations(
            taxon_id=taxon_id,
            place_id=args.place_id,
            project_id=args.project_id,
            user_id=args.user_id,
            photo_license=photo_license,
            per_page=args.per_page,
            max_obs=args.max_obs,
            polite_delay=args.polite_delay,
            timeout=args.timeout,
        )

        photos = flatten_photos(observations, size=args.size)

        # Make each record self-describing
        for p in photos:
            p["taxon_name"] = taxon_name
            p["taxon_id"] = taxon_id

        per_taxon_results[taxon_name] = {
            "taxon_name": taxon_name,
            "taxon_id": taxon_id,
            "count_observations": len(observations),
            "count_photos": len(photos),
            "photos": photos,
            "updated_at_utc": utc_now_iso(),
            "filters": {
                "place_id": args.place_id,
                "project_id": args.project_id,
                "user_id": args.user_id,
                "photo_license": photo_license,
                "per_page": args.per_page,
                "max_obs": args.max_obs,
                "size": args.size,
            },
            "retrieved_from_api": INAT_V2_OBS,
        }

    # Write output:
    # - if --update_out: load existing and replace only taxa queried this run
    # - else: overwrite file
    if args.update_out:
        existing = load_existing_out(args.out)
        existing["taxa"].update(per_taxon_results)
        existing["updated_at_utc"] = utc_now_iso()
        existing["last_run"] = {
            "taxa_updated": list(per_taxon_results.keys()),
            "count_taxa_updated": len(per_taxon_results),
        }
        out_payload = existing
    else:
        out_payload = {
            "taxa": per_taxon_results,
            "updated_at_utc": utc_now_iso(),
            "last_run": {
                "taxa_updated": list(per_taxon_results.keys()),
                "count_taxa_updated": len(per_taxon_results),
            },
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    total_photos = sum(
        v.get("count_photos", 0) for v in per_taxon_results.values()
    )
    total_obs = sum(
        v.get("count_observations", 0) for v in per_taxon_results.values()
    )

    print(
        f"Wrote/updated {total_photos} photo records from {total_obs} observations "
        f"across {len(per_taxon_results)} taxa to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
