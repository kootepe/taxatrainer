#!/usr/bin/env python3
"""
Scrape iNaturalist taxon pages for their curated photos (CoverImage tiles),
then "click" each image by visiting /photos/<id> to get license + attribution.

This does NOT use observations search.

How photos are discovered:
- Looks for elements whose id starts with "cover-image-"
  Example:
    id="cover-image-https-inaturalist-open-data-s-3-amazonaws-com-photos-162318518-medium-jpg"
- Also (fallback) parses background-image URLs from style attributes.

Output JSON is keyed by taxon key (name / taxon_id:<id> / url) and supports:
- --update_out: replace taxa fetched this run in an existing output file
- --resume: skip taxa already present in output
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


INAT_V2_TAXA = "https://api.inaturalist.org/v2/taxa"

PHOTO_ID_RE = re.compile(r"/photos/(\d+)/", re.IGNORECASE)
BG_URL_RE = re.compile(
    r"background-image:\s*url\((['\"]?)(.*?)\1\)", re.IGNORECASE
)

LICENSE_CODE_RE = re.compile(r'"license_code"\s*:\s*"([^"]+)"', re.IGNORECASE)
ATTRIBUTION_RE = re.compile(r'"attribution"\s*:\s*"([^"]+)"', re.IGNORECASE)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def normalize_license_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    out: List[str] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(p.lower().replace(" ", ""))
    return out or None


def upgrade_photo_url(url: str, size: str) -> str:
    if not url or size == "square":
        return url
    for s in ("square", "small", "medium", "large", "original"):
        url = url.replace(f"/{s}.", f"/{size}.")
    return url


def extract_photo_id_from_url(url: str) -> Optional[int]:
    m = PHOTO_ID_RE.search(url or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# -----------------------------
# Robust HTTP
# -----------------------------
def get_text_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            r = session.get(
                url, timeout=timeout, headers=headers, allow_redirects=True
            )
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_s = backoff_base * (2**attempt) * random.uniform(0.7, 1.3)
                if attempt >= max_retries:
                    r.raise_for_status()
                print(
                    f"HTTP {r.status_code} for {url} — retrying in {sleep_s:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.text
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            sleep_s = backoff_base * (2**attempt) * random.uniform(0.7, 1.3)
            print(
                f"{type(e).__name__}: {e} — retrying in {sleep_s:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected failure")


def fetch_json_v2_taxa(
    session: requests.Session,
    params: Dict[str, Any],
    *,
    timeout: int,
    max_retries: int,
    backoff_base: float,
) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            r = session.get(INAT_V2_TAXA, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_s = backoff_base * (2**attempt) * random.uniform(0.7, 1.3)
                if attempt >= max_retries:
                    r.raise_for_status()
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            sleep_s = backoff_base * (2**attempt) * random.uniform(0.7, 1.3)
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected failure")


# -----------------------------
# Taxon name -> id (exact-first)
# -----------------------------
def resolve_taxon_name_exact_first(
    session: requests.Session,
    name: str,
    *,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    strict_exact: bool = False,
) -> Optional[int]:
    q = norm(name)
    if not q:
        return None

    params: Dict[str, Any] = {
        "q": q,
        "per_page": 30,
        "order_by": "observations_count",
        "order": "desc",
    }

    data = fetch_json_v2_taxa(
        session,
        params,
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=backoff_base,
    )
    results = data.get("results") or []
    if not results:
        return None

    ql = q.lower()
    for t in results:
        if (t.get("name") or "").strip().lower() == ql:
            return t.get("id")
    if name == "asteroidea":
        return None if strict_exact else results[1].get("id")
    else:
        return None if strict_exact else results[0].get("id")


# -----------------------------
# Extract taxa from your JSON
# -----------------------------
def extract_taxon_queries(tree: Any) -> List[str]:
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
            if node.get("lat"):
                add(node["lat"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(tree)

    seen = set()
    deduped: List[str] = []
    for q in out:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(q)
    return deduped


def taxon_url_from_id(taxon_id: int) -> str:
    return f"https://www.inaturalist.org/taxa/{taxon_id}"


# -----------------------------
# Extract URLs from CoverImage IDs + styles
# -----------------------------
def decode_cover_image_id_to_url(cover_id: str) -> Optional[str]:
    """
    cover_id example:
      cover-image-https-inaturalist-open-data-s-3-amazonaws-com-photos-162318518-medium-jpg

    This is essentially a URL with separators replaced by '-'.
    We can reconstruct reliably enough for the known iNat image hosts.

    Strategy:
    - Strip prefix 'cover-image-'
    - Detect host:
        https-inaturalist-open-data-s-3-amazonaws-com  -> https://inaturalist-open-data.s3.amazonaws.com
        https-static-inaturalist-org                   -> https://static.inaturalist.org
    - Then parse: photos-<id>-<size>-<ext>
      and rebuild: https://<host>/photos/<id>/<size>.<ext>
    """
    if not cover_id.startswith("cover-image-"):
        print("no id with cover-image")
        return None
    s = cover_id[len("cover-image-") :]

    host = None
    rest = None

    if s.startswith("https-inaturalist-open-data-s-3-amazonaws-com-"):
        host = "https://inaturalist-open-data.s3.amazonaws.com"
        rest = s[len("https-inaturalist-open-data-s-3-amazonaws-com-") :]
    elif s.startswith("https-static-inaturalist-org-"):
        host = "https://static.inaturalist.org"
        rest = s[len("https-static-inaturalist-org-") :]
    else:
        return None

    # Expect rest like: photos-162318518-medium-jpg  (or jpeg)
    m = re.match(
        r"photos-(\d+)-([a-z]+)-([a-z0-9]+)$", rest, flags=re.IGNORECASE
    )
    if not m:
        return None

    photo_id = m.group(1)
    size = m.group(2).lower()
    ext = m.group(3).lower()
    return f"{host}/photos/{photo_id}/{size}.{ext}"


def extract_photo_json(html: str) -> json:
    print("Extracting json")
    # print(html)
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []

    data = None
    # 1) From CoverImage ids
    for tag in soup.find("script", string=re.compile("SERVER_PAYLOAD")):
        match = re.search(r"\s*taxon: ({.*?})\s*;", tag, re.S)
        if match is None:
            print("No match")
            continue
        data = re.sub(r"\.results.*", "}", match.group(1), flags=re.S)
        data = data[:-1]
        data = json.loads(data)["results"][0]
    return data


def extract_photo_from_json(data: dict) -> List[str]:
    print("Extracting photos")

    images = [img["photo"] for img in data["taxon_photos"]]
    return images


# -----------------------------
# Follow photo page for license/attribution
# -----------------------------
def scrape_photo_page_license_attrib(
    session: requests.Session,
    photo_id: int,
    *,
    timeout: int,
    max_retries: int,
    backoff_base: float,
) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://www.inaturalist.org/photos/{photo_id}"
    html = get_text_with_retries(
        session,
        url,
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=backoff_base,
        headers={"User-Agent": "inat-taxon-coverimage-scraper/1.0"},
    )

    lic = None
    attrib = None

    m1 = LICENSE_CODE_RE.search(html)
    if m1:
        lic = (m1.group(1) or "").strip().lower()

    m2 = ATTRIBUTION_RE.search(html)
    if m2:
        attrib = m2.group(1)

    return lic, attrib


# -----------------------------
# Output
# -----------------------------
def load_existing_out(path: str) -> Dict[str, Any]:
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


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--taxon_url", type=str, default=None)
    ap.add_argument("--taxon_id", type=int, default=None)
    ap.add_argument("--taxon_name", type=str, default=None)
    ap.add_argument("--taxa_tree_json", type=str, default=None)
    ap.add_argument("--max_obs", type=int, default=500)

    ap.add_argument("--strict_exact", action="store_true")

    ap.add_argument(
        "--size",
        type=str,
        default="large",
        choices=["square", "small", "medium", "large", "original"],
    )
    ap.add_argument(
        "--photo_license",
        type=str,
        default=None,
        help="Allowed: cc-by,cc-by-sa,...",
    )
    ap.add_argument("--max_photos_per_taxon", type=int, default=5)

    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--backoff_base", type=float, default=1.0)
    ap.add_argument("--polite_delay", type=float, default=1.0)

    ap.add_argument("--out", type=str, default="taxa_photos.json")
    ap.add_argument("--update_out", action="store_true")
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    allowed_licenses = normalize_license_list(args.photo_license)
    print(f"Licenses: {allowed_licenses}")

    taxa_to_fetch: List[Tuple[str, str]] = []

    if args.taxon_url:
        taxa_to_fetch.append((args.taxon_url, args.taxon_url))

    if args.taxon_id is not None:
        taxa_to_fetch.append(
            (f"taxon_id:{args.taxon_id}", taxon_url_from_id(args.taxon_id))
        )

    if args.taxon_name:
        with requests.Session() as session:
            tid = resolve_taxon_name_exact_first(
                session,
                args.taxon_name,
                timeout=args.timeout,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                strict_exact=args.strict_exact,
            )
        if not tid:
            print(
                f"Could not resolve taxon name: {args.taxon_name}",
                file=sys.stderr,
            )
            return 2
        taxa_to_fetch.append((args.taxon_name, taxon_url_from_id(tid)))
    taxon_num_id = None

    if args.taxa_tree_json:
        with open(args.taxa_tree_json, "r", encoding="utf-8") as f:
            tree = json.load(f)
        queries = extract_taxon_queries(tree)
        with requests.Session() as session:
            for q in queries:
                tid = resolve_taxon_name_exact_first(
                    session,
                    q,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    backoff_base=args.backoff_base,
                    strict_exact=args.strict_exact,
                )
                if tid:
                    taxa_to_fetch.append((q, taxon_url_from_id(tid)))
                    taxon_num_id = tid
                else:
                    print(f"Could not resolve (skipping): {q}", file=sys.stderr)

    # dedupe keys
    seen = set()
    taxa_to_fetch = [
        (k, u) for (k, u) in taxa_to_fetch if not (k in seen or seen.add(k))
    ]

    if not taxa_to_fetch:
        print("No taxa to fetch.", file=sys.stderr)
        return 2

    existing = (
        load_existing_out(args.out)
        if (args.update_out or args.resume)
        else {"taxa": {}}
    )
    existing_taxa = existing.get("taxa", {})

    per_taxon_results: Dict[str, Any] = {}

    with requests.Session() as session:
        for idx, (taxon_key, taxon_page_url) in enumerate(
            taxa_to_fetch, start=1
        ):
            if args.resume and taxon_key in existing_taxa:
                print(
                    f"[{idx}/{len(taxa_to_fetch)}] SKIP (resume): {taxon_key}",
                    file=sys.stderr,
                )
                continue

            print(
                f"[{idx}/{len(taxa_to_fetch)}] TAXON: {taxon_key}",
                file=sys.stderr,
            )
            html = get_text_with_retries(
                session,
                taxon_page_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                headers={"User-Agent": "inat-taxon-coverimage-scraper/1.0"},
            )
            time.sleep(args.polite_delay)

            photo_json = extract_photo_json(html)
            if photo_json is None:
                print(
                    f"No photo data for taxon: {taxon_key} at {taxon_page_url}"
                )
                continue
            photo_data = extract_photo_from_json(photo_json)
            photo_data = [
                photo
                for photo in photo_data
                if photo["license_code"] in allowed_licenses
            ]
            photo_data = photo_data[: args.max_obs]
            if len(photo_data) == 0:
                print(
                    f"No photos with allowed licences for taxon: {taxon_key} at {taxon_page_url}"
                )
                continue

            photo_records: List[Dict[str, Any]] = []

            for photo in photo_data:
                photo_url = photo["original_url"]
                try:
                    tid = photo_json["id"]
                    pid = photo["id"]
                    lic = photo["license_code"]
                    attrib = photo["attribution"]
                    taxa_url = f"https://www.inaturalist.org/taxa/{tid}"
                    photo_page_url = f"https://www.inaturalist.org/photos/{pid}"
                    rec = {
                        "taxon_key": taxon_key,
                        "taxon_page_url": taxa_url,
                        "photo_id": pid,
                        "photo_page_url": photo_page_url,
                        "license_code": lic,
                        "attribution": attrib,
                        "url_found": photo_page_url,
                        "url": upgrade_photo_url(photo_url, args.size),
                        "url_square": upgrade_photo_url(photo_url, "square"),
                        "url_small": upgrade_photo_url(photo_url, "small"),
                        "url_medium": upgrade_photo_url(photo_url, "medium"),
                        "url_large": upgrade_photo_url(photo_url, "large"),
                        "url_original": upgrade_photo_url(
                            photo_url, "original"
                        ),
                        "retrieved_from": "taxon_page_coverimage_id_or_style",
                    }
                    photo_records.append(rec)
                except Exception as e:
                    print(e)
                    purl = f"https://www.inaturalist.org/taxa/{taxon_num_id}"
                    print(f"No data for {taxon_key} at {purl}")
                    continue

            per_taxon_results[taxon_key] = {
                "taxon_key": taxon_key,
                "taxon_page_url": taxon_page_url,
                "count_photos": len(photo_records),
                "photos": photo_records,
                "updated_at_utc": utc_now_iso(),
                "filters": {
                    "size": args.size,
                    "allowed_licenses": allowed_licenses,
                    "max_photos_per_taxon": args.max_photos_per_taxon,
                },
            }

            if args.update_out:
                existing_taxa.update({taxon_key: per_taxon_results[taxon_key]})
                existing["taxa"] = existing_taxa
                existing["updated_at_utc"] = utc_now_iso()
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

    if args.update_out:
        existing_taxa.update(per_taxon_results)
        existing["taxa"] = existing_taxa
        existing["updated_at_utc"] = utc_now_iso()
        out_payload = existing
    else:
        out_payload = {
            "taxa": per_taxon_results,
            "updated_at_utc": utc_now_iso(),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    total_photos = sum(
        v.get("count_photos", 0) for v in per_taxon_results.values()
    )
    print(
        f"Done. Updated {len(per_taxon_results)} taxa; kept {total_photos} photos. Output: {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

