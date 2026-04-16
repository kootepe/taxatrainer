"""
Laji.fi API client.

Handles:
  - Taxon search by scientific name
  - Media (photo) fetching per taxon
  - Photo record building with multi-size URLs
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests

from common import check_partition, SUBSP_PARTITIONS


# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://api.laji.fi"
DEFAULT_TOKEN = (
    "dc4db51295478ce6fcac891f8777d617d080bd3d54c03dd2174733affb2d8fce"
)

ACCEPTED_LICENSES = {"CC-BY-NC-ND-4.0", "CC-BY-NC-4.0"}
# markku lehtonen has pictures of dried specimen and they dont have an extra
# label to filter them out
DISABLED_AUTHORS = {"Markku Lehtonen"}


# ── URL helpers ──────────────────────────────────────────────────────────────


def upgrade_photo_url(url: str, size: str) -> str:
    """
    Resize a laji.fi photo URL.

    Patterns:
      full:  .../<name>.jpg        (no suffix → original resolution)
      sized: .../<name>_large.jpg  (_<size> inserted before extension)
    """
    if not url:
        return url
    # Strip any existing size suffix to get the base
    base = re.sub(r"_(large|medium|small|square|thumb)(\.\w+)$", r"\2", url)
    if size == "full":
        return base
    root, ext = os.path.splitext(base)
    return f"{root}_{size}{ext}"


def taxon_page_url(taxon_id: str) -> str:
    return f"https://www.laji.fi/taxon/{taxon_id}"


# ── Client ───────────────────────────────────────────────────────────────────


class LajiClient:
    """Stateless client wrapping the laji.fi REST API."""

    def __init__(self, token: str = DEFAULT_TOKEN, timeout: int = 15):
        self.token = token
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "API-version": "1",
        }

    # -- taxon search ------------------------------------------------------

    def search_taxon(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a taxon by scientific name.
        Returns the best-matching result dict, or None.
        """
        params = {"query": name, "languages": "suomi,finnish"}
        url = f"{BASE_URL}/taxa/search"
        resp = requests.get(
            url, headers=self.headers, params=params, timeout=self.timeout
        )
        resp.raise_for_status()

        results = resp.json().get("results", [])
        if not results:
            return None

        # Prefer exact scientific-name match
        for item in results:
            if item.get("matchingName", "").lower() == name.lower():
                return item

        return results[0] if results else None

    # -- media fetching ----------------------------------------------------

    def fetch_media(self, taxon_id: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch media items for a taxon ID.  Returns list or None."""
        url = f"{BASE_URL}/taxa/{taxon_id}/media"
        resp = requests.get(url, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("results", None)

    # -- combined convenience method ---------------------------------------

    def fetch_photos(
        self,
        latin_name: str,
        *,
        max_photos: int = 500,
    ) -> tuple[List[Dict[str, Any]], str, Optional[str]]:
        """
        Search for *latin_name* on laji.fi, fetch its media, build photo
        records.

        Returns (photo_records, taxon_page_url, taxon_id).
        Raises SystemExit if the taxon is not found (preserving original
        behaviour — callers may want to change this).
        """
        search_name = check_partition(latin_name, SUBSP_PARTITIONS)
        taxon = self.search_taxon(search_name)

        if not taxon:
            print(
                f"Laji.fi: no taxon found for '{latin_name}'", file=sys.stderr
            )
            return [], "", None

        tid = taxon.get("id")
        sci_name = taxon.get("scientificName", latin_name)
        vernacular = taxon.get("vernacularName", "")
        vn_display = vernacular

        print(
            f"Laji.fi found: {sci_name} (ID: {tid})"
            f"{'  —  ' + vn_display if vn_display else ''}",
            file=sys.stderr,
        )

        media_items = self.fetch_media(tid)
        if not media_items:
            print(f"Laji.fi: no images for {sci_name}", file=sys.stderr)
            return [], taxon_page_url(tid), tid

        print(f"Laji.fi: {len(media_items)} media item(s)", file=sys.stderr)

        records = build_photo_records(media_items, tid, latin_name, "full")
        records = records[:max_photos]
        return records, taxon_page_url(tid), tid


# ── Photo record builder ────────────────────────────────────────────────────


def build_photo_records(
    media_items: List[Dict[str, Any]],
    taxon_id: str,
    taxon_key: str,
    size: str,
) -> List[Dict[str, Any]]:
    """Convert laji.fi media items into standardised photo records."""
    records: List[Dict[str, Any]] = []
    taxa_url = taxon_page_url(taxon_id)

    for item in media_items:
        try:
            lic = item.get("licenseAbbreviation")
            has_type = item.get("type")
            is_primary = item.get("primaryForTaxon")
            author = item.get("author", "")
            if lic not in ACCEPTED_LICENSES:
                print(
                    f"Laji.fi: skipping license {lic} (not in {ACCEPTED_LICENSES})"
                )
                continue
            if has_type:
                if "microsco" in has_type.lower():
                    continue
            if author in DISABLED_AUTHORS:
                continue

            pid = item.get("id")
            full_url = item.get("fullURL", "")

            img_data = {
                "taxon_key": taxon_key,
                "taxon_page_url": taxa_url,
                "photo_id": pid,
                "photo_page_url": full_url,
                "license_code": lic,
                "attribution": f"(c) {author} {lic}",
                "url": upgrade_photo_url(full_url, size),
                "url_square": upgrade_photo_url(full_url, "thumb"),
                "url_small": upgrade_photo_url(full_url, "large"),
                "url_medium": upgrade_photo_url(full_url, "large"),
                "url_large": upgrade_photo_url(full_url, "large"),
                "url_original": upgrade_photo_url(full_url, "full"),
                "retrieved_from": "laji.fi",
            }
            if is_primary:
                records.insert(0, img_data)
            else:
                records.append(img_data)
        except Exception as e:
            print(f"Error building laji.fi photo record: {e}", file=sys.stderr)

    return records
