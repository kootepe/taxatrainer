from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import timedelta
from collections import Counter, defaultdict

from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


app.config.update(
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=365),
    SESSION_COOKIE_SAMESITE="Lax",
    # SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "1") == "1",
)

# ---------- taxonomy config ----------
RANK_FI = {
    "phylum": "pääjakso",
    "class": "luokka",
    "order": "lahko",
    "family": "heimo",
    "genus": "suku",
    "species": "laji",
}

RANK_KEYS = ["phylum", "class", "order", "family", "genus", "species"]
RANK_INDEX = {r: i for i, r in enumerate(RANK_KEYS)}

DEFAULT_ENABLED_RANKS = [
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

PLANT_DATASETS = [
    "Helsinki_ME_kasvit",
    "UEF_kasvit_kaksisirkkaiset",
    "UEF_kasvit_yksisirkkaiset",
    "UEF_kasvit_itiokasvit",
    "UEF_kasvit_suppea",
]

IMAGE_DATASET_MAP = {
    "Laji.fi": "static/image_datasets/taxa_photos_laji.json",
    "iNaturalist": "static/image_datasets/taxa_photos.json",
}

PICTURE_SOURCE = ["laji", "inat"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DEFAULT_DATASET = "selkarangattomat"
DEFAULT_IMAGE_DATASET = "taxa_photos"
DATASET_DIR = Path("static/datasets/")
IMAGE_DATASET_DIR = Path("static/image_datasets/")

# JSON uses "specie" as an array key
REAL_CHILD_KEYS = {"class", "order", "family", "specie"}
META_KEYS = {"lat", "fin", "req", "image"}


# ---------- models ----------
@dataclass
class StudyItem:
    answer: Dict[str, str]  # phylum/class/order/family/genus/species
    meta: Dict[str, Any]  # fin/req/image
    node_id: str


def bump_card_counter() -> int:
    n = int(session.get("card_counter", 0) or 0) + 1
    session["card_counter"] = n
    return n


def get_hint_language() -> str:
    v = session.get("hint_language")
    return v if v in ("fin", "lat") else "fin"


def get_img_toggle() -> bool:
    v = session.get("img_toggle")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v in ("1", "true", "on", "yes")
    return True  # default ON


def get_photo_mode() -> bool:
    v = session.get("photo_mode")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v in ("1", "true", "on", "yes")
    return False  # default OFF


# ---------- transparent container traversal (Option C) ----------
def looks_taxonomic_container(node: Dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False

    for k in REAL_CHILD_KEYS:
        v = node.get(k)
        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
            return True

    for k, v in node.items():
        if k in REAL_CHILD_KEYS or k in META_KEYS:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            if any(looks_taxonomic_container(x) for x in v):
                return True

    return False


def get_pool_total(items: List[StudyItem]) -> int:
    try:
        return len(build_pool_indices(items))
    except Exception:
        return len(items)


def iter_transparent_children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(node, dict):
        return out

    for k, v in node.items():
        if k in REAL_CHILD_KEYS or k in META_KEYS:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            if any(looks_taxonomic_container(x) for x in v):
                out.extend([x for x in v if isinstance(x, dict)])
    return out


def collect_real_children_through_containers(
    node: Dict[str, Any], key: str
) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    direct = node.get(key)
    if isinstance(direct, list):
        found.extend([x for x in direct if isinstance(x, dict)])

    for t in iter_transparent_children(node):
        found.extend(collect_real_children_through_containers(t, key))

    return found


def collect_species_lists(node: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    v = node.get("specie")
    if isinstance(v, list):
        out.append(v)
    for t in iter_transparent_children(node):
        out.extend(collect_species_lists(t))
    return out


# ---------- session helpers ----------
def list_datasets() -> list[str]:
    files = sorted(DATASET_DIR.glob("*.json"))
    return [f.stem for f in files]


def list_image_datasets() -> list[str]:
    files = sorted(IMAGE_DATASET_DIR.glob("*.json"))
    return [f.stem for f in files]


def get_selected_dataset() -> str:
    name = session.get("dataset_name")
    all_sets = list_datasets()
    if isinstance(name, str) and name in all_sets:
        return name
    return (
        DEFAULT_DATASET
        if DEFAULT_DATASET in all_sets
        else (all_sets[0] if all_sets else DEFAULT_DATASET)
    )


def get_selected_image_dataset() -> str:
    name = session.get("image_dataset_name")
    all_sets = list_image_datasets()
    if isinstance(name, str) and name in all_sets:
        return name
    return (
        DEFAULT_IMAGE_DATASET
        if DEFAULT_IMAGE_DATASET in all_sets
        else (all_sets[0] if all_sets else DEFAULT_IMAGE_DATASET)
    )


def get_enabled_ranks() -> List[str]:
    enabled = session.get("enabled_ranks")

    if not isinstance(enabled, list) or not all(
        isinstance(x, str) for x in enabled
    ):
        enabled = list(DEFAULT_ENABLED_RANKS)
    else:
        enabled = [r for r in RANK_KEYS if r in enabled]
        if not enabled:
            enabled = list(DEFAULT_ENABLED_RANKS)

    has_genus = "genus" in enabled
    has_species = "species" in enabled
    if has_genus ^ has_species:
        enabled = [r for r in enabled if r not in ("genus", "species")]
        enabled += ["genus", "species"]
        enabled = [r for r in RANK_KEYS if r in enabled]

    return enabled


def safe_lat(node: Dict[str, Any]) -> str:
    v = node.get("lat", "")
    return v.strip() if isinstance(v, str) else ""


def safe_fin(node: Dict[str, Any]) -> str:
    v = node.get("fin", "")
    return v.strip() if isinstance(v, str) else ""


def make_node_id(parent_id: str, rank: str, lat: str) -> str:
    part = f"{rank}:{lat}"
    return part if not parent_id else f"{parent_id}>{part}"


def is_allowed(node_id: str, enabled: Set[str]) -> bool:
    for e in enabled:
        if node_id == e or node_id.startswith(e + ">"):
            return True
    return False


def fin_or_lat(fin: Any, lat: Any) -> str:
    fin_s = str(fin or "").strip()
    if fin_s:
        return fin_s
    return str(lat or "").strip()


def first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return None


def find_nearest_image(path_nodes: List[Dict[str, Any]]) -> Optional[str]:
    for node in reversed(path_nodes):
        img = node.get("image")
        if isinstance(img, str) and img.strip():
            return img.strip()
    return None


# ---------- data loading ----------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=16)
def load_dataset_cached(stem: str, image_dataset_stem: str):
    """Load and extract items from a dataset.

    Both `stem` (the species dataset) and `image_dataset_stem` (the photo
    dataset) are part of the cache key, so changing either one triggers a
    fresh load.
    """
    path = DATASET_DIR / f"{stem}.json"
    data = load_json(str(path))

    taxa_photos = load_taxa_photos_cached(image_dataset_stem)

    items = extract_items(data, taxa_photos)

    node_images = build_node_images_index(items)
    has_desc = build_has_descendants(items)
    for it in items:
        if it.node_id in has_desc:
            if it.meta.get("_kind") == "node":
                imgs = node_images.get(it.node_id)
                if imgs:
                    it.meta["images"] = imgs
                    it.meta["image"] = (
                        imgs[0].get("url") or imgs[0].get("url_square") or ""
                    )
            else:
                it.meta["images"] = it.meta.get("images", []) or []

    taxa_tree = build_taxa_tree(data)
    return data, items, taxa_tree


def _load_current_dataset():
    """Convenience: load with both the species and image dataset from session."""
    return load_dataset_cached(
        get_selected_dataset(), get_selected_image_dataset()
    )


def get_actions_position() -> str:
    v = session.get("actions_position")
    return v if v in ("top", "bottom") else "top"


# ---------- taxa tree + enabled nodes ----------
def build_taxa_tree(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    phyla = data.get("phylum", [])
    out: List[Dict[str, Any]] = []
    if not isinstance(phyla, list):
        return out

    def build_children(
        node: Dict[str, Any], parent_id: str, rank: str
    ) -> Dict[str, Any]:
        lat = safe_lat(node) or "UNKNOWN"
        fin = safe_fin(node) or "Ei suomenkielistä nimeä"
        node_id = make_node_id(parent_id, rank, lat)

        tnode = {
            "rank": RANK_FI.get(rank),
            "lat": lat,
            "fin": fin,
            "id": node_id,
            "children": [],
        }

        for ck in ["class", "order", "family"]:
            child_list = collect_real_children_through_containers(node, ck)
            for ch in child_list:
                if isinstance(ch, dict):
                    tnode["children"].append(build_children(ch, node_id, ck))

        return tnode

    for ph in phyla:
        if isinstance(ph, dict):
            out.append(build_children(ph, "", "phylum"))

    return out


def get_all_node_ids(dataset: str) -> Set[str]:
    _, _, taxa_tree = load_dataset_cached(dataset, get_selected_image_dataset())
    ids: Set[str] = set()

    def walk(n: Dict[str, Any]):
        ids.add(n["id"])
        for ch in n.get("children", []):
            walk(ch)

    for root in taxa_tree:
        walk(root)

    return ids


def get_enabled_nodes() -> Set[str]:
    raw = session.get("enabled_nodes")
    if isinstance(raw, list):
        s = set([x for x in raw if isinstance(x, str)])
        if s:
            return s

    dataset = get_selected_dataset()
    return get_all_node_ids(dataset)


# ---------- item extraction ----------
def extract_items(
    data: Dict[str, Any], taxa_photos: Optional[Dict[str, Any]] = None
) -> List[StudyItem]:
    items: List[StudyItem] = []

    phyla = data.get("phylum", [])
    if not isinstance(phyla, list):
        return items

    def node_has_children(node: Dict[str, Any]) -> bool:
        if collect_real_children_through_containers(node, "class"):
            return True
        if collect_real_children_through_containers(node, "order"):
            return True
        if collect_real_children_through_containers(node, "family"):
            return True

        for specie_list in collect_species_lists(node):
            if isinstance(specie_list, list) and len(specie_list) > 0:
                return True

        return False

    def make_node_item(
        node: Dict[str, Any],
        path: Dict[str, str],
        path_fin: Dict[str, str],
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        ans = {k: "" for k in RANK_KEYS}
        ans.update(path)

        taxon_key = (path.get("genus") or "").strip()
        if not taxon_key:
            taxon_key = safe_lat(node)
        images = get_images_for_taxon(taxa_photos or {}, taxon_key)

        meta = {
            "_kind": "node",
            "has_children": node_has_children(node),
            "fin": fin_or_lat(
                node.get("fin"),
                path.get("family")
                or path.get("order")
                or path.get("class")
                or path.get("phylum"),
            ),
            "fin_by_rank": dict(path_fin),
        }

        meta["images"] = [
            {
                "url": (p.get("url") or "").strip(),
                "url_square": (p.get("url_square") or "").strip(),
                "attribution": (p.get("attribution") or "").strip(),
                "photo_page_url": (p.get("photo_page_url") or "").strip(),
                "observation_url": (p.get("observation_url") or "").strip(),
                "license_code": (p.get("license_code") or "").strip(),
            }
            for p in images
        ]

        items.append(StudyItem(answer=ans, meta=meta, node_id=current_node_id))

    def walk(
        node: Dict[str, Any],
        rank: str,
        path: Dict[str, str],
        path_fin: Dict[str, str],
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        if rank in ("phylum", "class", "order", "family"):
            make_node_item(node, path, path_fin, path_nodes, current_node_id)

        for specie_list in collect_species_lists(node):
            for sp in specie_list:
                if not isinstance(sp, dict):
                    continue

                genus = str(sp.get("genus", "") or "")
                species = str(sp.get("species", "") or "")

                ans = {k: "" for k in RANK_KEYS}
                ans.update(path)
                ans["genus"] = genus
                ans["species"] = species
                fin_by_rank = dict(path_fin)
                sp_fin = str(sp.get("fin") or "").strip()
                if sp_fin:
                    fin_by_rank["species"] = sp_fin

                taxon_key = genus.strip() or f"{genus} {species}".strip()
                taxon_key = (
                    genus.strip()
                    if species.strip() == ""
                    else f"{genus} {species}"
                )
                images = get_images_for_taxon(taxa_photos or {}, taxon_key)
                meta = {
                    "_kind": "specie",
                    "fin": fin_or_lat(
                        sp.get("fin"), f"{genus} {species}".strip()
                    ),
                    "fin_by_rank": fin_by_rank,
                    "req": sp.get("req", ""),
                    "image": first_nonempty(
                        sp.get("image"),
                        find_nearest_image(path_nodes + [sp]),
                    ),
                }

                meta["images"] = [
                    {
                        "url": (p.get("url") or "").strip(),
                        "url_square": (p.get("url_square") or "").strip(),
                        "attribution": (p.get("attribution") or "").strip(),
                        "photo_page_url": (
                            p.get("photo_page_url") or ""
                        ).strip(),
                        "observation_url": (
                            p.get("observation_url") or ""
                        ).strip(),
                        "license_code": (p.get("license_code") or "").strip(),
                    }
                    for p in images
                ]

                items.append(
                    StudyItem(answer=ans, meta=meta, node_id=current_node_id)
                )

        for ck in ["class", "order", "family"]:
            child_list = collect_real_children_through_containers(node, ck)
            if not child_list:
                continue

            for ch in child_list:
                if not isinstance(ch, dict):
                    continue

                lat = safe_lat(ch)
                if not lat:
                    continue

                new_path = dict(path)
                new_path[ck] = lat

                new_path_fin = dict(path_fin)
                ch_fin = safe_fin(ch)
                if ch_fin:
                    new_path_fin[ck] = ch_fin

                new_node_id = make_node_id(current_node_id, ck, lat)
                walk(
                    ch,
                    ck,
                    new_path,
                    new_path_fin,
                    path_nodes + [ch],
                    new_node_id,
                )

    for ph in phyla:
        if not isinstance(ph, dict):
            continue

        path: Dict[str, str] = {k: "" for k in RANK_KEYS}
        path_fin = {k: "" for k in RANK_KEYS}
        ph_lat = safe_lat(ph)
        if ph_lat:
            path["phylum"] = ph_lat

        ph_fin = safe_fin(ph)
        if ph_fin:
            path_fin["phylum"] = ph_fin

        ph_id = make_node_id("", "phylum", ph_lat if ph_lat else "UNKNOWN")
        walk(ph, "phylum", path, path_fin, [ph], ph_id)

    return items


@lru_cache(maxsize=8)
def load_taxa_photos_cached(stem: str) -> Dict[str, Any]:
    path = IMAGE_DATASET_DIR / f"{stem}.json"
    if not path.exists():
        return {}
    return load_json(str(path))


def normalize_taxon_key(s: str) -> str:
    return (s or "").strip().lower()


def get_images_for_taxon(
    taxa_photos: Dict[str, Any], taxon_name: str
) -> List[Dict[str, Any]]:
    if not isinstance(taxa_photos, dict):
        return []
    taxa = taxa_photos.get("taxa")
    if not isinstance(taxa, dict):
        return []

    key = normalize_taxon_key(taxon_name)
    entry = taxa.get(key)
    if not isinstance(entry, dict):
        return []

    photos = entry.get("photos", [])
    if not isinstance(photos, list):
        return []

    out = []
    for p in photos:
        if not isinstance(p, dict):
            continue
        url = p.get("url_medium") or ""
        sq = p.get("url_square") or ""
        if isinstance(url, str) and url.strip():
            out.append(p)
        elif isinstance(sq, str) and sq.strip():
            out.append(p)
    return out


def pick_taxon_key_for_item(item: StudyItem) -> Optional[str]:
    ans = item.answer
    g = (ans.get("genus") or "").strip()
    if g:
        return g

    for r in reversed(RANK_KEYS):
        v = (ans.get(r) or "").strip()
        if v:
            return v
    return None


# ---------- quiz logic ----------
def make_quiz_payload(item: StudyItem) -> Dict[str, Any]:
    enabled_ranks = get_enabled_ranks()

    present = [r for r in enabled_ranks if item.answer.get(r, "").strip()]

    payload = {
        "answer": item.answer,
        "meta": item.meta,
        "mode": "taxonomy",
        "given": None,
        "quiz_ranks": present if present else enabled_ranks,
    }

    if len(present) < 2:
        return payload

    given_rank = min(present, key=lambda r: RANK_INDEX[r])

    payload["given"] = {"rank": given_rank, "value": item.answer[given_rank]}
    payload["mode"] = "higher_from_subtaxon"
    payload["quiz_ranks"] = present
    return payload


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def build_pool_indices(items: List[StudyItem]) -> List[int]:
    enabled_ranks = get_enabled_ranks()
    max_enabled_idx = max(RANK_INDEX[r] for r in enabled_ranks)

    def deepest_filled_rank_idx(it: StudyItem) -> int:
        ans = it.answer

        if it.meta.get("_kind") == "specie":
            if (
                "genus" in enabled_ranks
                and "species" in enabled_ranks
                and ans.get("genus", "").strip()
            ):
                return RANK_INDEX["species"]

        for r in reversed(RANK_KEYS):
            if ans.get(r, "").strip():
                return RANK_INDEX[r]
        return -1

    def has_any_enabled_value(it: StudyItem) -> bool:
        ans = it.answer

        if "genus" in enabled_ranks and "species" in enabled_ranks:
            if ans.get("genus", "").strip():
                return True

        for r in enabled_ranks:
            if r in ("genus", "species"):
                continue
            if ans.get(r, "").strip():
                return True

        return False

    pool = [
        i
        for i, it in enumerate(items)
        if deepest_filled_rank_idx(it) <= max_enabled_idx
        and has_any_enabled_value(it)
    ]

    if not pool:
        return []

    enabled_nodes = get_enabled_nodes()
    if enabled_nodes:
        depth_ok = [
            i for i in pool if is_allowed(items[i].node_id, enabled_nodes)
        ]

    if not depth_ok:
        return []

    eligible_node_ids = [items[i].node_id for i in depth_ok]

    def has_deeper_descendant(node_id: str) -> bool:
        prefix = node_id + ">"
        for j in depth_ok:
            jt = items[j]
            nid = jt.node_id
            if nid.startswith(prefix):
                return True
            if nid == node_id and jt.meta.get("_kind") == "specie":
                return True
        return False

    final = []
    for i in depth_ok:
        it = items[i]
        if it.meta.get("_kind") == "node":
            if has_deeper_descendant(it.node_id):
                continue
        final.append(i)

    return final


def node_ancestors(node_id: str) -> List[str]:
    parts = node_id.split(">")
    out = []
    for i in range(1, len(parts) + 1):
        out.append(">".join(parts[:i]))
    return out


def dedupe_photos(
    photos: List[Dict[str, Any]], limit: int = 24
) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in photos:
        url = (p.get("url") or p.get("url_square") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def build_node_images_index(
    items: List[StudyItem],
) -> Dict[str, List[Dict[str, Any]]]:
    node_images: Dict[str, List[Dict[str, Any]]] = {}

    for it in items:
        photos = it.meta.get("images")
        if not isinstance(photos, list) or not photos:
            continue

        for anc in node_ancestors(it.node_id):
            if anc == it.node_id:
                continue
            node_images.setdefault(anc, []).extend(photos)

        if it.meta.get("_kind") == "specie":
            node_images.setdefault(it.node_id, []).extend(photos)

    for k in list(node_images.keys()):
        node_images[k] = dedupe_photos(node_images[k], limit=24)

    return node_images


def build_has_descendants(items: list[StudyItem]) -> set[str]:
    ids = [it.node_id for it in items if isinstance(it.node_id, str)]
    ids.sort()
    has: set[str] = set()

    for i, nid in enumerate(ids):
        prefix = nid + ">"
        for j in range(i + 1, len(ids)):
            if ids[j].startswith(prefix):
                has.add(nid)
                break
            if ids[j] > prefix and not ids[j].startswith(prefix):
                continue

    by_nid = defaultdict(set)
    for it in items:
        kind = it.meta.get("_kind", "")
        if kind:
            by_nid[it.node_id].add(kind)
    for nid, kinds in by_nid.items():
        if "node" in kinds and "specie" in kinds:
            has.add(nid)

    return has


def get_deck_key(dataset: str) -> str:
    ranks_key = ",".join(get_enabled_ranks())
    nodes = sorted(get_enabled_nodes())
    nodes_key = str(hash(tuple(nodes)))
    return f"{dataset}|{ranks_key}|{nodes_key}"


def draw_item_from_deck(items: List[StudyItem]) -> StudyItem:
    dataset = get_selected_dataset()
    deck_key = get_deck_key(dataset)

    if session.get("deck_key") != deck_key:
        session["deck_key"] = deck_key
        session.pop("deck", None)

    deck = session.get("deck")
    if not isinstance(deck, list) or not deck:
        session["card_counter"] = 0
        pool = build_pool_indices(items)
        if not pool:
            raise RuntimeError("No eligible items for current settings.")
        random.shuffle(pool)
        deck = pool
        session["deck"] = deck

    next_idx = int(deck.pop())
    session["deck"] = deck
    return items[next_idx]


def choose_item() -> StudyItem:
    _, items, _ = _load_current_dataset()
    if not items:
        dataset = get_selected_dataset()
        raise RuntimeError(f"No study items extracted from {dataset}.json")
    return draw_item_from_deck(items)


def get_car_mode() -> bool:
    v = session.get("car_mode")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v in ("1", "true", "on", "yes")
    return False


# ---------- routes ----------
@app.get("/")
def index():
    session.pop("deck", None)
    session.pop("deck_key", None)
    session.pop("current_idx", None)
    session["card_counter"] = 0

    _, items, _ = _load_current_dataset()
    total = get_pool_total(items)

    item = choose_item()
    idx = items.index(item)
    session["current_idx"] = idx

    counter = bump_card_counter()

    payload = make_quiz_payload(item)
    payload["counter"] = counter
    payload["total"] = total

    actions_position = get_actions_position()
    return render_template(
        "index.html",
        item=item,
        ranks=payload["quiz_ranks"],
        rank_fi=RANK_FI,
        quiz_mode=payload["mode"],
        given=payload["given"],
        counter=counter,
        total=total,
        actions_position=actions_position,
        hint_language=get_hint_language(),
        img_toggle=get_img_toggle(),
        car_mode=get_car_mode(),
        photo_mode=get_photo_mode(),
    )


@app.get("/manual")
def manual():
    return render_template("manual.html")


@app.post("/new")
def new_item():
    _, items, _ = _load_current_dataset()

    item = choose_item()
    session["current_idx"] = items.index(item)

    counter = bump_card_counter()
    total = get_pool_total(items)

    payload = make_quiz_payload(item)
    payload["counter"] = counter
    payload["total"] = total
    payload["hint_language"] = get_hint_language()
    payload["img_toggle"] = get_img_toggle()
    payload["photo_mode"] = get_photo_mode()
    return jsonify(payload)


@app.post("/check")
def check():
    _, items, _ = _load_current_dataset()

    idx = session.get("current_idx")
    if idx is None or not (0 <= int(idx) < len(items)):
        return jsonify({"error": "no active item"}), 400

    item = items[int(idx)]
    answer_dict = item.answer

    payload = request.get_json(force=True) or {}
    inputs = payload.get("inputs", {})

    quiz_ranks = payload.get("quiz_ranks")
    if not isinstance(quiz_ranks, list):
        quiz_ranks = get_enabled_ranks()
    else:
        quiz_ranks = [r for r in RANK_KEYS if r in quiz_ranks]

    result = {}
    for rank in quiz_ranks:
        expected = norm(answer_dict.get(rank, ""))
        got = norm(str(inputs.get(rank, "")))

        if expected == "":
            result[rank] = {
                "correct": None,
                "expected": answer_dict.get(rank, ""),
            }
        else:
            result[rank] = {
                "correct": (got == expected),
                "expected": answer_dict.get(rank, ""),
            }

    return jsonify(result)


@app.get("/settings")
def settings():
    enabled = set(get_enabled_ranks())
    datasets = list_datasets()
    selected = get_selected_dataset()
    image_datasets = list_image_datasets()
    selected_image_dataset = get_selected_image_dataset()

    return render_template(
        "settings.html",
        all_ranks=RANK_KEYS,
        enabled=enabled,
        datasets=datasets,
        selected_dataset=selected,
        rank_fi=RANK_FI,
        actions_position=get_actions_position(),
        hint_language=get_hint_language(),
        img_toggle=get_img_toggle(),
        car_mode=get_car_mode(),
        photo_mode=get_photo_mode(),
        image_datasets=image_datasets,
        selected_image_dataset=selected_image_dataset,
    )


@app.post("/settings")
def save_settings():
    session.permanent = True
    selected_ranks = request.form.getlist("ranks")
    selected_ranks = [r for r in RANK_KEYS if r in selected_ranks]

    gs_on = request.form.get("ranks_genus_species") == "1"
    if gs_on:
        if "genus" not in selected_ranks:
            selected_ranks.append("genus")
        if "species" not in selected_ranks:
            selected_ranks.append("species")
    else:
        selected_ranks = [
            r for r in selected_ranks if r not in ("genus", "species")
        ]

    if not selected_ranks:
        selected_ranks = DEFAULT_ENABLED_RANKS

    selected_ranks = [r for r in RANK_KEYS if r in selected_ranks]
    session["enabled_ranks"] = selected_ranks

    datasets = list_datasets()
    selected = get_selected_dataset()
    image_datasets = list_image_datasets()
    selected_image_dataset = get_selected_image_dataset()

    pos = request.form.get("actions_position", "top")
    session["actions_position"] = pos if pos in ("top", "bottom") else "top"

    session.pop("deck", None)
    session.pop("deck_key", None)
    hint_lang = request.form.get("hint_language", "fin")
    session["hint_language"] = (
        hint_lang if hint_lang in ("fin", "lat") else "fin"
    )
    img_toggle = request.form.get("img_toggle") == "1"
    session["img_toggle"] = img_toggle
    session["car_mode"] = request.form.get("car_mode") == "1"
    session["photo_mode"] = request.form.get("photo_mode") == "1"

    return render_template(
        "settings.html",
        all_ranks=RANK_KEYS,
        enabled=set(selected_ranks),
        datasets=datasets,
        selected_dataset=selected,
        image_datasets=image_datasets,
        selected_image_dataset=selected_image_dataset,
        rank_fi=RANK_FI,
        actions_position=session.get("actions_position", "top"),
        hint_language=get_hint_language(),
        img_toggle=img_toggle,
        saved=True,
        car_mode=request.form.get("car_mode") == "1",
        photo_mode=request.form.get("photo_mode") == "1",
    )


@app.get("/settings/dataset")
def set_dataset():
    d = request.args.get("d", "")
    if isinstance(d, str) and d in list_datasets():
        prev = session.get("dataset_name", DEFAULT_DATASET)

        session["dataset_name"] = d
        session.pop("enabled_nodes", None)
        session.pop("current_idx", None)

        session.pop("deck", None)
        session.pop("deck_key", None)

        session.pop("card_counter", None)

        if d in PLANT_DATASETS and prev not in PLANT_DATASETS:
            session.permanent = True
            session["enabled_ranks"] = ["genus", "species"]

    return ("", 302, {"Location": "/settings"})


@app.get("/settings/image_dataset")
def set_image_dataset():
    d = request.args.get("di", "")
    if isinstance(d, str) and d in list_image_datasets():
        session["image_dataset_name"] = d

        # Clear deck / current card so the new photos take effect immediately
        session.pop("deck", None)
        session.pop("deck_key", None)
        session.pop("current_idx", None)
        session.pop("card_counter", None)

    return ("", 302, {"Location": "/settings"})


@app.get("/settings/taxa")
def settings_taxa():
    _, _, taxa_tree = _load_current_dataset()

    enabled = get_enabled_nodes()
    return render_template(
        "settings_taxa.html", tree=taxa_tree, enabled=enabled
    )


@app.post("/settings/taxa")
def save_settings_taxa():
    session.permanent = True
    _, _, taxa_tree = _load_current_dataset()

    selected = request.form.getlist("enabled")
    selected = [x for x in selected if isinstance(x, str) and x.strip()]

    session["enabled_nodes"] = selected

    session.pop("deck", None)
    session.pop("deck_key", None)

    return render_template(
        "settings_taxa.html",
        tree=taxa_tree,
        enabled=set(selected),
        saved=True,
    )


@app.get("/debug/payload")
def debug_payload():
    _, items, _ = _load_current_dataset()

    if not items:
        dataset = get_selected_dataset()
        return jsonify({"error": f"no items in dataset {dataset}"}), 400

    idx = session.get("current_idx")
    if idx is None or not (0 <= int(idx) < len(items)):
        item = choose_item()
        idx = items.index(item)
        session["current_idx"] = idx
    else:
        item = items[int(idx)]

    payload = make_quiz_payload(item)
    payload["_debug"] = {
        "dataset": get_selected_dataset(),
        "image_dataset": get_selected_image_dataset(),
        "current_idx": int(idx),
        "enabled_ranks": get_enabled_ranks(),
        "enabled_nodes_count": len(get_enabled_nodes()),
        "node_id": getattr(item, "node_id", None),
    }

    return jsonify(payload)


from collections import Counter


@app.get("/debug/stats")
def debug_stats():
    dataset = get_selected_dataset()
    _, items, _ = _load_current_dataset()

    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        limit = 200
    try:
        offset = int(request.args.get("offset", "0"))
    except ValueError:
        offset = 0

    limit = max(1, min(limit, 5000))
    offset = max(0, offset)

    include_pool_cards = request.args.get("pool", "0") == "1"

    def deepest_idx(it: StudyItem) -> int:
        ans = it.answer
        if it.meta.get("_kind") == "specie" and ans.get("genus", "").strip():
            return RANK_INDEX["species"]
        for r in reversed(RANK_KEYS):
            if ans.get(r, "").strip():
                return RANK_INDEX[r]
        return -1

    kind_counts = Counter(it.meta.get("_kind", "unknown") for it in items)
    depth_counts = Counter(deepest_idx(it) for it in items)
    depth_counts_named = {
        (RANK_KEYS[k] if 0 <= k < len(RANK_KEYS) else str(k)): v
        for k, v in depth_counts.items()
    }

    pool = build_pool_indices(items)

    def serialize(it: StudyItem, idx: int) -> Dict[str, Any]:
        return {
            "idx": idx,
            "node_id": it.node_id,
            "kind": it.meta.get("_kind"),
            "answer": it.answer,
            "fin_by_rank": it.meta.get("fin_by_rank", {}),
            "fin": it.meta.get("fin", ""),
            "image": it.meta.get("image", ""),
        }

    page_items = items[offset : offset + limit]
    cards = [serialize(it, offset + i) for i, it in enumerate(page_items)]

    out: Dict[str, Any] = {
        "dataset": dataset,
        "image_dataset": get_selected_image_dataset(),
        "items_total": len(items),
        "pool_total": len(pool),
        "enabled_ranks": get_enabled_ranks(),
        "enabled_nodes_count": len(get_enabled_nodes()),
        "kind_counts": dict(kind_counts),
        "deepest_rank_counts": depth_counts_named,
        "paging": {
            "offset": offset,
            "limit": limit,
            "returned": len(cards),
            "next_offset": (offset + limit)
            if (offset + limit) < len(items)
            else None,
        },
        "cards": cards,
        "pool_indices": pool[:2000],
    }

    if include_pool_cards:
        pool_cards = [
            serialize(items[i], i) for i in pool[:2000] if 0 <= i < len(items)
        ]
        out["pool_cards_first_2000"] = pool_cards

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
