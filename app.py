from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import timedelta
from collections import Counter

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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DEFAULT_DATASET = "selkarangattomat"

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


# ---------- transparent container traversal (Option C) ----------
def looks_taxonomic_container(node: Dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False

    # direct known children?
    for k in REAL_CHILD_KEYS:
        v = node.get(k)
        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
            return True

    # nested unknown list-of-dicts that might themselves be containers
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
    files = sorted(DATA_DIR.glob("*.json"))
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


def get_enabled_ranks() -> List[str]:
    enabled = session.get("enabled_ranks")

    # Normalize to a list of strings
    if not isinstance(enabled, list) or not all(
        isinstance(x, str) for x in enabled
    ):
        enabled = list(DEFAULT_ENABLED_RANKS)
    else:
        enabled = [
            r for r in RANK_KEYS if r in enabled
        ]  # canonical order + known ranks
        if not enabled:
            enabled = list(DEFAULT_ENABLED_RANKS)

    # Force genus/species to be coupled
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
def load_dataset_cached(stem: str):
    path = DATA_DIR / f"{stem}.json"
    data = load_json(str(path))
    items = extract_items(data)
    taxa_tree = build_taxa_tree(data)
    return data, items, taxa_tree


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
    _, _, taxa_tree = load_dataset_cached(dataset)
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

    # default: everything enabled
    dataset = get_selected_dataset()
    return get_all_node_ids(dataset)


# ---------- item extraction ----------
def extract_items(data: Dict[str, Any]) -> List[StudyItem]:
    items: List[StudyItem] = []

    phyla = data.get("phylum", [])
    if not isinstance(phyla, list):
        return items

    def node_has_children(node: Dict[str, Any]) -> bool:
        # group children through transparent containers
        if collect_real_children_through_containers(node, "class"):
            return True
        if collect_real_children_through_containers(node, "order"):
            return True
        if collect_real_children_through_containers(node, "family"):
            return True

        # species lists (also through containers)
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

        meta = {
            "_kind": "node",
            "has_children": node_has_children(node),  # ✅ ADD THIS
            "fin": fin_or_lat(
                node.get("fin"),
                path.get("family")
                or path.get("order")
                or path.get("class")
                or path.get("phylum"),
            ),
            "fin_by_rank": dict(path_fin),
            "image": first_nonempty(
                node.get("image"),
                find_nearest_image(path_nodes + [node]),
            ),
        }

        items.append(StudyItem(answer=ans, meta=meta, node_id=current_node_id))

    def walk(
        node: Dict[str, Any],
        rank: str,
        path: Dict[str, str],
        path_fin: Dict[str, str],
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        # ✅ 0) add an item for this rank node (phylum/class/order/family)
        # (this is the key fix so "family" is selectable even if species exist below)
        if rank in ("phylum", "class", "order", "family"):
            make_node_item(node, path, path_fin, path_nodes, current_node_id)

        # 1) species items (specie may exist through any number of transparent containers)
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
                # Finnish for the organism itself (best effort)
                sp_fin = str(sp.get("fin") or "").strip()
                if sp_fin:
                    fin_by_rank["species"] = sp_fin
                meta = {
                    "_kind": "specie",
                    "fin": fin_or_lat(
                        sp.get("fin"), f"{genus} {species}".strip()
                    ),
                    "fin_by_rank": fin_by_rank,  # ✅ add this
                    "req": sp.get("req", ""),
                    "image": first_nonempty(
                        sp.get("image"),
                        find_nearest_image(path_nodes + [sp]),
                    ),
                }

                items.append(
                    StudyItem(answer=ans, meta=meta, node_id=current_node_id)
                )

        # 2) recurse into real rank children (through transparent containers)
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


# ---------- quiz logic ----------
def make_quiz_payload(item: StudyItem) -> Dict[str, Any]:
    enabled_ranks = get_enabled_ranks()

    # ranks that are ticked AND exist on this item
    present = [r for r in enabled_ranks if item.answer.get(r, "").strip()]

    payload = {
        "answer": item.answer,
        "meta": item.meta,
        "mode": "taxonomy",
        "given": None,
        "quiz_ranks": present if present else enabled_ranks,
    }

    # if we can't do a "given" + "asked", just ask what exists
    if len(present) < 2:
        return payload

    # ✅ highest (most general) selected rank is allowed to be given
    given_rank = min(present, key=lambda r: RANK_INDEX[r])

    # quiz_ranks = [
    #     r
    #     for r in enabled_ranks
    #     if RANK_INDEX[r] > RANK_INDEX[given_rank]
    #     and item.answer.get(r, "").strip()
    # ]

    # if not quiz_ranks:
    #     return payload

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
        if it.meta.get("_kind") == "specie" and ans.get("genus", "").strip():
            return RANK_INDEX["species"]
        for r in reversed(RANK_KEYS):
            if ans.get(r, "").strip():
                return RANK_INDEX[r]
        return -1

    def current_node_rank_idx(it: StudyItem) -> int:
        # for node-items, the deepest filled rank is the node's own rank
        return deepest_filled_rank_idx(it)

    pool = []
    for i, it in enumerate(items):
        d = deepest_filled_rank_idx(it)

        # keep only items that are not deeper than user wants
        if d > max_enabled_idx:
            continue

        # ✅ drop "higher node" cards if they have children AND user trains deeper ranks
        if it.meta.get("_kind") == "node" and it.meta.get("has_children"):
            node_rank = current_node_rank_idx(it)
            if max_enabled_idx > node_rank:
                continue

        pool.append(i)

    if not pool:
        pool = list(range(len(items)))

    enabled_nodes = get_enabled_nodes()
    if enabled_nodes:
        leaves = [
            i for i in pool if is_allowed(items[i].node_id, enabled_nodes)
        ]
    return leaves


def get_deck_key(dataset: str) -> str:
    # include settings in the key so deck refreshes when settings change
    ranks_key = ",".join(get_enabled_ranks())
    # enabled_nodes can be large; but we need it in the key to avoid stale decks
    nodes = sorted(get_enabled_nodes())
    nodes_key = str(hash(tuple(nodes)))
    return f"{dataset}|{ranks_key}|{nodes_key}"


def draw_item_from_deck(items: List[StudyItem]) -> StudyItem:
    dataset = get_selected_dataset()
    deck_key = get_deck_key(dataset)

    # if settings changed since last time, rebuild
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

    # draw next (no repeats until empty)
    next_idx = int(deck.pop())
    session["deck"] = deck
    return items[next_idx]


def choose_item() -> StudyItem:
    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)
    if not items:
        raise RuntimeError(f"No study items extracted from {dataset}.json")
    return draw_item_from_deck(items)


# ---------- routes ----------
@app.get("/")
def index():
    # Refresh = restart run (new deck + counter)
    session.pop("deck", None)
    session.pop("deck_key", None)
    session.pop("current_idx", None)
    session["card_counter"] = 0

    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)
    total = get_pool_total(items)

    item = choose_item()  # will rebuild + shuffle deck because we popped it
    idx = items.index(item)
    session["current_idx"] = idx

    counter = bump_card_counter()  # becomes 1

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
    )


@app.get("/manual")
def manual():
    return render_template("manual.html")


@app.post("/new")
def new_item():
    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)

    item = choose_item()
    session["current_idx"] = items.index(item)

    counter = bump_card_counter()
    total = get_pool_total(items)

    payload = make_quiz_payload(item)
    payload["counter"] = counter
    payload["total"] = total
    payload["hint_language"] = get_hint_language()
    return jsonify(payload)


@app.post("/check")
def check():
    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)

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

    return render_template(
        "settings.html",
        all_ranks=RANK_KEYS,
        enabled=enabled,
        datasets=datasets,
        selected_dataset=selected,
        rank_fi=RANK_FI,
        actions_position=get_actions_position(),
        hint_language=get_hint_language(),
    )


@app.post("/settings")
def save_settings():
    session.permanent = True
    selected_ranks = request.form.getlist("ranks")
    selected_ranks = [r for r in RANK_KEYS if r in selected_ranks]

    # genus+species combined toggle
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

    # If nothing selected, fallback
    if not selected_ranks:
        selected_ranks = DEFAULT_ENABLED_RANKS

    # keep canonical order
    selected_ranks = [r for r in RANK_KEYS if r in selected_ranks]
    session["enabled_ranks"] = selected_ranks

    datasets = list_datasets()
    selected = get_selected_dataset()

    pos = request.form.get("actions_position", "top")
    session["actions_position"] = pos if pos in ("top", "bottom") else "top"

    session.pop("deck", None)
    session.pop("deck_key", None)
    hint_lang = request.form.get("hint_language", "fin")
    session["hint_language"] = (
        hint_lang if hint_lang in ("fin", "lat") else "fin"
    )

    return render_template(
        "settings.html",
        all_ranks=RANK_KEYS,
        enabled=set(selected_ranks),
        datasets=datasets,
        selected_dataset=selected,
        rank_fi=RANK_FI,
        actions_position=session.get("actions_position", "top"),
        hint_language=get_hint_language(),
        saved=True,
    )


@app.get("/settings/dataset")
def set_dataset():
    d = request.args.get("d", "")
    if isinstance(d, str) and d in list_datasets():
        session["dataset_name"] = d
        session.pop("enabled_nodes", None)  # reset taxa filters per dataset
        session.pop("current_idx", None)

        session.pop("deck", None)
        session.pop("deck_key", None)

        session.pop("card_counter", None)
    return ("", 302, {"Location": "/settings"})


@app.get("/settings/taxa")
def settings_taxa():
    dataset = get_selected_dataset()
    _, _, taxa_tree = load_dataset_cached(dataset)

    enabled = get_enabled_nodes()
    return render_template(
        "settings_taxa.html", tree=taxa_tree, enabled=enabled
    )


@app.post("/settings/taxa")
def save_settings_taxa():
    session.permanent = True
    dataset = get_selected_dataset()
    _, _, taxa_tree = load_dataset_cached(dataset)

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
    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)

    if not items:
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
        "dataset": dataset,
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
    _, items, _ = load_dataset_cached(dataset)

    # paging
    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        limit = 200
    try:
        offset = int(request.args.get("offset", "0"))
    except ValueError:
        offset = 0

    limit = max(1, min(limit, 5000))  # safety cap
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

    # serialize ALL items (but paginate in response)
    def serialize(it: StudyItem, idx: int) -> Dict[str, Any]:
        return {
            "idx": idx,
            "node_id": it.node_id,
            "kind": it.meta.get("_kind"),
            "answer": it.answer,  # latin per rank
            "fin_by_rank": it.meta.get("fin_by_rank", {}),
            "fin": it.meta.get("fin", ""),
            "image": it.meta.get("image", ""),
        }

    page_items = items[offset : offset + limit]
    cards = [serialize(it, offset + i) for i, it in enumerate(page_items)]

    out: Dict[str, Any] = {
        "dataset": dataset,
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
        "cards": cards,  # ✅ this is the card list (paged)
        "pool_indices": pool[:2000],  # keep pool list from becoming enormous
    }

    if include_pool_cards:
        # pool cards can also be huge, so return up to first 2000
        pool_cards = [
            serialize(items[i], i) for i in pool[:2000] if 0 <= i < len(items)
        ]
        out["pool_cards_first_2000"] = pool_cards

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
