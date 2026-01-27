from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import timedelta

from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


app.config.update(
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=365),
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True,  # set True if you serve over HTTPS
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
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        ans = {k: "" for k in RANK_KEYS}
        ans.update(path)

        meta = {
            "_kind": "node",
            "fin": fin_or_lat(
                node.get("fin"),
                path.get("family")
                or path.get("order")
                or path.get("class")
                or path.get("phylum"),
            ),
            "req": (node.get("req") or ""),
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
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        # ✅ 0) add an item for this rank node (phylum/class/order/family)
        # (this is the key fix so "family" is selectable even if species exist below)
        if rank in ("phylum", "class", "order", "family"):
            make_node_item(node, path, path_nodes, current_node_id)

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

                meta = {
                    "_kind": "specie",
                    "fin": fin_or_lat(
                        sp.get("fin"), f"{genus} {species}".strip()
                    ),
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

                new_node_id = make_node_id(current_node_id, ck, lat)
                walk(ch, ck, new_path, path_nodes + [ch], new_node_id)

        # 3) optional: leaf/end-node item (kept, but now less important)
        if not node_has_children(node):
            make_node_item(node, path, path_nodes, current_node_id)

    for ph in phyla:
        if not isinstance(ph, dict):
            continue

        path: Dict[str, str] = {k: "" for k in RANK_KEYS}

        ph_lat = safe_lat(ph)
        if ph_lat:
            path["phylum"] = ph_lat

        ph_id = make_node_id("", "phylum", ph_lat if ph_lat else "UNKNOWN")
        walk(ph, "phylum", path, [ph], ph_id)

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


def choose_item() -> StudyItem:
    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)

    if not items:
        raise RuntimeError(f"No study items extracted from {dataset}.json")

    enabled_ranks = get_enabled_ranks()
    max_enabled_idx = max(RANK_INDEX[r] for r in enabled_ranks)

    def deepest_filled_rank_idx(it: StudyItem) -> int:
        ans = it.answer

        # If this item came from the "specie" list, treat it as a terminal leaf at species-level
        # even if species is empty (genus-only records like "Arion sp.")
        if it.meta.get("_kind") == "specie" and ans.get("genus", "").strip():
            return RANK_INDEX["species"]

        for r in reversed(RANK_KEYS):
            if ans.get(r, "").strip():
                return RANK_INDEX[r]
        return -1

    # ✅ LEAVES relative to settings:
    # only choose items whose deepest rank is EXACTLY the deepest enabled rank
    leaves = [
        it for it in items if deepest_filled_rank_idx(it) == max_enabled_idx
    ]

    # Fallbacks (in case dataset doesn't have that level at all)
    if not leaves:
        leaves = [
            it for it in items if deepest_filled_rank_idx(it) <= max_enabled_idx
        ]
    if not leaves:
        leaves = items

    enabled_nodes = get_enabled_nodes()
    pool = (
        [it for it in leaves if is_allowed(it.node_id, enabled_nodes)]
        if enabled_nodes
        else leaves
    )
    if not pool:
        pool = leaves

    return random.choice(pool)


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# ---------- routes ----------
@app.get("/")
def index():
    dataset = get_selected_dataset()
    _, items, _ = load_dataset_cached(dataset)

    item = choose_item()
    idx = items.index(item)
    session["current_idx"] = idx

    payload = make_quiz_payload(item)
    return render_template(
        "index.html",
        item=item,
        ranks=payload["quiz_ranks"],
        rank_fi=RANK_FI,
        quiz_mode=payload["mode"],
        given=payload["given"],
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
    return jsonify(make_quiz_payload(item))


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

    return render_template(
        "settings.html",
        all_ranks=RANK_KEYS,
        enabled=set(selected_ranks),
        datasets=datasets,
        selected_dataset=selected,
        rank_fi=RANK_FI,
        saved=True,
    )


@app.get("/settings/dataset")
def set_dataset():
    d = request.args.get("d", "")
    if isinstance(d, str) and d in list_datasets():
        session["dataset_name"] = d
        session.pop("enabled_nodes", None)  # reset taxa filters per dataset
        session.pop("current_idx", None)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
