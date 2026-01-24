from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Set, Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

DATA_PATH = os.environ.get("TAXONOMY_JSON", "data.json")


# ---------- helpers to flatten your nested JSON into study items ----------

RANK_KEYS = ["phylum", "class", "order", "family", "genus", "species"]
DEFAULT_ENABLED_RANKS = [
    "phylum",
    "class",
    "family",
    "genus",
    "species",
]  # tweak default

# Your JSON uses "specie" (singular) as an array key.
CHILD_KEYS = ["class", "order", "family", "specie"]
NODE_RANKS = ["phylum", "class", "order", "family"]


@dataclass
class StudyItem:
    answer: Dict[
        str, str
    ]  # phylum/class/order/family/genus/species -> expected latin strings
    meta: Dict[str, Any]  # anything extra (fin, req, image, etc.)
    node_id: str


def safe_lat(node: Dict[str, Any]) -> str:
    v = node.get("lat", "")
    return v.strip() if isinstance(v, str) else ""


def make_node_id(parent_id: str, rank: str, lat: str) -> str:
    part = f"{rank}:{lat}"
    return part if not parent_id else f"{parent_id}>{part}"


def get_disabled_nodes() -> Set[str]:
    raw = session.get("disabled_nodes")
    if isinstance(raw, list):
        return set([x for x in raw if isinstance(x, str)])
    return set()


def is_blocked(node_id: str, disabled: Set[str]) -> bool:
    # blocked if any disabled id is a prefix of this node id
    for d in disabled:
        if node_id == d or node_id.startswith(d + ">"):
            return True
    return False


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def get_enabled_ranks() -> List[str]:
    enabled = session.get("enabled_ranks")
    if isinstance(enabled, list) and all(isinstance(x, str) for x in enabled):
        # keep only known ranks, keep order
        enabled = [r for r in RANK_KEYS if r in enabled]
        if enabled:
            return enabled
    return DEFAULT_ENABLED_RANKS


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return None


def find_nearest_image(path_nodes: List[Dict[str, Any]]) -> Optional[str]:
    """
    If you add `"image": "/static/img/foo.jpg"` to any node,
    this will pick the closest (deepest) one on the current path.
    """
    for node in reversed(path_nodes):
        img = node.get("image")
        if isinstance(img, str) and img.strip():
            return img.strip()
    return None


def extract_items(data: Dict[str, Any]) -> List[StudyItem]:
    items: List[StudyItem] = []

    phyla = data.get("phylum", [])
    if not isinstance(phyla, list):
        return items

    def walk(
        node: Dict[str, Any],
        path: Dict[str, str],
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        # If node contains specie list, create items
        specie_list = node.get("specie")
        if isinstance(specie_list, list):
            for sp in specie_list:
                if not isinstance(sp, dict):
                    continue
                genus = str(sp.get("genus", "") or "")
                species = str(sp.get("species", "") or "")

                ans = dict(path)
                ans["genus"] = genus
                ans["species"] = species

                meta = {
                    "fin": sp.get("fin", ""),
                    "req": sp.get("req", ""),
                    "image": first_nonempty(
                        sp.get("image"),
                        find_nearest_image(path_nodes + [sp]),
                    ),
                }
                items.append(
                    StudyItem(answer=ans, meta=meta, node_id=current_node_id)
                )

        # Recurse into child arrays
        for ck in CHILD_KEYS:
            child_list = node.get(ck)
            if not isinstance(child_list, list):
                continue

            # "specie" handled above; others are group ranks
            if ck == "specie":
                continue

            child_rank = ck  # class/order/family
            for ch in child_list:
                if not isinstance(ch, dict):
                    continue

                new_path = dict(path)
                lat = safe_lat(ch)
                if lat:
                    new_path[child_rank] = lat

                new_node_id = current_node_id
                if lat:
                    new_node_id = make_node_id(current_node_id, child_rank, lat)

                walk(ch, new_path, path_nodes + [ch], new_node_id)

    for ph in phyla:
        if not isinstance(ph, dict):
            continue

        path: Dict[str, str] = {k: "" for k in RANK_KEYS}

        ph_lat = safe_lat(ph)
        if ph_lat:
            path["phylum"] = ph_lat

        ph_id = make_node_id("", "phylum", ph_lat if ph_lat else "UNKNOWN")
        walk(ph, path, [ph], ph_id)

    return items


DATA = load_json(DATA_PATH)
ITEMS = extract_items(DATA)

if not ITEMS:
    print("WARNING: No study items extracted. Check your JSON structure.")


def choose_item() -> StudyItem:
    disabled = get_disabled_nodes()
    pool = [it for it in ITEMS if not is_blocked(it.node_id, disabled)]
    if not pool:
        # fallback: if user disabled everything, ignore filters
        pool = ITEMS
    return random.choice(pool)


def build_taxa_tree(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of tree nodes:
    { "rank": "phylum", "lat": "cnidaria", "id": "phylum:cnidaria", "children": [...] }
    """
    phyla = data.get("phylum", [])
    out: List[Dict[str, Any]] = []
    if not isinstance(phyla, list):
        return out

    def build_children(
        node: Dict[str, Any], parent_id: str, rank: str
    ) -> Dict[str, Any]:
        lat = safe_lat(node) or "UNKNOWN"
        node_id = make_node_id(parent_id, rank, lat)

        tnode = {"rank": rank, "lat": lat, "id": node_id, "children": []}

        for ck in ["class", "order", "family"]:
            child_list = node.get(ck)
            if isinstance(child_list, list):
                for ch in child_list:
                    if isinstance(ch, dict):
                        tnode["children"].append(
                            build_children(ch, node_id, ck)
                        )

        return tnode

    for ph in phyla:
        if isinstance(ph, dict):
            out.append(build_children(ph, "", "phylum"))

    return out


TAXA_TREE = build_taxa_tree(DATA)

# ---------- routes ----------


@app.get("/")
def index():
    item = choose_item()
    idx = ITEMS.index(item)
    session["current_idx"] = idx

    enabled_ranks = get_enabled_ranks()
    return render_template("index.html", item=item, ranks=enabled_ranks)


@app.post("/new")
def new_item():
    item = choose_item()
    idx = ITEMS.index(item)
    session["current_idx"] = idx
    return jsonify(
        {
            "answer": item.answer,
            "meta": item.meta,
        }
    )


@app.post("/check")
def check():
    """
    Client can send current inputs and we return per-field correctness.
    """
    idx = session.get("current_idx")
    if idx is None or not (0 <= int(idx) < len(ITEMS)):
        return jsonify({"error": "no active item"}), 400

    item = ITEMS[int(idx)]
    payload = request.get_json(force=True) or {}
    inputs = payload.get("inputs", {})

    enabled_ranks = get_enabled_ranks()
    result = {}

    for rank in enabled_ranks:
        expected = norm(item.answer.get(rank, ""))
        got = norm(str(inputs.get(rank, "")))

        if expected == "":
            result[rank] = {
                "correct": None,
                "expected": item.answer.get(rank, ""),
            }
        else:
            result[rank] = {
                "correct": (got == expected),
                "expected": item.answer.get(rank, ""),
            }

    return jsonify(result)


@app.get("/settings")
def settings():
    enabled = set(get_enabled_ranks())
    return render_template(
        "settings.html", all_ranks=RANK_KEYS, enabled=enabled
    )


@app.post("/settings")
def save_settings():
    # ranks come from checkbox list named "ranks"
    selected = request.form.getlist("ranks")
    selected = [r for r in RANK_KEYS if r in selected]  # sanitize + keep order

    # Avoid "nothing selected" footgun: if empty, keep previous or default
    if not selected:
        selected = DEFAULT_ENABLED_RANKS

    session["enabled_ranks"] = selected
    return render_template(
        "settings.html", all_ranks=RANK_KEYS, enabled=set(selected), saved=True
    )


@app.get("/settings/taxa")
def settings_taxa():
    disabled = get_disabled_nodes()
    return render_template(
        "settings_taxa.html", tree=TAXA_TREE, disabled=disabled
    )


@app.post("/settings/taxa")
def save_settings_taxa():
    # checkboxes named "disabled"
    selected = request.form.getlist("disabled")
    selected = [x for x in selected if isinstance(x, str) and x.strip()]
    session["disabled_nodes"] = selected
    return render_template(
        "settings_taxa.html", tree=TAXA_TREE, disabled=set(selected), saved=True
    )


if __name__ == "__main__":
    # debug=True is fine for localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
