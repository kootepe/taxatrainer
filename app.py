from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Set, Any, Dict, List, Optional, Tuple
from functools import lru_cache
from pathlib import Path


from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

DATA_PATH = os.environ.get("TAXONOMY_JSON", "data.json")


# ---------- helpers to flatten your nested JSON into study items ----------
RANK_FI = {
    "phylum": "pääjakso",
    "class": "luokka",
    "order": "lahko",
    "family": "heimo",
    "genus": "suku",
    "species": "laji",
}

RANK_KEYS = ["phylum", "class", "order", "family", "genus", "species"]

DEFAULT_ENABLED_RANKS = [
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]  # tweak default

# Your JSON uses "specie" (singular) as an array key.
CHILD_KEYS = ["class", "order", "family", "specie"]
NODE_RANKS = ["phylum", "class", "order", "family"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR  # keep json files here
DEFAULT_DATASET = "selkarangattomat"  # means data.json


DEFAULT_QUIZ_MODE = "taxonomy"  # "taxonomy" or "higher_from_subtaxon"
DEFAULT_PROMPT_RANKS = ["family"]  # allowed given ranks in higher_from_subtaxon

RANK_INDEX = {r: i for i, r in enumerate(RANK_KEYS)}  # phylum=0 .. species=5


@dataclass
class PromptItem:
    rank: str  # "class" | "order" | "family"
    value: str  # latin at that rank
    answer: Dict[
        str, str
    ]  # full path dict (phylum/class/order/family/genus/species), but only higher ranks are used
    meta: Dict[str, Any]
    node_id: str  # node id at this rank (so disabling works)


@dataclass
class StudyItem:
    answer: Dict[
        str, str
    ]  # phylum/class/order/family/genus/species -> expected latin strings
    meta: Dict[str, Any]  # anything extra (fin, req, image, etc.)
    node_id: str


def get_quiz_mode() -> str:
    mode = session.get("quiz_mode")
    return (
        mode
        if mode in ("taxonomy", "higher_from_subtaxon")
        else DEFAULT_QUIZ_MODE
    )


def get_prompt_ranks() -> List[str]:
    raw = session.get("prompt_ranks")
    if isinstance(raw, list):
        raw = [r for r in raw if r in ("class", "order", "family")]
        if raw:
            return raw
    return DEFAULT_PROMPT_RANKS


def list_datasets() -> list[str]:
    # returns ["data", "animals", "plants"] from files like data.json, animals.json
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


@lru_cache(maxsize=16)
def load_dataset_cached(stem: str):
    path = DATA_DIR / f"{stem}.json"
    data = load_json(str(path))
    items = extract_items(data)
    prompt_items = extract_prompt_items(data)
    taxa_tree = build_taxa_tree(data)
    return data, items, prompt_items, taxa_tree


def safe_lat(node: Dict[str, Any]) -> str:
    v = node.get("lat", "")
    return v.strip() if isinstance(v, str) else ""


def make_node_id(parent_id: str, rank: str, lat: str) -> str:
    part = f"{rank}:{lat}"
    return part if not parent_id else f"{parent_id}>{part}"


def get_enabled_nodes() -> Set[str]:
    raw = session.get("enabled_nodes")
    if isinstance(raw, list):
        return set([x for x in raw if isinstance(x, str)])
    return set()


def is_allowed(node_id: str, enabled: Set[str]) -> bool:
    # allowed if node_id is inside ANY enabled prefix
    for e in enabled:
        if node_id == e or node_id.startswith(e + ">"):
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


def extract_prompt_items(data: Dict[str, Any]) -> List[PromptItem]:
    out: List[PromptItem] = []

    phyla = data.get("phylum", [])
    if not isinstance(phyla, list):
        return out

    def walk(
        node: Dict[str, Any],
        path: Dict[str, str],
        path_nodes: List[Dict[str, Any]],
        current_node_id: str,
    ):
        # Recurse into child arrays (class/order/family)
        for ck in ["class", "order", "family"]:
            child_list = node.get(ck)
            if not isinstance(child_list, list):
                continue

            for ch in child_list:
                if not isinstance(ch, dict):
                    continue

                lat = safe_lat(ch)
                if not lat:
                    continue

                child_rank = ck
                new_path = dict(path)
                new_path[child_rank] = lat

                new_node_id = make_node_id(current_node_id, child_rank, lat)

                # Each node becomes ONE quiz candidate
                meta = {
                    "fin": ch.get("fin", "")
                    or "",  # optional, if you add fin on nodes
                    "req": ch.get("req", "") or "",
                    "image": first_nonempty(
                        ch.get("image"),
                        find_nearest_image(path_nodes + [ch]),
                    ),
                }

                # Ensure all ranks exist in dict (frontend/check code expects keys)
                full_answer = {k: "" for k in RANK_KEYS}
                full_answer.update(new_path)

                out.append(
                    PromptItem(
                        rank=child_rank,
                        value=lat,
                        answer=full_answer,
                        meta=meta,
                        node_id=new_node_id,
                    )
                )

                walk(ch, new_path, path_nodes + [ch], new_node_id)

    for ph in phyla:
        if not isinstance(ph, dict):
            continue

        ph_lat = safe_lat(ph) or ""
        ph_id = make_node_id("", "phylum", ph_lat if ph_lat else "UNKNOWN")

        path = {k: "" for k in RANK_KEYS}
        if ph_lat:
            path["phylum"] = ph_lat

        walk(ph, path, [ph], ph_id)

    return out


def choose_species_item() -> StudyItem:
    dataset = get_selected_dataset()
    _, items, _, _ = load_dataset_cached(dataset)

    if not items:
        raise RuntimeError(f"No study items extracted from {dataset}.json")

    enabled = get_enabled_nodes()
    pool = [it for it in items if not is_allowed(it.node_id, enabled)]
    if not pool:
        pool = items
    return random.choice(pool)


def choose_prompt_item() -> PromptItem:
    dataset = get_selected_dataset()
    _, _, prompt_items, _ = load_dataset_cached(dataset)

    if not prompt_items:
        raise RuntimeError(f"No prompt items extracted from {dataset}.json")

    allowed_prompt_ranks = set(
        get_prompt_ranks()
    )  # e.g. {"family"} or {"order","family"}
    enabled_ranks = get_enabled_ranks()

    def has_something_to_ask(p: PromptItem) -> bool:
        # ranks ABOVE prompt rank, that are enabled and present in the answer
        quiz_ranks = [
            r
            for r in enabled_ranks
            if RANK_INDEX[r] < RANK_INDEX[p.rank]
            and p.answer.get(r, "").strip()
        ]
        return len(quiz_ranks) > 0

    enabled = get_enabled_nodes()

    # 1) only allowed prompt ranks
    candidates = [
        p
        for p in prompt_items
        if p.rank in allowed_prompt_ranks and has_something_to_ask(p)
    ]

    # If nothing matches (edge case: settings make it impossible), fall back to any usable prompt item
    if not candidates:
        candidates = [p for p in prompt_items if has_something_to_ask(p)]
    if not candidates:
        candidates = prompt_items  # absolute fallback (should be rare)

    # 2) apply your taxa filter logic (keeping your current behavior)
    pool = [p for p in candidates if not is_allowed(p.node_id, enabled)]
    if not pool:
        pool = candidates

    return random.choice(pool)


# def choose_item():
#     dataset = get_selected_dataset()
#     _, items, _, _, _ = load_dataset_cached(dataset)
#
#     if not items:
#         raise RuntimeError(f"No study items extracted from {dataset}.json")
#
#     disabled = get_disabled_nodes()
#     pool = [it for it in items if not is_blocked(it.node_id, disabled)]
#     if not pool:
#         pool = items
#     return random.choice(pool)


# def choose_item() -> StudyItem:
#     disabled = get_disabled_nodes()
#     pool = [it for it in ITEMS if not is_blocked(it.node_id, disabled)]
#     if not pool:
#         # fallback: if user disabled everything, ignore filters
#         pool = ITEMS
#     return random.choice(pool)


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


def make_quiz_payload(item: StudyItem) -> Dict[str, Any]:
    enabled_ranks = get_enabled_ranks()
    mode = get_quiz_mode()

    payload = {
        "answer": item.answer,
        "meta": item.meta,
        "mode": mode,
        "given": None,  # e.g. {"rank":"family","value":"spongillidae"}
        "quiz_ranks": enabled_ranks,  # ranks the UI should render inputs for
    }

    if mode != "higher_from_subtaxon":
        return payload

    # pick a prompt rank that exists on this item
    prompt_candidates = [
        r for r in get_prompt_ranks() if item.answer.get(r, "").strip()
    ]
    if not prompt_candidates:
        return payload  # fallback to normal if nothing usable

    prompt_rank = random.choice(prompt_candidates)

    # ranks ABOVE prompt_rank (higher taxa)
    quiz_ranks = [
        r
        for r in enabled_ranks
        if RANK_INDEX[r] < RANK_INDEX[prompt_rank]
        and item.answer.get(r, "").strip()
    ]

    # if nothing to ask, fallback to normal
    if not quiz_ranks:
        return payload

    payload["given"] = {"rank": prompt_rank, "value": item.answer[prompt_rank]}
    payload["quiz_ranks"] = quiz_ranks
    return payload


def make_prompt_quiz_payload(p: PromptItem) -> Dict[str, Any]:
    enabled_ranks = get_enabled_ranks()

    prompt_rank = p.rank
    # only allow if user enabled it as a prompt rank in settings
    allowed = get_prompt_ranks()
    if prompt_rank not in allowed:
        # fallback: pretend normal taxonomy
        return {
            "answer": p.answer,
            "meta": p.meta,
            "mode": "taxonomy",
            "given": None,
            "quiz_ranks": enabled_ranks,
        }

    quiz_ranks = [
        r
        for r in enabled_ranks
        if RANK_INDEX[r] < RANK_INDEX[prompt_rank]
        and p.answer.get(r, "").strip()
    ]
    if not quiz_ranks:
        # nothing higher to ask -> fallback to taxonomy-like
        return {
            "answer": p.answer,
            "meta": p.meta,
            "mode": "taxonomy",
            "given": None,
            "quiz_ranks": enabled_ranks,
        }

    return {
        "answer": p.answer,
        "meta": p.meta,
        "mode": "higher_from_subtaxon",
        "given": {"rank": prompt_rank, "value": p.value},
        "quiz_ranks": quiz_ranks,
    }


# ---------- routes ----------


@app.get("/")
def index():
    mode = get_quiz_mode()

    if mode == "higher_from_subtaxon":
        p = choose_prompt_item()
        session["current_kind"] = "prompt"
        session["current_node_id"] = p.node_id
        payload = make_prompt_quiz_payload(p)

        return render_template(
            "index.html",
            item=p,  # p has .answer and .meta just like StudyItem
            ranks=payload["quiz_ranks"],
            rank_fi=RANK_FI,
            quiz_mode=payload["mode"],
            given=payload["given"],
        )

    # taxonomy mode (old behavior)
    dataset = get_selected_dataset()
    _, items, _, _ = load_dataset_cached(dataset)

    item = choose_species_item()
    idx = items.index(item)
    session["current_kind"] = "species"
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


@app.post("/new")
def new_item():
    mode = get_quiz_mode()

    if mode == "higher_from_subtaxon":
        p = choose_prompt_item()
        session["current_kind"] = "prompt"
        session["current_node_id"] = p.node_id
        return jsonify(make_prompt_quiz_payload(p))

    dataset = get_selected_dataset()
    _, items, _, _ = load_dataset_cached(dataset)

    item = choose_species_item()
    idx = items.index(item)
    session["current_kind"] = "species"
    session["current_idx"] = idx
    return jsonify(make_quiz_payload(item))


@app.post("/check")
def check():
    dataset = get_selected_dataset()
    _, items, prompt_items, _ = load_dataset_cached(dataset)

    kind = session.get("current_kind")

    if kind == "prompt":
        node_id = session.get("current_node_id")
        if not isinstance(node_id, str):
            return jsonify({"error": "no active item"}), 400

        item = next((p for p in prompt_items if p.node_id == node_id), None)
        if item is None:
            return jsonify({"error": "no active item"}), 400

        answer_dict = item.answer

    else:
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
        quiz_mode=get_quiz_mode(),
        prompt_ranks=get_prompt_ranks(),
    )


@app.post("/settings")
def save_settings():
    # rank toggles (existing)
    selected_ranks = request.form.getlist("ranks")
    selected_ranks = [r for r in RANK_KEYS if r in selected_ranks]
    if not selected_ranks:
        selected_ranks = DEFAULT_ENABLED_RANKS
    session["enabled_ranks"] = selected_ranks

    # NEW: quiz mode
    mode = request.form.get("quiz_mode", DEFAULT_QUIZ_MODE)
    if mode not in ("taxonomy", "higher_from_subtaxon"):
        mode = DEFAULT_QUIZ_MODE
    session["quiz_mode"] = mode

    # NEW: allowed prompt ranks (class/order/family)
    pr = request.form.getlist("prompt_ranks")
    pr = [r for r in pr if r in ("class", "order", "family")]
    if not pr:
        pr = DEFAULT_PROMPT_RANKS
    session["prompt_ranks"] = pr

    datasets = list_datasets()
    selected = get_selected_dataset()

    return render_template(
        "settings.html",
        all_ranks=RANK_KEYS,
        enabled=set(selected_ranks),
        datasets=datasets,
        selected_dataset=selected,
        rank_fi=RANK_FI,
        quiz_mode=mode,
        prompt_ranks=pr,
        saved=True,
    )


@app.get("/settings/dataset")
def set_dataset():
    d = request.args.get("d", "")
    if isinstance(d, str) and d in list_datasets():
        session["dataset_name"] = d
        session.pop(
            "disabled_nodes", None
        )  # optional: reset taxa filters per dataset
        session.pop("current_idx", None)  # reset current card
    return ("", 302, {"Location": "/settings"})


@app.get("/settings/taxa")
def settings_taxa():
    dataset = get_selected_dataset()
    _, _, _, taxa_tree = load_dataset_cached(dataset)

    enabled = get_enabled_nodes()
    return render_template(
        "settings_taxa.html", tree=taxa_tree, enabled=enabled
    )


@app.post("/settings/taxa")
def save_settings_taxa():
    dataset = get_selected_dataset()
    _, _, _, taxa_tree = load_dataset_cached(dataset)

    selected = request.form.getlist("enabled")
    selected = [x for x in selected if isinstance(x, str) and x.strip()]
    session["enabled_nodes"] = selected

    return render_template(
        "settings_taxa.html", tree=taxa_tree, enabled=set(selected), saved=True
    )


if __name__ == "__main__":
    # debug=True is fine for localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
