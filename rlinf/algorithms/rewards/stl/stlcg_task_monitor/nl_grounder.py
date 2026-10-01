"""Deterministic NL -> BDDL-atom grounder for LIBERO.

Closed-set grounder: every output name is SELECTED from the scene
(``object_pos`` keys ∪ ``site_names``); nothing is invented. Given a natural
language task instruction and the current privileged MuJoCo state, it produces
the goal atoms ``(pred, obj, target)`` and the parallel ``grasp_objs`` list that
:class:`OnlineRobustnessTracker` consumes.

Design (see repo notes):
  * actions are segmented from the NL by verb keyword (no dependency on a lossy
    external parse);
  * OBJECT instance  : object_pos category match -> color filter -> geometric
                       disambiguation (between / next_to / left-right / front-back
                       / middle=median / from-center) -> lexicographic tiebreak;
  * TARGET           : pred + parent instance + NL type word -> the right
                       ``{parent}_{type}_region`` from site_names, or a table
                       spatial region, or a body instance.

Public entry point: :func:`ground_task_state`.
"""

import re
from typing import List, Optional, Tuple

import numpy as np

COLORS = {"black", "white", "red", "green", "blue", "yellow",
          "brown", "orange", "purple", "pink", "gray", "grey"}

CATEGORY_ALIAS = {
    "cabinet": ["cabinet", "wooden_cabinet", "white_cabinet",
                "wooden_two_layer_shelf", "shelf"],
    "drawer": ["cabinet", "wooden_cabinet", "white_cabinet", "shelf"],
    "shelf": ["wooden_two_layer_shelf", "shelf", "layer_shelf"],
    "stove": ["stove", "flat_stove"],
    "basket": ["basket", "bin"],
    "tray": ["tray", "wooden_tray"],
    "microwave": ["microwave"],
    "caddy": ["caddy", "desk_caddy"],
    "rack": ["wine_rack", "rack"],
    "frying_pan": ["frypan", "frying_pan", "pan"],
    "mug": ["mug"],
    "bowl": ["bowl"],
    "plate": ["plate"],
    "book": ["book"],
    "ramekin": ["ramekin"],
    "moka_pot": ["moka_pot"],
    "wine_bottle": ["wine_bottle"],
}

# putting these "in" really means "on" (they are surfaces)
SURFACE_ON = {"bowl", "plate", "mug", "dish", "pan", "ramekin", "frying_pan"}
# fixture categories -> target is a REGION, not a manipulable object
FIXTURE_CATEGORIES = {"cabinet", "drawer", "shelf", "stove", "basket", "tray",
                      "bin", "box", "microwave", "caddy", "rack", "fridge"}

VERB_RE = [
    ("turn_off", r"\bturn\s+off\b"),
    ("turn_on", r"\bturn\s+on\b"),
    ("pick", r"\bpick\s+up\b|\bpick\b"),
    ("put", r"\bput\b"),
    ("place", r"\bplace\b"),
    ("push", r"\bpush\b"),
    ("stack", r"\bstack\b"),
    ("open", r"\bopen\b"),
    ("close", r"\bclose\b"),
]

# prepositions that introduce the TARGET (object phrase ends before these)
_TARGET_PREP = re.compile(r"\b(on\s+top\s+of|on|in|into|under|to\s+the|inside)\b")

# instance-name tokens that carry an implicit color not present in the name
NAME_IMPLICIT_COLOR = {"porcelain": "white"}


# ---------------------------------------------------------------------------
# vocab / category matching
# ---------------------------------------------------------------------------

def _stem_re(stem: str) -> str:
    parts = [re.escape(p) for p in stem.split("_")]
    # tolerate plurals on the last token (moka pot(s), bowl(s)); strip a trailing s
    # first so an already-plural stem like "cookies" still matches singular "cookie"
    parts[-1] = re.sub(r"s$", r"", parts[-1]) + r"s?"
    return r"[\s_]".join(parts)


def name_colors(name: str) -> set:
    low = name.lower().replace("_", " ")  # underscore is NOT a \b boundary
    cols = {c for c in COLORS if re.search(r"\b" + c + r"\b", low)}
    for tok, col in NAME_IMPLICIT_COLOR.items():
        if tok in low:
            cols.add(col)
    return cols


def build_vocab(priv: dict) -> list:
    stems = set()
    for k in priv.get("object_pos", {}):
        if "_to_robot0_eef" in k:
            continue
        s = re.sub(r"_\d+$", "", k)
        for pre in ("akita_", "wooden_", "flat_", "glazed_rim_porcelain_", "chefmate_8_",
                    "porcelain_", "red_", "yellow_", "white_", "new_"):
            if s.startswith(pre):
                s = s[len(pre):]
        stems.add(s)
    return sorted(stems, key=len, reverse=True)


def _all_category_matches(phrase: str, vocab: list):
    found = []
    for canon, stems in CATEGORY_ALIAS.items():
        for st in stems:
            for m in re.finditer(r"\b" + _stem_re(st) + r"\b", phrase):
                found.append((m.start(), canon))
    for st in vocab:
        if len(st) > 2:
            for m in re.finditer(r"\b" + _stem_re(st) + r"\b", phrase):
                found.append((m.start(), st))
    return found


def _first_category(phrase: str, vocab: list) -> Optional[str]:
    matches = _all_category_matches(phrase, vocab)
    return min(matches)[1] if matches else None


def _category_in_phrase(phrase: str, vocab: list) -> Optional[str]:
    matches = _all_category_matches(phrase, vocab)
    canon_matches = [(p, c) for p, c in matches if c in CATEGORY_ALIAS]
    if canon_matches:
        return min(canon_matches)[1]
    return min(matches)[1] if matches else None


def _strip_instance(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def category_candidates(canonical, object_pos: dict) -> list:
    if not canonical:
        return []
    stems = CATEGORY_ALIAS.get(canonical, [canonical])
    out = []
    for k in object_pos:
        if "_to_robot0_eef" in k:
            continue
        sk = _strip_instance(k)
        for st in stems:
            if st == sk or st in sk or sk.endswith(st) or st in k:
                out.append(k)
                break
    return out


def find_instance(canonical, object_pos, body_names=None, body_xpos=None):
    if not canonical:
        return None
    c = category_candidates(canonical, object_pos)
    if c:
        return c[0]
    if body_names:
        for n in body_names:
            if canonical in n:
                return n
    return None


def pos_of(name, priv):
    op = priv.get("object_pos", {})
    if name in op:
        return np.asarray(op[name], dtype=float).reshape(3)
    bn = priv.get("body_names", [])
    bx = priv.get("body_xpos")
    if bx is not None and len(bn) == len(bx):
        for i, n in enumerate(bn):
            if n == name:
                return np.asarray(bx[i], dtype=float).reshape(3)
    return None


# ---------------------------------------------------------------------------
# NL segmentation
# ---------------------------------------------------------------------------

def parse_actions(nl: str):
    low = nl.lower()
    spans = []
    for verb, pat in VERB_RE:
        for m in re.finditer(pat, low):
            spans.append((m.start(), m.end(), verb))
    spans.sort()
    actions = []
    for i, (s, e, verb) in enumerate(spans):
        seg_end = spans[i + 1][0] if i + 1 < len(spans) else len(low)
        actions.append({"verb": verb, "clause": low[e:seg_end].strip(" ,.")})
    return actions


def _split_object_target(clause: str):
    m = _TARGET_PREP.search(clause)
    if not m:
        return clause.strip(), None, None
    head = clause[:m.start()].strip()
    rel_word = m.group(1)
    tail = clause[m.end():].strip()
    if rel_word == "on top of":
        relation = "on_top_of"
    elif rel_word in ("into", "inside"):
        relation = "into"
    elif rel_word == "in":
        relation = "in"
    elif rel_word == "under":
        relation = "under"
    elif rel_word == "on":
        relation = "on"
    elif rel_word == "to the":
        relation = "to_dir"
    else:
        relation = "on"
    return head, relation, tail


# ---------------------------------------------------------------------------
# object cues (geometric disambiguation)
# ---------------------------------------------------------------------------

def object_cues(clause: str) -> dict:
    c = {}
    m = re.search(r"\bbetween\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)(?:$|,|\.)", clause)
    if m:
        c["between"] = (m.group(1).strip(), m.group(2).strip())
    m = re.search(r"\bnext\s+to\s+(?:the\s+)?(.+?)(?:$|,|\.|\band\b)", clause)
    if m:
        c["next_to"] = m.group(1).strip()
    m = re.search(r"\bon\s+the\s+(\w+)", clause)
    if m and m.group(1) not in ("left", "right", "front", "back", "table", "top"):
        c["on_ref"] = m.group(1)
    if "table center" in clause or re.search(r"\bin\s+the\s+middle\b", clause):
        c["from_center"] = True
    if re.search(r"\b(middle)\s+\w+", clause) or re.search(r"\bthe\s+middle\b", clause):
        c["middle"] = True
    if re.search(r"\b(left|right)\b", clause):
        c["hpos"] = "right" if re.search(r"\bright\b", clause) else "left"
    if re.search(r"\bat\s+the\s+(front|back)\b", clause) or re.search(r"\bon\s+the\s+(front|back)\b", clause):
        c["vpos"] = "front" if re.search(r"\bfront\b", clause) else "back"
    return c


def _dominant_axis(ps_valid):
    if len(ps_valid) < 2:
        return 0
    xs = [p[1][0] for p in ps_valid]
    ys = [p[1][1] for p in ps_valid]
    return 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1


def _nearest(pv, target):
    if not pv:
        return None
    return sorted(pv, key=lambda cp: (float(np.linalg.norm(cp[1] - target)), cp[0]))[0][0]


def _nearest_xy(pv, target):
    """Nearest by HORIZONTAL (xy) distance only -- for "on top of X" cues, where
    3D distance to a tall fixture's centre misjudges the object resting on top."""
    if not pv:
        return None
    return sorted(pv, key=lambda cp: (float(np.linalg.norm(cp[1][:2] - target[:2])), cp[0]))[0][0]


def _resolve_ref(phrase, priv):
    object_pos = priv.get("object_pos", {})
    phrase = phrase.strip()
    canon = _category_in_phrase(phrase, build_vocab(priv))
    if canon and find_instance(canon, object_pos):
        return find_instance(canon, object_pos)
    words = [w for w in phrase.split() if w not in ("the", "a", "of", "and")]
    for w in reversed(words):
        inst = find_instance(w, object_pos, priv.get("body_names"), priv.get("body_xpos"))
        if inst:
            return inst
    return None


def resolve_object(canonical, cue_clause, priv, used):
    object_pos = priv.get("object_pos", {})
    cands = category_candidates(canonical, object_pos)
    if not cands:
        return None
    cands = [c for c in cands if c not in used] or cands
    if len(cands) == 1:
        return cands[0]

    cols = {col for col in COLORS if re.search(r"\b" + col + r"\b", cue_clause)}
    if cols:
        exact = [c for c in cands if name_colors(c) == cols]
        if exact:
            cands = exact
        else:
            sub = [c for c in cands if name_colors(c) and name_colors(c) <= cols]
            if sub:
                cands = sub
            else:
                inter = [c for c in cands if name_colors(c) & cols]
                if inter:
                    cands = inter
    if len(cands) == 1:
        return cands[0]

    cues = object_cues(cue_clause)
    ps = [(c, pos_of(c, priv)) for c in cands]
    pv = [(c, p) for c, p in ps if p is not None]

    if cues.get("between"):
        a = _resolve_ref(cues["between"][0], priv)
        b = _resolve_ref(cues["between"][1], priv)
        pa, pb = pos_of(a, priv), pos_of(b, priv)
        if pa is not None and pb is not None:
            return _nearest(pv, (pa + pb) / 2.0)
    if cues.get("next_to"):
        pr = pos_of(_resolve_ref(cues["next_to"], priv), priv)
        if pr is not None:
            return _nearest(pv, pr)
    if cues.get("on_ref"):
        pr = pos_of(_resolve_ref(cues["on_ref"], priv), priv)
        if pr is not None:
            return _nearest_xy(pv, pr)
    if cues.get("middle") and len(pv) >= 3:
        ax = _dominant_axis(pv)
        vals = sorted(p[1][ax] for p in pv)
        med = vals[len(vals) // 2]
        return min(pv, key=lambda cp: (abs(cp[1][ax] - med), cp[0]))[0]
    if cues.get("from_center"):
        return _nearest(pv, np.zeros(3))
    if cues.get("hpos") and len(pv) >= 2:
        ax = _dominant_axis(pv)
        return (max if cues["hpos"] == "right" else min)(pv, key=lambda cp: cp[1][ax])[0]
    if cues.get("vpos") and len(pv) >= 2:
        ax = _dominant_axis(pv)
        return (max if cues["vpos"] == "front" else min)(pv, key=lambda cp: cp[1][ax])[0]

    eef = priv.get("eef_pos")
    if eef is not None:
        return _nearest(pv, np.asarray(eef, dtype=float).reshape(3))
    return sorted(cands)[0]


# ---------------------------------------------------------------------------
# target / region resolution
# ---------------------------------------------------------------------------

def _site(site_names, parent, suffix):
    if not parent:
        return None
    tgt = f"{parent}_{suffix}"
    if tgt in site_names:
        return tgt
    for s in site_names:
        if s.endswith(tgt) or s.endswith(f"_{parent}_{suffix}"):
            return s
    return None


def _site_parent(site_names, stem):
    for s in site_names:
        m = re.match(rf"(\w*{re.escape(stem)}\w*_\d+)_", s)
        if m:
            return m.group(1)
    return None


def _table_region(site_names, stem, direction):
    for s in site_names:
        if s.endswith(f"_{direction}_region") and stem in s:
            return s
    return None


def _stem(canonical):
    if not canonical:
        return ""
    return CATEGORY_ALIAS.get(canonical, [canonical])[0]


def resolve_target(pred, tgt_canon, clause, priv, ctx):
    object_pos = priv.get("object_pos", {})
    site_names = priv.get("site_names", [])
    if tgt_canon in (None, "it"):
        if pred == "in" and ctx.get("last_drawer_region"):
            return ctx["last_drawer_region"]
        tgt_canon = ctx.get("last_fixture")

    if pred in ("turnon", "turnoff"):
        return find_instance("stove", object_pos)

    if pred in ("open", "close"):
        if tgt_canon and "microwave" in tgt_canon:
            return find_instance("microwave", object_pos)
        parent = find_instance("cabinet", object_pos) or _site_parent(site_names, "cabinet")
        level = ctx.get("last_drawer_level") or "middle"
        m = re.search(r"\b(top|middle|bottom)\b", clause)
        if m:
            level = m.group(1)
        return _site(site_names, parent, f"{level}_region") or parent

    if pred == "in":
        if "under" in clause:
            parent = find_instance(tgt_canon, object_pos) or _site_parent(site_names, _stem(tgt_canon))
            return _site(site_names, parent, "bottom_region") or parent
        m = re.search(r"\b(front|left|right|back)\s+compartment\b", clause)
        if m:
            parent = find_instance("caddy", object_pos) or _site_parent(site_names, "caddy")
            return _site(site_names, parent, f"{m.group(1)}_contain_region") or parent
        if tgt_canon == "microwave":
            parent = find_instance("microwave", object_pos)
            return _site(site_names, parent, "heating_region") or parent
        m = re.search(r"\b(top|middle|bottom)\b", clause)
        if m and any(w in clause for w in ("drawer", "shelf", "cabinet")):
            parent = (find_instance("cabinet", object_pos) or _site_parent(site_names, "cabinet")
                      or find_instance("shelf", object_pos) or _site_parent(site_names, "shelf"))
            return _site(site_names, parent, f"{m.group(1)}_region") or parent
        if tgt_canon == "shelf":
            parent = find_instance("shelf", object_pos) or _site_parent(site_names, "shelf")
            return _site(site_names, parent, "top_region") or parent
        parent = find_instance(tgt_canon, object_pos) or _site_parent(site_names, _stem(tgt_canon))
        return _site(site_names, parent, "contain_region") or parent

    if pred == "on":
        m = re.search(r"\bto\s+the\s+(left|right|front|back)\b", clause)
        if m:
            d = m.group(1)
            m2 = re.search(r"to\s+the\s+(?:left|right|front|back)\s+of\s+(?:the\s+)?(.+)", clause)
            refphrase = m2.group(1) if m2 else clause
            stem = "stove"
            for s in ("porcelain_mug", "stove", "plate", "mug", "bowl", "caddy",
                      "cabinet", "box", "basket", "ramekin"):
                if s in refphrase:
                    stem = s
                    break
            reg = _table_region(site_names, stem, d)
            if reg:
                return reg
        if "on top of" in clause:
            parent = find_instance(tgt_canon, object_pos) or _site_parent(site_names, _stem(tgt_canon))
            return _site(site_names, parent, "top_side") or _site(site_names, parent, "top_region") or parent
        if tgt_canon == "stove" or (tgt_canon and "stove" in tgt_canon):
            parent = find_instance("stove", object_pos)
            return _site(site_names, parent, "cook_region") or parent
        parent = find_instance(tgt_canon, object_pos)
        r = _site(site_names, parent, "top_region") or _site(site_names, parent, "top_side")
        return r or parent

    return None


# ---------------------------------------------------------------------------
# per-action -> atoms
# ---------------------------------------------------------------------------

def _target_canonical(tail, vocab):
    if not tail:
        return None
    matches = _all_category_matches(tail, vocab)
    canon_matches = [(p, c) for p, c in matches if c in CATEGORY_ALIAS]
    pool = canon_matches if canon_matches else matches
    return max(pool)[1] if pool else None


def _level_of(region):
    for lv in ("top", "middle", "bottom"):
        if f"_{lv}_region" in str(region):
            return lv
    return None


def _object_specs(head, vocab, ctx, uses_it):
    if uses_it and ctx.get("obj_context"):
        return [ctx["obj_context"]]
    h = re.sub(r"^both\s+", "", head.strip())
    h = re.sub(r"\s+(and|then)$", "", h).strip()
    if "both" in head and " and " in h:
        parts = [p.strip() for p in re.split(r"\s+and\s+", h)]
        return [(_category_in_phrase(p, vocab), "the " + p) for p in parts]
    canon = _category_in_phrase(h, vocab) or _first_category(h, vocab)
    if "both" in head:  # plural same category -> all instances
        return [(canon, "the " + h), (canon, "the " + h)]
    return [(canon, "the " + h)]


def action_to_atoms(act, priv, vocab, ctx, used):
    clause = act["clause"]
    verb = act["verb"]
    atoms = []

    if verb == "pick":
        canon = _first_category(clause, vocab)
        ctx = {**ctx, "obj_context": (canon, clause)}
        return [], ctx

    if verb in ("put", "place", "push", "stack"):
        head, relation, tail = _split_object_target(clause)
        tgt_canon = _target_canonical(tail, vocab)
        uses_it = bool(re.search(r"\bit\b", head)) or (head.strip() in ("them", "it"))

        if verb == "stack" or relation == "on_top_of":
            pred = "on"
        elif relation in ("into", "in"):
            pred = "in"
            if tgt_canon in SURFACE_ON:
                pred = "on"
        elif relation == "under":
            pred = "in"
        else:
            pred = "on"
        if tgt_canon == "shelf" and relation != "on_top_of":
            pred = "in"
        if verb == "push" or relation == "to_dir":
            pred = "on"

        if (head.strip() in ("them", "it") and not ctx.get("obj_context")
                and ctx.get("last_stack_base")):
            obj_specs = [(None, head)]
        else:
            obj_specs = _object_specs(head, vocab, ctx, uses_it)

        if relation == "to_dir":
            tgt = resolve_target(pred, tgt_canon, clause, priv, ctx)
        elif tgt_canon and tgt_canon not in FIXTURE_CATEGORIES and pred == "on":
            tgt = resolve_object(tgt_canon, "the " + (tail or ""), priv, used)
        else:
            tgt = resolve_target(pred, tgt_canon, clause, priv, ctx)

        consumes_drawer = None
        if pred == "in" and tgt and "_region" in str(tgt) and ("drawer" in clause or "inside" in clause):
            consumes_drawer = tgt
            ctx = {**ctx, "last_drawer_region": tgt, "last_drawer_level": _level_of(tgt)}
        if tgt_canon in ("stove", "cabinet", "shelf", "microwave", "caddy"):
            ctx = {**ctx, "last_fixture": tgt_canon}
        if verb == "stack" and tgt:
            ctx = {**ctx, "last_stack_base": tgt}

        for oc, ocue in obj_specs:
            if oc is None and ctx.get("last_stack_base"):
                obj = ctx["last_stack_base"]
            else:
                obj = resolve_object(oc, ocue, priv, used) if oc else None
            if obj:
                used.add(obj)
            atoms.append({"pred": pred, "obj": obj, "target": tgt,
                          "consumes_drawer": consumes_drawer if len(obj_specs) == 1 else None})
            if (obj and "moka" in obj and pred == "on" and tgt and "cook_region" in str(tgt)
                    and len(category_candidates("moka_pot", priv.get("object_pos", {}))) >= 2):
                atoms.append({"pred": "turnon", "obj": find_instance("stove", priv.get("object_pos", {})),
                              "target": None, "consumes_drawer": None})
        return atoms, ctx

    if verb in ("open", "close"):
        if "microwave" in clause:
            tgt_canon = "microwave"
        elif "drawer" in clause:
            tgt_canon = "drawer"
        elif "it" in clause.split() or clause.strip() == "it":
            tgt_canon = ctx.get("last_fixture") or "drawer"
        else:
            tgt_canon = "drawer"
        atom_obj = resolve_target(verb, tgt_canon, clause, priv, ctx)
        if verb == "open" and atom_obj and "_region" in str(atom_obj):
            ctx = {**ctx, "last_drawer_region": atom_obj, "last_drawer_level": _level_of(atom_obj)}
        return [{"pred": verb, "obj": atom_obj, "target": None, "consumes_drawer": None}], ctx

    if verb == "turn_on":
        return [{"pred": "turnon", "obj": find_instance("stove", priv.get("object_pos", {})),
                 "target": None, "consumes_drawer": None}], {**ctx, "last_fixture": "stove"}
    if verb == "turn_off":
        return [{"pred": "turnoff", "obj": find_instance("stove", priv.get("object_pos", {})),
                 "target": None, "consumes_drawer": None}], {**ctx, "last_fixture": "stove"}

    return [], ctx


# ---------------------------------------------------------------------------
# grounder entry points
# ---------------------------------------------------------------------------

def _grasp_of(pred: str, obj: Optional[str]) -> Optional[str]:
    """Object that must be grasped before this goal atom (matches libero_env rule:
    grasp_obj = obj for on/in/stack, else None)."""
    return obj if pred in ("on", "in", "stack") else None


def ground_task_state(task_desc: str, priv: dict) -> Tuple[
        List[Tuple[str, Optional[str], Optional[str]]], List[Optional[str]]]:
    """Ground one NL instruction against a privileged state.

    Args:
        task_desc: natural-language task instruction (``task.language``).
        priv: privileged MuJoCo state from ``_extract_privileged_state`` (must
            contain ``object_pos``, ``site_names`` and optionally ``body_*`` /
            ``eef_pos``).

    Returns:
        (goal_atoms, grasp_objs) -- two parallel lists.
        goal_atoms: ``[(pred, obj, target), ...]``
        grasp_objs: ``[obj_or_None, ...]`` -- the grasp precondition per atom.
    """
    if not task_desc or not priv:
        return [], []
    vocab = build_vocab(priv)
    actions = parse_actions(task_desc)
    ctx = {"obj_context": None, "last_fixture": None,
           "last_drawer_region": None, "last_drawer_level": None,
           "last_stack_base": None}
    used = set()
    atom_dicts = []
    for act in actions:
        atoms, ctx = action_to_atoms(act, priv, vocab, ctx, used)
        atom_dicts.extend(atoms)

    consumed = {a["consumes_drawer"] for a in atom_dicts if a["consumes_drawer"]}
    goal_atoms: List[Tuple[str, Optional[str], Optional[str]]] = []
    grasp_objs: List[Optional[str]] = []
    seen = set()
    for a in atom_dicts:
        if a["pred"] == "open" and a["obj"] in consumed:
            continue
        key = (a["pred"], a["obj"], a["target"])
        if key in seen:
            continue
        seen.add(key)
        goal_atoms.append(key)
        grasp_objs.append(_grasp_of(key[0], key[1]))
    return goal_atoms, grasp_objs
