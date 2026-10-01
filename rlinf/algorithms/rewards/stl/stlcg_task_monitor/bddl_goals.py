"""Read the exact bddl goal atoms (predicate + object + site/target) per task.

The goal atom's TARGET is often a SITE (e.g. wooden_cabinet_1_top_side,
flat_stove_1_cook_region, wine_rack_1_top_region), evaluated by robosuite via the
site's under()/in_box() — NOT the object-object check_ontop (0.03). Using the exact
site from bddl is what makes the re-derived predicate match goal_flags bit-for-bit.

bddl files are static (per task), so this needs no re-record.
"""
import os
import re
import sys

_LIBERO = os.environ.get("LIBERO_PATH")
if not _LIBERO:
    raise RuntimeError("LIBERO_PATH must point to the LIBERO checkout")
if _LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO)

_CACHE = {}


def _suite_benchmark(suite: str):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]()


def goal_atoms(suite: str, task_id: int):
    """Return list of (predicate, object_name, target_name) for the task's goal.

    predicate in {on, in, open, close, turnon, turnoff}. target_name is the second
    arg (a site or object body)."""
    key = (suite, task_id)
    if key in _CACHE:
        return _CACHE[key]
    from libero.libero import get_libero_path
    b = _suite_benchmark(suite)
    task = b.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    txt = open(bddl).read()
    m = re.search(r"\(:goal(.*)\)", txt, re.S)
    body = m.group(1) if m else ""
    atoms = []
    # match (On a b), (In a b), (Open a), ... args exclude ')' and whitespace
    for pred, a, b2 in re.findall(r"\((On|In)\s+([^\s)]+)\s+([^\s)]+)\)", body):
        atoms.append((pred.lower(), a, b2))
    for pred, a in re.findall(r"\((Open|Close|Turnon|Turnoff)\s+([^\s)]+)\)", body):
        atoms.append((pred.lower(), a, None))
    _CACHE[key] = atoms
    return atoms
