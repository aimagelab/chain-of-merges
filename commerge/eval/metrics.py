import json
import os
from typing import Dict, List


def normalized_report(acc: Dict[str, float], reference: Dict[str, float], tasks: List[str]) -> Dict[str, float]:
    out = {}
    for t in tasks:
        out[f"{t}/acc"] = 100 * acc[t]
        out[f"{t}/norm_acc"] = 100 * acc[t] / reference[t] * 100
    out["avg/acc"] = sum(out[f"{t}/acc"] for t in tasks) / len(tasks)
    out["avg/norm_acc"] = sum(out[f"{t}/norm_acc"] for t in tasks) / len(tasks)
    return out


def load_reference(spec, cache_file: str, measure_fn) -> Dict[str, float]:
    if isinstance(spec, dict):
        return spec
    if spec != "measured":
        raise ValueError(f"eval.reference_acc must be a table or 'measured', got {spec!r}")
    if os.path.exists(cache_file):
        return json.load(open(cache_file))
    ref = measure_fn()
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    json.dump(ref, open(cache_file, "w"), indent=1)
    return ref
