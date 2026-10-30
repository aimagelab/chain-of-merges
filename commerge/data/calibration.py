import math
import random
from collections import defaultdict
from typing import List, Optional, Sequence

from torch.utils.data import ConcatDataset, Subset


def class_balanced_indices(labels: Sequence[int], num_samples: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)
    for idxs in by_class.values():
        rng.shuffle(idxs)
    classes = sorted(by_class)
    base, rem = divmod(num_samples, len(classes))
    extra = set(rng.sample(classes, rem))
    chosen, leftovers = [], []
    for c in classes:
        quota = base + (1 if c in extra else 0)
        chosen += by_class[c][:quota]
        leftovers += by_class[c][quota:]
    if len(chosen) < num_samples:
        rng.shuffle(leftovers)
        chosen += leftovers[: num_samples - len(chosen)]
    rng.shuffle(chosen)
    return chosen


def random_indices(labels: Sequence[int], num_samples: int, seed: int = 0) -> List[int]:
    return random.Random(seed).sample(range(len(labels)), min(num_samples, len(labels)))


def few_class_indices(labels: Sequence[int], classes: Sequence[int], num_samples: int, fraction: float, rng: random.Random) -> List[int]:
    k = max(1, math.floor(fraction * len(classes)))
    keep = set(rng.sample(list(classes), k))
    pool = [i for i, lab in enumerate(labels) if int(lab) in keep]
    return rng.sample(pool, min(num_samples, len(pool)))


def sample_indices(labels: Sequence[int], num_samples: int, seed: int, sampling: str) -> List[int]:
    if sampling == "balanced":
        return class_balanced_indices(labels, min(num_samples, len(labels)), seed)
    if sampling == "random":
        return random_indices(labels, num_samples, seed)
    raise ValueError(f"data.calibration.sampling must be 'balanced' or 'random', got '{sampling}'")


def calibration_dataset(task, cal: dict, few_class_rng: Optional[random.Random] = None, classes: Optional[Sequence[int]] = None):
    split, n, seed, sampling = cal["split"], cal["num_samples"], cal["seed"], cal.get("sampling", "balanced")
    labels = task.labels(split)
    if cal.get("max_class_fraction"):
        return Subset(task.split(split), few_class_indices(labels, classes, n, cal["max_class_fraction"], few_class_rng))
    if cal.get("top_up") and len(labels) < n:
        parts = [Subset(task.split(split), sample_indices(labels, len(labels), seed, sampling))]
        extra = cal["top_up"]
        parts.append(Subset(task.split(extra), sample_indices(task.labels(extra), n - len(labels), seed, sampling)))
        return ConcatDataset(parts)
    return Subset(task.split(split), sample_indices(labels, n, seed, sampling))
