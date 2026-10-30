import json
import os
from typing import List, Tuple

import torch
from PIL import Image


def synthetic_dir(root: str, task: str) -> str:
    path = os.path.join(root, "synthetic", task)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} not found: run `python -m commerge.prepare --synthetic`")
    return path


def synthetic_images(root: str, task: str, preprocess) -> Tuple[List[str], torch.Tensor]:
    path = synthetic_dir(root, task)
    files = sorted(f for f in os.listdir(path) if f.endswith(".png"))
    return files, torch.stack([preprocess(Image.open(os.path.join(path, f)).convert("RGB")) for f in files])


def synthetic_batches(root: str, task: str, preprocess, batch_size: int):
    _, xs = synthetic_images(root, task, preprocess)
    return [(xs[i:i + batch_size], torch.zeros(len(xs[i:i + batch_size]), dtype=torch.long)) for i in range(0, len(xs), batch_size)]


def synthetic_labelled(root: str, task: str, preprocess) -> Tuple[torch.Tensor, torch.Tensor]:
    files, xs = synthetic_images(root, task, preprocess)
    labels: dict = json.load(open(os.path.join(synthetic_dir(root, task), "labels.json")))
    keep = [i for i, f in enumerate(files) if labels.get(f) is not None]
    return xs[keep], torch.tensor([labels[files[i]] for i in keep])
