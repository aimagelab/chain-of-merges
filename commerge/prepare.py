import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
RELEASE = "https://github.com/aimagelab/chain-of-merges/releases/download/v1.0"
ASSETS = {"llama1b_adapters": ("llama32_1b_lora_adapters.tar.gz", "checkpoints/llama32_1b_lora"),
          "synthetic": ("synthetic_sd15.tar.gz", "synthetic")}
VISION_DIRS = {"stanford_cars": "stanford_cars/cars_train", "dtd": "dtd", "eurosat": "EuroSAT_splits", "gtsrb": "gtsrb", "mnist": "MNIST/raw",
               "resisc45": "resisc45", "sun397": "SUN397", "svhn": "svhn"}
SETTINGS = ["vit_b32_lora", "vit_l14_lora", "vit_b32_full", "vit_l14_full"]
GATED = ["llama3_8b_lora", "llama32_1b_lora"]


def data_root() -> Path:
    return Path(os.environ.get("COMMERGE_DATA", "data")).resolve()


def ensure_vision(root, tasks) -> None:
    from commerge import prepare_vision
    root = Path(root)
    missing = [t for t in tasks if not (root / VISION_DIRS[t]).exists()]
    if any(not (root / "heads" / d).exists() for d in prepare_vision.HEAD_DIRS):
        missing.append("heads")
    if missing:
        print(f"[prepare] building vision data under {root}: {missing}")
        prepare_vision.prepare(str(root), missing)


def ensure_asset(root, name: str) -> Path:
    archive, folder = ASSETS[name]
    target = Path(root) / folder
    if target.exists():
        return target
    Path(root).mkdir(parents=True, exist_ok=True)
    tmp = Path(root) / archive
    print(f"[prepare] downloading {RELEASE}/{archive}")
    urllib.request.urlretrieve(f"{RELEASE}/{archive}", tmp)
    with tarfile.open(tmp) as tar:
        tar.extractall(Path(root), filter="data")
    tmp.unlink()
    return target


def prefetch_models(gated: bool) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    for setting in SETTINGS + (GATED if gated else []):
        cfg = yaml.safe_load(open(REPO / "configs" / "models" / f"{setting}.yaml"))["model"]
        print(f"[prepare] {setting}: {cfg['base']}")
        snapshot_download(cfg["base"], allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
        for ckpt in cfg["checkpoints"]:
            if ckpt.startswith("checkpoints/"):
                continue
            if ":" in ckpt:
                repo, filename = ckpt.split(":", 1)
                hf_hub_download(repo, filename)
            else:
                snapshot_download(ckpt)


def prefetch_nli() -> None:
    from commerge.data.nli import NLI_TASKS, load_raw
    for task in NLI_TASKS:
        print(f"[prepare] nli: {task}")
        load_raw(task, str(data_root()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download / build the data, checkpoints and release assets (everything goes under $COMMERGE_DATA, default ./data).")
    ap.add_argument("--vision", action="store_true", help="the 8 vision datasets and the CLIP zero-shot heads")
    ap.add_argument("--tasks", nargs="+", default=list(VISION_DIRS), choices=list(VISION_DIRS), help="restrict --vision")
    ap.add_argument("--nli", action="store_true", help="the 6 NLI datasets")
    ap.add_argument("--models", action="store_true", help="CLIP models, public LoRA adapters and full fine-tuned checkpoints")
    ap.add_argument("--gated", action="store_true", help="with --models: also Llama-3-8B and Llama-3.2-1B (accept the licences, then `huggingface-cli login`)")
    ap.add_argument("--llama1b-adapters", action="store_true", help="the six Llama-3.2-1B LoRA adapters of the paper (GitHub release asset)")
    ap.add_argument("--synthetic", action="store_true", help="the Stable Diffusion calibration images of the paper (GitHub release asset)")
    a = ap.parse_args(argv)
    if not any([a.vision, a.nli, a.models, a.llama1b_adapters, a.synthetic]):
        ap.print_help()
        return 1
    root = data_root()
    if a.vision:
        ensure_vision(root, a.tasks)
    if a.nli:
        prefetch_nli()
    if a.models:
        prefetch_models(a.gated)
    if a.llama1b_adapters:
        ensure_asset(root, "llama1b_adapters")
    if a.synthetic:
        ensure_asset(root, "synthetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
