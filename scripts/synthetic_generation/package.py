import argparse
import json
import re
import shutil
from pathlib import Path

from classes import DATASETS

from commerge.data.vision import load_vision_task

FOLDERS = {"stanford_cars": "0_cars", "dtd": "1_dtd", "eurosat": "2_eurosat", "gtsrb": "3_gtsrb", "mnist": "4_mnist",
           "resisc45": "5_resisc45", "sun397": "6_sun397", "svhn": "7_svhn"}


def normalise(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def label_of(task, class_name, names):
    if task in ("gtsrb", "eurosat"):
        return DATASETS[task]["classes"].index(class_name)
    if normalise(class_name) in names:
        return names[normalise(class_name)]
    candidates = [k for n, k in names.items() if n.startswith(normalise(class_name))]
    return candidates[0] if len(candidates) == 1 else None


def main():
    ap = argparse.ArgumentParser(description="Write <out>/synthetic/<task>/{*.png, labels.json} from generate.py + downsample.py outputs.")
    ap.add_argument("generated", help="folder with <task>/metadata.json (generate.py output)")
    ap.add_argument("downsampled", help="folder with the downsampled images")
    ap.add_argument("--data-root", default="data", help="vision data (for the class names)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    for task, folder in FOLDERS.items():
        names = {normalise(n): k for k, n in enumerate(load_vision_task(task, a.data_root, lambda image: 0).class_names)}
        meta = json.load(open(Path(a.generated) / task / "metadata.json"))["images"]
        labels = {m["filename"]: label_of(task, m["class_name"], names) for m in meta}
        source = Path(a.downsampled) / folder if (Path(a.downsampled) / folder).exists() else Path(a.downsampled) / task
        target = Path(a.out) / "synthetic" / task
        target.mkdir(parents=True, exist_ok=True)
        for f in sorted(source.glob("*.png")):
            shutil.copy2(f, target / f.name)
        json.dump(labels, open(target / "labels.json", "w"), indent=0)
        print(task, len(labels), "images,", sum(v is not None for v in labels.values()), "with a label")


if __name__ == "__main__":
    main()
