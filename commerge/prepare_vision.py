import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MANIFESTS = Path(__file__).parent / "data" / "assets" / "manifests"
VISION_TASKS = ["mnist", "svhn", "gtsrb", "dtd", "eurosat", "stanford_cars", "resisc45", "sun397"]
ALL = VISION_TASKS + ["heads"]

URLS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/{}.gz",
    "svhn": "http://ufldl.stanford.edu/housenumbers/{}",
    "gtsrb": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/{}",
    "dtd": "https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz",
    "eurosat": "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
}
CHECKSUMS = {
    "mnist": {"train-images-idx3-ubyte": "6bbc9ace898e44ae57da46a324031adb", "train-labels-idx1-ubyte": "a25bea736e30d166cdddb491f175f624",
              "t10k-images-idx3-ubyte": "2646ac647ad5339dbf082846283269ea", "t10k-labels-idx1-ubyte": "27ae3e4e09519cfbb04c329615203637"},
    "svhn": {"train_32x32.mat": "e26dedcc434d2e4c54c9b2d4a06d8373", "test_32x32.mat": "eb5a983be6a315427106f1b164d9cef3"},
    "gtsrb": {"GTSRB-Training_fixed.zip": "513f3c79a4c5141765e10e952eaa2478", "GTSRB_Final_Test_Images.zip": "c7e4e6327067d32654124b0fe9e82185",
              "GTSRB_Final_Test_GT.zip": "fe31e9c9270bbcd7b84b7f21a9d9d9e5"},
}
MIRRORS = {"svhn": ("MarMaster/svhn", "370f42ab3e68b00a14a51f9a6e408b3d41bc020a"),
           "gtsrb": ("Zzyy2000/GTSRB", "93173a3d09d1c0eb60f3ca09af0e65527be2bbf1")}
GTSRB_TEST_GT = MANIFESTS.parent / "gtsrb_GT-final_test.csv.gz"
HF = {"stanford_cars": ("tanganke/stanford_cars", "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"),
      "resisc45": ("timm/resisc45", "fe12fc5f1b7606543b0355eda392f1ddc54625c6"),
      "sun397": ("tanganke/sun397", "9c4afd6174afa3974f209a0ef099dbbe244dfd27"),
      "heads": ("nik-dim/tall_masks", "c8d352f3802fc221a0e993fe6335ed03dfe91d73")}
PARQUETS = {
    "stanford_cars": [f"data/{s}-{i:05d}-of-00002.parquet" for s in ("test", "train") for i in range(2)],
    "resisc45": [f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")],
    "sun397": [f"data/{s}-{i:05d}-of-00018.parquet" for s in ("train", "test") for i in range(18)],
}
HEAD_TASKS = {"stanford_cars": "Cars", "dtd": "DTD", "eurosat": "EuroSAT", "gtsrb": "GTSRB", "mnist": "MNIST",
              "resisc45": "RESISC45", "sun397": "SUN397", "svhn": "SVHN"}
HEAD_DIRS = {"ViT-B-32-CLIP": "ViT-B-32", "ViT-L-14-CLIP": "VIT-L-14"}


def log(msg):
    print(f"[prepare_vision] {msg}", flush=True)


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(data) -> str:
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data).convert("RGB"))
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def load_manifest(task):
    with gzip.open(MANIFESTS / f"{task}.json.gz", "rt") as f:
        return json.load(f)


def download(url: str, dest: Path, md5: str = None, tries: int = 5, mirror=None) -> Path:
    dest = Path(dest)
    if dest.exists() and (md5 is None or md5_file(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for t in range(2 if mirror else tries):
        try:
            log(f"download {url}")
            part = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            part.rename(dest)
            if md5 is not None and md5_file(dest) != md5:
                raise IOError(f"md5 mismatch for {dest.name}")
            return dest
        except Exception as e:
            log(f"  failed ({e}); retry {t + 1}/{2 if mirror else tries}")
            time.sleep(10 * (t + 1))
    if mirror:
        repo, rev = mirror
        log(f"  falling back to the Hugging Face mirror {repo}")
        cache = dest.parent / ".hf_mirror"
        shutil.copyfile(hf_file(repo, dest.name, rev, cache), dest)
        shutil.rmtree(cache, ignore_errors=True)
        if md5 is not None and md5_file(dest) != md5:
            dest.unlink()
            raise IOError(f"md5 mismatch for {dest.name} from {repo}")
        return dest
    raise RuntimeError(f"could not download {url}")


def hf_file(repo, filename, revision, cache_dir, repo_type="dataset", tries=8) -> str:
    from huggingface_hub import hf_hub_download
    for t in range(tries):
        try:
            return hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, cache_dir=str(cache_dir))
        except Exception as e:
            log(f"  hf download {repo}/{filename} failed ({str(e)[:80]}); retry {t + 1}/{tries} in 60 s")
            time.sleep(60)
    raise RuntimeError(f"could not download {repo}/{filename}")


def parquet_images(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image", "label"]):
        for row, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            yield row["bytes"], label


def hub_rows(task, cache_dir, keep=False):
    repo, rev = HF[task]
    row = 0
    for fn in PARQUETS[task]:
        path = hf_file(repo, fn, rev, cache_dir)
        for data, label in parquet_images(path):
            yield row, fn.split("/")[-1].split("-")[0], data, label
            row += 1
        if not keep:
            shutil.rmtree(Path(cache_dir) / ("datasets--" + repo.replace("/", "--")), ignore_errors=True)


def write_from_hub(task, entries, root: Path, cache_dir, verify, keep):
    wanted = {r: (d, h) for r, d, h in entries}
    assert len(wanted) == len(entries), "a hub row is used twice in the manifest"
    done = 0
    for row, _, data, _ in hub_rows(task, cache_dir, keep):
        if row not in wanted:
            continue
        dest, h = wanted.pop(row)
        if verify and pixel_hash(data) != h:
            raise RuntimeError(f"{task}: pixel hash mismatch for {dest} (hub row {row}); the pinned dataset changed?")
        out = root / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        done += 1
        if done % 5000 == 0:
            log(f"  {task}: {done}/{len(entries)} files")
    if wanted:
        raise RuntimeError(f"{task}: {len(wanted)} manifest rows were not found in the downloaded parquet files")


def prep_mnist(root, dl, **_):
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, md5 in CHECKSUMS["mnist"].items():
        if (raw / name).exists() and md5_file(raw / name) == md5:
            continue
        gz = download(URLS["mnist"].format(name), dl / (name + ".gz"))
        with gzip.open(gz, "rb") as f, open(raw / name, "wb") as out:
            shutil.copyfileobj(f, out)
        if md5_file(raw / name) != md5:
            raise RuntimeError(f"md5 mismatch for MNIST {name}")


def prep_svhn(root, dl, **_):
    (root / "svhn").mkdir(parents=True, exist_ok=True)
    for name, md5 in CHECKSUMS["svhn"].items():
        download(URLS["svhn"].format(name), root / "svhn" / name, md5, mirror=MIRRORS["svhn"])


def prep_gtsrb(root, dl, **_):
    base = root / "gtsrb" / "gtsrb"
    base.mkdir(parents=True, exist_ok=True)
    for name in ("GTSRB-Training_fixed.zip", "GTSRB_Final_Test_Images.zip"):
        z = download(URLS["gtsrb"].format(name), dl / name, CHECKSUMS["gtsrb"][name], mirror=MIRRORS["gtsrb"])
        zipfile.ZipFile(z).extractall(base)
    try:
        z = download(URLS["gtsrb"].format("GTSRB_Final_Test_GT.zip"), dl / "GTSRB_Final_Test_GT.zip", CHECKSUMS["gtsrb"]["GTSRB_Final_Test_GT.zip"], tries=2)
        zipfile.ZipFile(z).extractall(base)
    except RuntimeError:
        log("  using the bundled copy of GT-final_test.csv")
        with gzip.open(GTSRB_TEST_GT, "rb") as f:
            (base / "GT-final_test.csv").write_bytes(f.read())
    for p in ("GTSRB/Training", "GTSRB/Final_Test/Images", "GT-final_test.csv"):
        if not (base / p).exists():
            raise RuntimeError(f"gtsrb: {p} missing after extraction")


def prep_dtd(root, dl, **_):
    tar = download(URLS["dtd"], dl / "dtd-r1.0.1.tar.gz")
    d = root / "dtd"
    if not (d / "images").exists():
        with tarfile.open(tar) as t:
            t.extractall(root)
    for split in ("train", "val", "test"):
        for line in open(d / "labels" / f"{split}1.txt"):
            rel = line.strip()
            if rel:
                dest = d / split / rel
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(d / "images" / rel, dest)


def prep_eurosat(root, dl, **_):
    z = download(URLS["eurosat"], dl / "EuroSAT.zip")
    tmp = dl / "eurosat_extracted"
    if not tmp.exists():
        zipfile.ZipFile(z).extractall(tmp)
    man = load_manifest("eurosat")
    for split in ("train", "val", "test"):
        for rel in man[split]:
            dest = root / "EuroSAT_splits" / split / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp / "2750" / rel, dest)
    shutil.rmtree(tmp, ignore_errors=True)


def prep_cars(root, dl, verify=True, keep=False, **_):
    import scipy.io as sio
    man = load_manifest("stanford_cars")
    d = root / "stanford_cars"
    tr = man["train"]
    entries = [(r, f"stanford_cars/cars_test/{f}", h) for r, f, h in zip(man["rows"], man["fnames"], man["pix"])]
    entries += [(r, f"stanford_cars/cars_train/{f}", h) for r, f, h in zip(tr["rows"], tr["fnames"], tr["pix"])]
    write_from_hub("stanford_cars", entries, root, dl / "hf", verify, keep)

    def annotations(fnames, classes):
        ann = np.zeros(len(fnames), dtype=[("bbox_x1", "O"), ("bbox_y1", "O"), ("bbox_x2", "O"), ("bbox_y2", "O"), ("class", "O"), ("fname", "O")])
        for i, (f, c) in enumerate(zip(fnames, classes)):
            ann[i] = (0, 0, 0, 0, c + 1, f)
        return {"annotations": ann}
    sio.savemat(d / "cars_test_annos_withlabels.mat", annotations(man["fnames"], man["classes"]))
    (d / "devkit").mkdir(exist_ok=True)
    sio.savemat(d / "devkit" / "cars_train_annos.mat", annotations(tr["fnames"], tr["classes"]))
    sio.savemat(d / "devkit" / "cars_meta.mat", {"class_names": np.array(man["class_names"], dtype=object).reshape(1, -1)})


def prep_resisc45(root, dl, verify=True, keep=False, **_):
    man = load_manifest("resisc45")
    classes = man["classes"]
    entries = [(r, f"resisc45/NWPU-RESISC45/{classes[c]}/{n}", h) for n, c, r, h in zip(man["names"], man["class_idx"], man["rows"], man["pix"])]
    write_from_hub("resisc45", entries, root, dl / "hf", verify, keep)
    for split in ("train", "val", "test"):
        (root / "resisc45" / f"resisc45-{split}.txt").write_text("\n".join(man["lists"][split]) + "\n")


def prep_sun397(root, dl, verify=True, keep=False, **_):
    man = load_manifest("sun397")
    classes = man["classes"]
    entries = []
    for split in ("train", "test"):
        s = man[split]
        entries += [(r, f"SUN397/{split}/{classes[c]}/{n}", h) for n, c, r, h in zip(s["names"], s["class_idx"], s["rows"], s["pix"])]
    write_from_hub("sun397", entries, root, dl / "hf", verify, keep)


def prep_heads(root, dl, **_):
    import importlib.abc
    import importlib.machinery
    import types

    import torch

    class Stub(torch.nn.Module):
        def __setstate__(self, s):
            self.__dict__.update(s)

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == "src" or name.startswith("src."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda n: type(n, (Stub,), {})
            return m

        def exec_module(self, m):
            pass

    repo, rev = HF["heads"]
    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        for head_dir, arch in HEAD_DIRS.items():
            for task, name in HEAD_TASKS.items():
                dest = root / "heads" / head_dir / f"{task}_head.pt"
                if dest.exists():
                    continue
                p = hf_file(repo, f"single_task/{arch}/head_{name}Val.pt", rev, dl / "hf", repo_type="model")
                head = torch.load(p, map_location="cpu", weights_only=False)
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save(head.weight.detach().float().cpu().clone(), dest)
    finally:
        sys.meta_path.remove(finder)


RUN = {"mnist": prep_mnist, "svhn": prep_svhn, "gtsrb": prep_gtsrb, "dtd": prep_dtd, "eurosat": prep_eurosat,
       "stanford_cars": prep_cars, "resisc45": prep_resisc45, "sun397": prep_sun397, "heads": prep_heads}


def prepare(data_dir, tasks=ALL, verify=True, keep=False):
    root = Path(data_dir).resolve()
    dl = root / ".downloads"
    for task in tasks:
        marker = root / ".prepared" / f"{task}.json"
        if marker.exists():
            log(f"{task}: already prepared")
            continue
        log(f"{task}: preparing")
        RUN[task](root, dl, verify=verify, keep=keep)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"task": task, "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
        log(f"{task}: done")
    if not keep:
        shutil.rmtree(dl, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Rebuild the eight vision datasets and the CLIP zero-shot heads from public sources, verified against the shipped manifests.")
    p.add_argument("--data-dir", default=os.environ.get("COMMERGE_DATA", "data"))
    p.add_argument("--tasks", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--no-verify", action="store_true", help="skip the per-image pixel-hash check (faster)")
    p.add_argument("--keep-downloads", action="store_true")
    a = p.parse_args(argv)
    prepare(a.data_dir, a.tasks, verify=not a.no_verify, keep=a.keep_downloads)


if __name__ == "__main__":
    main()
