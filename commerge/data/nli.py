from typing import Dict, List, Optional

import datasets
from datasets import Dataset, DatasetDict

from commerge.data.base import TaskData

NLI_TASKS = ["snli", "mnli", "sick", "qnli", "rte", "scitail"]

HF_IDS = {
    "snli": "stanfordnlp/snli", "mnli": "SetFit/mnli", "sick": "sick",
    "qnli": "SetFit/qnli", "rte": "SetFit/rte", "scitail": "allenai/scitail",
}
HF_CONFIGS = {"scitail": "tsv_format"}
TEXT_KEYS = {
    "snli": ("premise", "hypothesis"), "mnli": ("text1", "text2"), "sick": ("sentence_A", "sentence_B"),
    "qnli": ("text1", "text2"), "rte": ("text1", "text2"), "scitail": ("premise", "hypothesis"),
}
MASKED_CLASS: Dict[str, Optional[int]] = {
    "snli": None, "mnli": None, "sick": None, "qnli": 1, "rte": 1, "scitail": 2,
}
VAL_FRACTION = {"mnli": 0.2, "qnli": 0.2, "rte": 0.5}
MAX_LENGTH = {"scitail": 1000}
DEFAULT_MAX_LENGTH = 2000
CLASS_NAMES = ["entailment", "neutral", "contradiction"]


def remap_labels(ds: Dataset, task: str) -> Dataset:
    if task == "scitail":
        ds = ds.filter(lambda e: e["label"] in ("entails", "neutral"))
        return ds.map(lambda e: {"label": 0 if e["label"] == "entails" else 1})
    ds = ds.filter(lambda e: e["label"] in (0, 1, 2))
    if task in ("qnli", "rte"):
        ds = ds.map(lambda e: {"label": 2 if e["label"] == 1 else e["label"]})
    return ds


def make_splits(raw: DatasetDict, task: str, val_fraction: Optional[float] = None, seed: int = 42) -> Dict[str, Dataset]:
    raw = DatasetDict({k: remap_labels(v, task) for k, v in raw.items()})
    if task in VAL_FRACTION:
        frac = VAL_FRACTION[task] if val_fraction is None else val_fraction
        parts = raw["validation"].train_test_split(test_size=frac, shuffle=True, seed=seed)
        return {"train": raw["train"], "val": parts["test"], "test": parts["train"]}
    return {"train": raw["train"], "val": raw["validation"], "test": raw["test"]}


def tokenize_split(ds: Dataset, task: str, tokenizer, max_length: int) -> Dataset:
    k1, k2 = TEXT_KEYS[task]

    def fn(ex):
        return tokenizer(ex[k1], ex[k2], truncation=True, max_length=max_length)

    ds = ds.map(fn, batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
    return ds.rename_column("label", "labels")


def prepare_tokenizer(tokenizer):
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_task_data(raw: DatasetDict, task: str, tokenizer, max_length: Optional[int] = None,
                    val_fraction: Optional[float] = None) -> TaskData:
    tokenizer = prepare_tokenizer(tokenizer)
    max_length = max_length or MAX_LENGTH.get(task, DEFAULT_MAX_LENGTH)
    splits = {k: tokenize_split(v, task, tokenizer, max_length) for k, v in make_splits(raw, task, val_fraction).items()}

    def collate(examples: List[dict]):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    return TaskData(
        name=task, num_classes=3, class_names=CLASS_NAMES, train=splits["train"], val=splits["val"],
        test=splits["test"], labels_fn=lambda split: splits[split]["labels"], collate_fn=collate, masked_class=MASKED_CLASS[task])


def load_raw(task: str, root: Optional[str] = None) -> DatasetDict:
    cache = None if root is None else f"{root}/datasets"
    return datasets.load_dataset(HF_IDS[task], HF_CONFIGS.get(task), cache_dir=cache, trust_remote_code=True)


def load_nli_task(name: str, tokenizer, root: Optional[str] = None, max_length: Optional[int] = None) -> TaskData:
    if name not in NLI_TASKS:
        raise ValueError(f"unknown NLI task '{name}'")
    return build_task_data(load_raw(name, root), name, tokenizer, max_length)
