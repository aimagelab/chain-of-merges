from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from torch.utils.data import Dataset


@dataclass
class TaskData:
    name: str
    num_classes: int
    class_names: List[str]
    train: Dataset
    val: Dataset
    test: Dataset
    labels_fn: Callable[[str], Sequence[int]]
    collate_fn: Optional[Callable] = None
    masked_class: Optional[int] = None

    def split(self, name: str) -> Dataset:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def labels(self, split: str) -> Sequence[int]:
        return self.labels_fn(split)
