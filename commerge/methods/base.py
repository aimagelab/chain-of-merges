from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from commerge.models.task_vectors import TaskVector


@dataclass
class MergeContext:
    pretrained: nn.Module
    task_vectors: List[TaskVector]
    cfg: Dict[str, Any]
    device: torch.device
    arch: Optional[Any] = None
    calib: Optional[List[Any]] = None
    task_model: Optional[Callable[[int], nn.Module]] = None
    scaling: Optional[List[float]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tasks(self) -> int:
        return len(self.task_vectors)
