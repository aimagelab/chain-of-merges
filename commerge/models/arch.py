from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class BlockState:
    hidden: torch.Tensor
    kwargs: Dict[str, Any] = field(default_factory=dict)
    mask: Optional[torch.Tensor] = None


@dataclass
class StemSpec:
    weight_name: str
    kernel: int
    in_channels: int


@dataclass
class TailSpec:
    weight_name: str
    module_name: str


class ArchAdapter:
    stages: List[List[str]] = []

    def blocks(self, model: nn.Module) -> nn.ModuleList:
        raise NotImplementedError

    def block_prefix(self, index: int) -> str:
        raise NotImplementedError

    def stem(self, model: nn.Module) -> Optional[StemSpec]:
        return None

    def tail(self, model: nn.Module) -> Optional[TailSpec]:
        return None

    def tail_inputs(self, model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def stem_inputs(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def prepare(self, model: nn.Module, batch) -> BlockState:
        raise NotImplementedError

    def run_block(self, block: nn.Module, state: BlockState) -> torch.Tensor:
        raise NotImplementedError

    def forward_batch(self, model: nn.Module, batch) -> None:
        raise NotImplementedError

    def reset_noise(self) -> None:
        pass

    def token_mask(self, batch) -> Optional[torch.Tensor]:
        return None
