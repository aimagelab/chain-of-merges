import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModel

from commerge.models.arch import ArchAdapter, BlockState, StemSpec, TailSpec
from commerge.models.task_vectors import TaskVector


class ClipVisionEncoder(nn.Module):

    def __init__(self, vision_model: nn.Module, visual_projection: nn.Module):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(self.vision_model(pixel_values=pixel_values)[1])


def load_pretrained(model_cfg: Dict[str, Any]) -> ClipVisionEncoder:
    clip = CLIPModel.from_pretrained(model_cfg["base"], cache_dir=model_cfg.get("cache_dir"))
    return ClipVisionEncoder(clip.vision_model, clip.visual_projection).eval().requires_grad_(False)


def load_preprocess(model_cfg: Dict[str, Any]):
    proc = CLIPImageProcessor.from_pretrained(model_cfg["base"], cache_dir=model_cfg.get("cache_dir"))
    return lambda img: proc(img, return_tensors="pt")["pixel_values"][0]


def _lora_task_vector(adapter_id: str, cache_dir=None) -> TaskVector:
    import json
    from huggingface_hub import hf_hub_download

    def fetch(name):
        return hf_hub_download(adapter_id, name, cache_dir=cache_dir)

    cfg = json.load(open(fetch("adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    try:
        from safetensors.torch import load_file
        state = load_file(fetch("adapter_model.safetensors"))
    except Exception:
        state = torch.load(fetch("adapter_model.bin"), map_location="cpu")
    delta: TaskVector = {}
    for key, A in state.items():
        if "lora_A" not in key:
            continue
        B = state[key.replace("lora_A", "lora_B")]
        module = key.split(".lora_A")[0].replace("base_model.model.", "")
        delta[f"vision_model.{module}.weight"] = (B.float() @ A.float()) * scaling
    return delta


def _full_ft_task_vector(model_id: str, pretrained: ClipVisionEncoder, cache_dir=None) -> TaskVector:
    ft = CLIPVisionModel.from_pretrained(model_id, cache_dir=cache_dir).vision_model
    base = dict(pretrained.vision_model.named_parameters())
    return {f"vision_model.{n}": p.detach().float() - base[n].detach().float() for n, p in ft.named_parameters()}


def load_task_vectors(model_cfg: Dict[str, Any], pretrained: ClipVisionEncoder) -> List[TaskVector]:
    kind, cache = model_cfg["finetune"], model_cfg.get("cache_dir")
    if kind == "lora":
        return [_lora_task_vector(i, cache) for i in model_cfg["checkpoints"]]
    if kind == "full":
        return [_full_ft_task_vector(i, pretrained, cache) for i in model_cfg["checkpoints"]]
    if kind == "openclip":
        from commerge.models import openclip
        return [openclip.load_task_vector(i, pretrained, cache) for i in model_cfg["checkpoints"]]
    raise ValueError(f"unknown finetune type '{kind}'")


def load_head(root: str, head_dir: str, task: str) -> torch.Tensor:
    head = torch.load(os.path.join(root, "heads", head_dir, f"{task}_head.pt"), map_location="cpu").float()
    return head / head.norm(dim=-1, keepdim=True)


class ClipArch(ArchAdapter):
    stages = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.out_proj"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]

    def __init__(self, model: ClipVisionEncoder):
        conv = model.vision_model.embeddings.patch_embedding
        self.kernel, self.channels = conv.kernel_size[0], conv.in_channels

    def blocks(self, model):
        return model.vision_model.encoder.layers

    def block_prefix(self, index):
        return f"vision_model.encoder.layers.{index}."

    def stem(self, model):
        return StemSpec("vision_model.embeddings.patch_embedding.weight", self.kernel, self.channels)

    def stem_inputs(self, batch):
        k = self.kernel
        return F.unfold(batch[0], kernel_size=k, stride=k).transpose(1, 2)

    def tail(self, model):
        return TailSpec("visual_projection.weight", "visual_projection")

    def tail_inputs(self, model, hidden):
        return model.vision_model.post_layernorm(hidden[:, 0, :])

    def prepare(self, model, batch):
        pixels = batch[0].to(next(model.parameters()).device)
        vm = model.vision_model
        return BlockState(hidden=vm.pre_layrnorm(vm.embeddings(pixels)))

    def run_block(self, block, state):
        return block(state.hidden, None, None)[0]


    def forward_batch(self, model, batch):
        model(batch[0].to(next(model.parameters()).device))
