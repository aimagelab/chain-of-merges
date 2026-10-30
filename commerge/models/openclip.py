import re
from typing import Dict

import torch

from commerge.models.task_vectors import TaskVector

_SIMPLE = {
    "class_embedding": "vision_model.embeddings.class_embedding",
    "positional_embedding": "vision_model.embeddings.position_embedding.weight",
    "conv1.weight": "vision_model.embeddings.patch_embedding.weight",
    "ln_pre.weight": "vision_model.pre_layrnorm.weight",
    "ln_pre.bias": "vision_model.pre_layrnorm.bias",
    "ln_post.weight": "vision_model.post_layernorm.weight",
    "ln_post.bias": "vision_model.post_layernorm.bias",
}
_BLOCK = {
    "ln_1": "layer_norm1", "ln_2": "layer_norm2",
    "attn.out_proj": "self_attn.out_proj", "mlp.c_fc": "mlp.fc1", "mlp.c_proj": "mlp.fc2",
}


def _split_qkv(out: Dict[str, torch.Tensor], layer: str, kind: str, fused: torch.Tensor) -> None:
    for name, chunk in zip(("q_proj", "k_proj", "v_proj"), fused.chunk(3, dim=0)):
        out[f"vision_model.encoder.layers.{layer}.self_attn.{name}.{kind}"] = chunk.contiguous()


def convert_visual(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = re.sub(r"^model\.", "", key)
        if not key.startswith("visual."):
            continue
        rest = key[len("visual."):]
        if rest == "proj":
            out["visual_projection.weight"] = value.t().contiguous()
        elif rest in _SIMPLE:
            out[_SIMPLE[rest]] = value
        elif (m := re.fullmatch(r"transformer\.resblocks\.(\d+)\.attn\.in_proj_(weight|bias)", rest)):
            _split_qkv(out, *m.groups(), value)
        elif (m := re.fullmatch(r"transformer\.resblocks\.(\d+)\.(.+)\.(weight|bias)", rest)) and m.group(2) in _BLOCK:
            layer, module, kind = m.groups()
            out[f"vision_model.encoder.layers.{layer}.{_BLOCK[module]}.{kind}"] = value
        else:
            raise KeyError(f"unexpected open_clip key: {key}")
    return out


def load_task_vector(spec: str, pretrained, cache_dir=None) -> TaskVector:
    if ":" in spec and not spec.startswith("/"):
        from huggingface_hub import hf_hub_download
        repo, filename = spec.split(":", 1)
        path = hf_hub_download(repo, filename, cache_dir=cache_dir)
    else:
        path = spec
    ft = convert_visual(torch.load(path, map_location="cpu", weights_only=False))
    base = dict(pretrained.named_parameters())
    missing = set(base) - set(ft)
    if missing:
        raise KeyError(f"checkpoint {spec} lacks parameters of the pretrained model, e.g. {sorted(missing)[:3]}")
    return {k: ft[k].float() - base[k].detach().float() for k in base}
