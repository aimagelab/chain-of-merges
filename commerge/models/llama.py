import json
import os
from typing import Any, Dict, List

import torch

from commerge.models.arch import ArchAdapter, BlockState
from commerge.models.task_vectors import TaskVector

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def load_pretrained(model_cfg: Dict[str, Any]):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base"], num_labels=3, torch_dtype=DTYPES[model_cfg.get("dtype", "bfloat16")], cache_dir=model_cfg.get("cache_dir"))
    model.config.pad_token_id = model.config.eos_token_id if model.config.pad_token_id is None else model.config.pad_token_id
    model.config.use_cache = False
    return model.eval().requires_grad_(False)


def load_tokenizer(model_cfg: Dict[str, Any]):
    from transformers import AutoTokenizer
    from commerge.data.nli import prepare_tokenizer
    return prepare_tokenizer(AutoTokenizer.from_pretrained(model_cfg["base"], cache_dir=model_cfg.get("cache_dir")))


def _fetch(checkpoint: str, name: str, cache_dir=None) -> str:
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(checkpoint, name, cache_dir=cache_dir)


def read_adapter(checkpoint: str, cache_dir=None):
    from safetensors.torch import load_file
    cfg = json.load(open(_fetch(checkpoint, "adapter_config.json", cache_dir)))
    try:
        state = load_file(_fetch(checkpoint, "adapter_model.safetensors", cache_dir))
    except FileNotFoundError:
        state = torch.load(_fetch(checkpoint, "adapter_model.bin", cache_dir), map_location="cpu")
    return cfg, state


def _strip(key: str) -> str:
    return key.replace("base_model.model.", "", 1)


def lora_task_vector_from_state(cfg: Dict[str, Any], state: Dict[str, torch.Tensor]) -> TaskVector:
    scaling = cfg["lora_alpha"] / cfg["r"]
    delta: TaskVector = {}
    for key, A in state.items():
        if "lora_A" in key:
            B = state[key.replace("lora_A", "lora_B")]
            delta[f"{_strip(key.split('.lora_A')[0])}.weight"] = (B.float() @ A.float()) * scaling
    return delta


def task_head_from_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    for key, value in state.items():
        if _strip(key) in ("score.weight", "score.modules_to_save.default.weight", "classifier.modules_to_save.default.weight"):
            return value.float()
    raise KeyError("no score head found in adapter state dict")


def _full_task_vector(checkpoint: str, base: Dict[str, torch.Tensor]) -> TaskVector:
    from safetensors.torch import load_file
    state = load_file(os.path.join(checkpoint, "model.safetensors"))
    unknown = [k for k in state if k not in base]
    if unknown:
        raise KeyError(f"{checkpoint} has parameters unknown to the base model: {unknown[:3]}")
    return {name: state[name].float() - w for name, w in base.items() if name != "score.weight"}


def load_task_vectors(model_cfg: Dict[str, Any], pretrained=None) -> List[TaskVector]:
    kind = model_cfg.get("finetune", "lora")
    if kind == "lora":
        return [lora_task_vector_from_state(*read_adapter(c, model_cfg.get("cache_dir"))) for c in model_cfg["checkpoints"]]
    if kind == "full":
        base = {k: v.detach().float().cpu() for k, v in pretrained.named_parameters()}
        return [_full_task_vector(c, base) for c in model_cfg["checkpoints"]]
    raise ValueError(f"unknown finetune type '{kind}'")


def load_task_heads(model_cfg: Dict[str, Any]) -> List[torch.Tensor]:
    if model_cfg.get("finetune", "lora") == "full":
        from safetensors import safe_open
        heads = []
        for c in model_cfg["checkpoints"]:
            with safe_open(os.path.join(c, "model.safetensors"), "pt") as f:
                heads.append(f.get_tensor("score.weight").float())
        return heads
    return [task_head_from_state(read_adapter(c, model_cfg.get("cache_dir"))[1]) for c in model_cfg["checkpoints"]]


def block_kwargs(model, hidden: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
    inner = model.model
    cache_position = torch.arange(hidden.shape[1], device=hidden.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = inner._update_causal_mask(attention_mask, hidden, cache_position, None, False)
    return dict(attention_mask=causal_mask, position_ids=position_ids, cache_position=cache_position,
                position_embeddings=inner.rotary_emb(hidden, position_ids))


class LlamaArch(ArchAdapter):
    stages = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    def __init__(self, noise_std: float = 0.0, noise_seed: int = 0):
        self.noise_std, self.noise_seed, self._generators = noise_std, noise_seed, {}

    def reset_noise(self):
        self._generators = {}

    def _noise(self, like: torch.Tensor) -> torch.Tensor:
        if like.device not in self._generators:
            self._generators[like.device] = torch.Generator(device=like.device).manual_seed(self.noise_seed)
        noise = torch.randn(like.shape, generator=self._generators[like.device], device=like.device, dtype=torch.float32)
        return noise.to(like.dtype) * self.noise_std

    def blocks(self, model):
        return model.model.layers

    def block_prefix(self, index):
        return f"model.layers.{index}."

    @torch.no_grad()
    def prepare(self, model, batch):
        device = model.model.embed_tokens.weight.device
        ids = batch["input_ids"].to(device)
        mask = batch.get("attention_mask")
        mask = torch.ones_like(ids) if mask is None else mask.to(device)
        hidden = model.model.embed_tokens(ids)
        kwargs = block_kwargs(model, hidden, mask)
        if self.noise_std > 0:
            hidden = hidden + self._noise(hidden)
        return BlockState(hidden=hidden, kwargs=kwargs, mask=mask.bool())

    def run_block(self, block, state):
        out = block(state.hidden, **state.kwargs)
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def forward_batch(self, model, batch):
        device = model.model.embed_tokens.weight.device
        mask = batch.get("attention_mask")
        hook = None
        if self.noise_std > 0:
            hook = model.model.embed_tokens.register_forward_hook(lambda module, inputs, out: out + self._noise(out))
        try:
            model.model(input_ids=batch["input_ids"].to(device), attention_mask=None if mask is None else mask.to(device))
        finally:
            if hook is not None:
                hook.remove()

    def token_mask(self, batch):
        mask = batch.get("attention_mask")
        return None if mask is None else mask.bool()
