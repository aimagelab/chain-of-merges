import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from commerge.methods.base import MergeContext
from commerge.models.arch import BlockState
from commerge.models.task_vectors import TaskVector
from commerge.registry import METHODS


@dataclass
class LayerwiseConfig:
    alpha: float = 1.0
    scale: float = 1.0
    chain: bool = True
    task_weights: str = "gram_offdiag"
    weight_offset: float = 0.1
    normalize_features: bool = True
    normalize_by_tokens: bool = True
    dtype: str = "float32"
    pinv_rtol: float = 1e-6
    gram_spill_dir: Optional[str] = None
    gram_cache: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerwiseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class _Captured(Exception):
    pass


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _state_to(state: BlockState, device) -> BlockState:
    return BlockState(_to_device(state.hidden, device), _to_device(state.kwargs, device), _to_device(state.mask, device))


@torch.no_grad()
def capture_stage_input(arch, block, module_name: str, state: BlockState) -> torch.Tensor:
    box = []

    def hook(_, inputs):
        box.append(inputs[0].detach())
        raise _Captured

    handle = block.get_submodule(module_name).register_forward_pre_hook(hook)
    try:
        arch.run_block(block, state)
    except _Captured:
        pass
    finally:
        handle.remove()
    return box[0]


def features(x: torch.Tensor, mask: Optional[torch.Tensor], cfg: LayerwiseConfig) -> torch.Tensor:
    x = x[mask] if mask is not None else x.reshape(-1, x.shape[-1])
    x = x.to(getattr(torch, cfg.dtype))
    if cfg.normalize_features:
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x


def importance_weights(grams: List[torch.Tensor], cfg: LayerwiseConfig) -> torch.Tensor:
    if cfg.task_weights == "none":
        return torch.ones(len(grams), device=grams[0].device)
    if cfg.task_weights != "gram_offdiag":
        raise ValueError(f"unknown task_weights '{cfg.task_weights}'")
    eye = torch.eye(grams[0].shape[0], dtype=torch.bool, device=grams[0].device)
    r = torch.stack([g[~eye].abs().mean() for g in grams])
    return (r - r.min() + cfg.weight_offset) / (r.max() - r.min() + cfg.weight_offset)


def solve_layer(deltas: List[torch.Tensor], grams: List[torch.Tensor], tokens: List[int], cfg: LayerwiseConfig,
                recorder: Optional[list] = None) -> torch.Tensor:
    eye = torch.eye(grams[0].shape[0], dtype=torch.bool, device=grams[0].device)
    grams = [torch.where(eye, g, g * cfg.alpha) for g in grams]
    w = importance_weights(grams, cfg)
    if recorder is not None:
        recorder.append(importance_weights(grams, LayerwiseConfig(task_weights="gram_offdiag", weight_offset=cfg.weight_offset)).tolist())
    if cfg.normalize_by_tokens:
        grams = [g / n for g, n in zip(grams, tokens)]
    gram_sum = sum(wi * g for wi, g in zip(w, grams))
    num = sum(wi * (d @ g) for wi, d, g in zip(w, deltas, grams))
    return cfg.scale * (num @ torch.linalg.pinv(gram_sum, rtol=cfg.pinv_rtol))


class LayerwiseMerger:
    def __init__(self, ctx: MergeContext):
        self.ctx = ctx
        self.cfg = LayerwiseConfig.from_dict(ctx.cfg)
        self.arch = ctx.arch
        self.dev = ctx.device
        self.tvs = ctx.task_vectors
        self.dtype = getattr(torch, self.cfg.dtype)
        self.recorder = ctx.extras.get("importance_recorder")

    def _solve_and_write(self, names, grams, tokens, merged_params, pre_params):
        for name in names:
            deltas = [self.tvs[t][name].to(self.dev, self.dtype).flatten(1) for t in range(len(self.tvs))]
            new_delta = solve_layer(deltas, grams, tokens, self.cfg, self.recorder)
            merged_params[name].copy_((pre_params[name].to(self.dev, self.dtype).flatten(1) + new_delta).view_as(merged_params[name]))

    def _gram(self, blocks):
        gram, count = None, 0
        for f in blocks:
            if gram is None:
                gram = torch.zeros(f.shape[-1], f.shape[-1], device=self.dev, dtype=self.dtype)
            gram += f.T @ f
            count += f.shape[0]
        return gram, count

    def _start(self):
        pre_params = {k: v.detach() for k, v in self.ctx.pretrained.named_parameters()}
        merged = deepcopy(self.ctx.pretrained).to(self.dev).eval()
        merged_params = dict(merged.named_parameters())
        for k in self.tvs[0]:
            mean = torch.stack([tv[k].float() for tv in self.tvs]).mean(0)
            merged_params[k].add_(mean.to(self.dev, merged_params[k].dtype))
        return merged, merged_params, pre_params

    def _merge_stem(self, merged, merged_params, pre_params):
        stem = self.arch.stem(merged)
        if stem is None or stem.weight_name not in self.tvs[0]:
            return
        grams, tokens = [], []
        for batches in self.ctx.calib:
            g, n = self._gram(features(self.arch.stem_inputs(_to_device(b, self.dev)), None, self.cfg) for b in batches)
            grams.append(g)
            tokens.append(n)
        self._solve_and_write([stem.weight_name], grams, tokens, merged_params, pre_params)

    def _result(self, merged_params, pre_params) -> TaskVector:
        return {k: (merged_params[k] - pre_params[k].to(self.dev)).detach().cpu() for k in self.tvs[0]}

    def _stages(self):
        for b in range(len(self.arch.blocks(self.ctx.pretrained))):
            prefix = self.arch.block_prefix(b)
            for stage in self.arch.stages:
                names = [f"{prefix}{m}.weight" for m in stage]
                if all(n in self.tvs[0] for n in names):
                    yield b, stage, names

    @torch.no_grad()
    def run(self) -> TaskVector:
        self.arch.reset_noise()
        return self._run_chained() if self.cfg.chain else self._run_independent()

    def _run_chained(self) -> TaskVector:
        arch = self.arch
        if not all(isinstance(b, list) for b in self.ctx.calib):
            raise ValueError("chain=true needs a finite calibration set (data.calibration.num_samples: <int>)")
        merged, merged_params, pre_params = self._start()
        self._merge_stem(merged, merged_params, pre_params)
        states = [[_state_to(arch.prepare(merged, b), "cpu") for b in task_calib] for task_calib in self.ctx.calib]
        blocks = arch.blocks(merged)
        stages = list(self._stages())
        for b in tqdm(range(len(blocks)), desc="CoM"):
            block = blocks[b]
            for stage_block, stage, names in stages:
                if stage_block != b:
                    continue
                grams, tokens = [], []
                for task_states in states:
                    g, n = self._gram(
                        features(capture_stage_input(arch, block, stage[0], _state_to(s, self.dev)),
                                 None if s.mask is None else s.mask.to(self.dev), self.cfg)
                        for s in task_states)
                    grams.append(g)
                    tokens.append(n)
                self._solve_and_write(names, grams, tokens, merged_params, pre_params)
                del grams
            for task_states in states:
                for s in task_states:
                    s.hidden = arch.run_block(block, _state_to(s, self.dev)).cpu()
            torch.cuda.empty_cache()
        tail = arch.tail(merged)
        if tail is not None and tail.weight_name in self.tvs[0]:
            grams, tokens = [], []
            for task_states in states:
                g, n = self._gram(features(arch.tail_inputs(merged, _to_device(s.hidden, self.dev)), None, self.cfg) for s in task_states)
                grams.append(g)
                tokens.append(n)
            self._solve_and_write([tail.weight_name], grams, tokens, merged_params, pre_params)
        return self._result(merged_params, pre_params)

    def _gram_cache_dir(self) -> Optional[Path]:
        if not self.cfg.gram_cache:
            return None
        ex = self.ctx.extras
        key = {"model": ex["model_id"], "checkpoints": ex["checkpoints"], "calibration": ex["calibration"], "tasks": list(ex["task_names"]),
               "normalize_features": self.cfg.normalize_features, "normalize_by_tokens": self.cfg.normalize_by_tokens,
               "cls_weight": 0.0, "dtype": self.cfg.dtype}
        ident = json.dumps(key, sort_keys=True, default=str)
        path = Path(ex.get("cache_dir", "outputs/cache")) / "grams" / hashlib.sha1(ident.encode()).hexdigest()[:16]
        path.mkdir(parents=True, exist_ok=True)
        if not (path / "IDENT.json").exists():
            (path / "IDENT.json").write_text(ident)
        return path

    def _run_independent(self) -> TaskVector:
        arch, n_tasks = self.arch, len(self.tvs)
        merged, merged_params, pre_params = self._start()
        self._merge_stem(merged, merged_params, pre_params)
        items = [(names, arch.block_prefix(b) + stage[0]) for b, stage, names in self._stages()]
        tail = arch.tail(merged)
        if tail is not None and tail.weight_name in self.tvs[0]:
            items.append(([tail.weight_name], tail.module_name))
        paths = [path for _, path in items]
        store = GramStore(self.cfg.gram_spill_dir, self._gram_cache_dir())
        merged.cpu()
        torch.cuda.empty_cache()
        for t in tqdm(range(n_tasks), desc="RegMean"):
            if store.is_done(t):
                continue
            model_t = self.ctx.task_model(t).to(self.dev).eval()
            acc = {p: [None, 0] for p in paths}
            current = {"mask": None}

            def make_hook(path):
                def hook(_, inputs):
                    f = features(inputs[0].detach(), current["mask"], self.cfg)
                    entry = acc[path]
                    if entry[0] is None:
                        entry[0] = torch.zeros(f.shape[-1], f.shape[-1], device=self.dev, dtype=self.dtype)
                    entry[0] += f.T @ f
                    entry[1] += f.shape[0]
                return hook

            handles = [model_t.get_submodule(p).register_forward_pre_hook(make_hook(p)) for p in paths]
            try:
                for batch in self.ctx.calib[t]:
                    mask = arch.token_mask(batch)
                    current["mask"] = None if mask is None else mask.to(self.dev)
                    arch.forward_batch(model_t, batch)
            finally:
                for h in handles:
                    h.remove()
            for p, (g, n) in acc.items():
                store.put(t, p, g.cpu(), n)
            store.mark_done(t)
            del model_t
            torch.cuda.empty_cache()
        merged.to(self.dev)
        for names, path in items:
            loaded = [store.get(t, path) for t in range(n_tasks)]
            self._solve_and_write(names, [g.to(self.dev) for g, _ in loaded], [n for _, n in loaded], merged_params, pre_params)
            del loaded
        store.close()
        return self._result(merged_params, pre_params)


class GramStore:
    def __init__(self, spill_dir=None, persist_dir=None):
        self.mem: dict = {}
        self.persist = persist_dir is not None
        self.dir = Path(persist_dir) if persist_dir is not None else (Path(tempfile.mkdtemp(prefix="grams_", dir=spill_dir)) if spill_dir else None)

    def is_done(self, t: int) -> bool:
        return self.persist and (self.dir / f"done_{t}").exists()

    def mark_done(self, t: int) -> None:
        if self.persist:
            (self.dir / f"done_{t}").touch()

    def _file(self, t: int, path: str) -> Path:
        return self.dir / f"{t}__{path}.pt"

    def put(self, t: int, path: str, gram: torch.Tensor, tokens: int) -> None:
        if self.dir is None:
            self.mem[(t, path)] = (gram, tokens)
        else:
            torch.save((gram, tokens), self._file(t, path))

    def get(self, t: int, path: str):
        return self.mem[(t, path)] if self.dir is None else torch.load(self._file(t, path))

    def close(self) -> None:
        if self.dir is not None and not self.persist:
            shutil.rmtree(self.dir, ignore_errors=True)


@METHODS.register("com")
def com(ctx: MergeContext) -> TaskVector:
    return LayerwiseMerger(ctx).run()


@METHODS.register("regmean")
def regmean(ctx: MergeContext) -> TaskVector:
    ctx.cfg = {**dict(chain=False, task_weights="none", normalize_features=False), **ctx.cfg}
    return LayerwiseMerger(ctx).run()
