from typing import List

import torch

from commerge.methods.base import MergeContext
from commerge.methods.ops import average_others, task_weights
from commerge.models.task_vectors import TaskVector, matrix_keys
from commerge.registry import METHODS


def _polar(x: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


def tsv_factors(mats: List[torch.Tensor]):
    n, (out_dim, in_dim) = len(mats), mats[0].shape
    rank = min(out_dim, in_dim)
    k = rank // n
    u_cat = torch.zeros(out_dim, rank, dtype=torch.float64, device=mats[0].device)
    s_cat = torch.zeros(rank, dtype=torch.float64, device=mats[0].device)
    v_cat = torch.zeros(rank, in_dim, dtype=torch.float64, device=mats[0].device)
    slot_task = torch.zeros(rank, dtype=torch.long)
    for t, mat in enumerate(mats):
        u, s, vh = torch.linalg.svd(mat.double(), full_matrices=False)
        slot = slice(t * k, (t + 1) * k)
        u_cat[:, slot], s_cat[slot], v_cat[slot] = u[:, :k], s[:k], vh[:k]
        slot_task[slot] = t
    return _polar(u_cat), s_cat, _polar(v_cat), slot_task


def _compose(factors, weights: List[float], device) -> torch.Tensor:
    pu, s_cat, pv, slot_task = factors
    dev = device
    w = torch.tensor(weights, dtype=torch.float64, device=dev)[slot_task.to(dev)]
    return (pu.to(dev).double() * (w * s_cat.to(dev).double())) @ pv.to(dev).double()


def tsv_matrix(mats: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    return _compose(tsv_factors(mats), weights, mats[0].device)


_CACHE: dict = {}


@METHODS.register("tsv")
def merge(ctx: MergeContext) -> TaskVector:
    keys, weights = matrix_keys(ctx.task_vectors), task_weights(ctx)
    out = {}
    for k in keys:
        ident = (k, ctx.task_vectors[0][k].data_ptr(), tuple(ctx.task_vectors[0][k].shape), len(ctx.task_vectors),
                 float(ctx.task_vectors[0][k].double().sum()))
        if ident not in _CACHE:
            fac = tsv_factors([tv[k].to(ctx.device) for tv in ctx.task_vectors])
            _CACHE[ident] = tuple(f.float().cpu() if f.is_floating_point() else f.cpu() for f in fac)
        out[k] = _compose(_CACHE[ident], weights, ctx.device).float().cpu()
    out.update(average_others(ctx.task_vectors, keys))
    return out
