import torch

from commerge.methods.base import MergeContext
from commerge.methods.ops import average_others, task_weights
from commerge.methods.tsv import _compose, tsv_factors
from commerge.models.task_vectors import TaskVector, matrix_keys
from commerge.registry import METHODS

_CACHE: dict = {}


def thin_factors(mat: torch.Tensor, rank: int):
    u, s, v = torch.svd_lowrank(mat, q=min(rank + 16, min(mat.shape)), niter=8)
    return u[:, :rank], s[:rank], v[:, :rank]


def core_space(mats, rank: int):
    facs = [thin_factors(m, rank) for m in mats]
    qu, _ = torch.linalg.qr(torch.cat([f[0] for f in facs], dim=1))
    qv, _ = torch.linalg.qr(torch.cat([f[2] for f in facs], dim=1))
    cores = [(qu.T @ u) * s @ (qv.T @ v).T for u, s, v in facs]
    return qu, qv, cores


def _ident(ctx, k):
    tv = ctx.task_vectors[0][k]
    return k, tv.data_ptr(), tuple(tv.shape), len(ctx.task_vectors), float(tv.double().sum())


@METHODS.register("core_tsv")
def merge_core_tsv(ctx: MergeContext) -> TaskVector:
    keys, weights, rank = matrix_keys(ctx.task_vectors), task_weights(ctx), int(ctx.cfg.get("rank", 16))
    out = {}
    for k in keys:
        ident = _ident(ctx, k)
        if ident not in _CACHE:
            qu, qv, cores = core_space([tv[k].to(ctx.device, torch.float64) for tv in ctx.task_vectors], rank)
            fac = tsv_factors(cores)
            _CACHE[ident] = (qu.float().cpu(), qv.float().cpu(), tuple(f.float().cpu() if f.is_floating_point() else f.cpu() for f in fac))
        qu, qv, fac = _CACHE[ident]
        core = _compose(fac, weights, ctx.device)
        out[k] = (qu.to(ctx.device).double() @ core @ qv.to(ctx.device).double().T).float().cpu()
    out.update(average_others(ctx.task_vectors, keys))
    return out


_CACHE_ISO: dict = {}


@METHODS.register("core_iso")
def merge_core_iso(ctx: MergeContext) -> TaskVector:
    keys, weights, rank = matrix_keys(ctx.task_vectors), task_weights(ctx), int(ctx.cfg.get("rank", 16))
    if len(set(weights)) != 1:
        raise ValueError("core_iso supports one scale for all tasks")
    out = {}
    for k in keys:
        ident = _ident(ctx, k)
        if ident not in _CACHE_ISO:
            qu, qv, cores = core_space([tv[k].to(ctx.device, torch.float64) for tv in ctx.task_vectors], rank)
            u, sv, vh = torch.linalg.svd(sum(cores), full_matrices=False)
            _CACHE_ISO[ident] = (qu.float().cpu(), qv.float().cpu(), ((u * sv.mean()) @ vh).float().cpu())
        qu, qv, core = _CACHE_ISO[ident]
        out[k] = (weights[0] * (qu.to(ctx.device).double() @ core.to(ctx.device).double() @ qv.to(ctx.device).double().T)).float().cpu()
    out.update(average_others(ctx.task_vectors, keys))
    return out
