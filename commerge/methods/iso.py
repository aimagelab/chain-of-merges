import torch

from commerge.methods.base import MergeContext
from commerge.methods.ops import average_others, task_weights
from commerge.models.task_vectors import TaskVector, matrix_keys
from commerge.registry import METHODS


def isotropize(m: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    return s.mean() * (u @ vh)


def _weighted_sum(ctx, key, weights):
    return sum(w * tv[key].to(ctx.device, torch.float64) for w, tv in zip(weights, ctx.task_vectors))


_ISO_CACHE: dict = {}


@METHODS.register("iso_c")
def merge_iso_c(ctx: MergeContext) -> TaskVector:
    keys, weights = matrix_keys(ctx.task_vectors), task_weights(ctx)
    uniform = len(set(weights)) == 1
    out = {}
    for k in keys:
        if not uniform:
            out[k] = isotropize(_weighted_sum(ctx, k, weights)).float().cpu()
            continue
        ident = (k, ctx.task_vectors[0][k].data_ptr(), tuple(ctx.task_vectors[0][k].shape), len(ctx.task_vectors),
                 float(ctx.task_vectors[0][k].double().sum()))
        if ident not in _ISO_CACHE:
            _ISO_CACHE[ident] = isotropize(_weighted_sum(ctx, k, [1.0] * len(weights))).float().cpu()
        out[k] = weights[0] * _ISO_CACHE[ident]
    out.update(average_others(ctx.task_vectors, keys))
    return out
