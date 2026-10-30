import torch

from commerge.methods.base import MergeContext
from commerge.methods.ops import common_keys, task_weights
from commerge.models.task_vectors import TaskVector
from commerge.registry import METHODS


@METHODS.register("consensus")
def merge(ctx: MergeContext) -> TaskVector:
    lam, k = float(ctx.cfg["tall_threshold"]), int(ctx.cfg["consensus_threshold"])
    weights, out = task_weights(ctx), {}
    for key in common_keys(ctx.task_vectors):
        taus = torch.stack([tv[key].float() for tv in ctx.task_vectors]).to(ctx.device)
        total = taus.sum(0)
        support = (taus.abs() >= lam * (total - taus).abs()).sum(0)
        w = torch.tensor(weights, device=taus.device).view(-1, *[1] * (taus.ndim - 1))
        out[key] = ((support >= k) * (w * taus).sum(0)).cpu()
    return out
