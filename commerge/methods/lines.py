from commerge.methods.base import MergeContext
from commerge.methods.ops import common_keys, layer_index, task_weights
from commerge.models.task_vectors import TaskVector
from commerge.registry import METHODS


@METHODS.register("lines")
def merge(ctx: MergeContext) -> TaskVector:
    alpha, beta = float(ctx.cfg.get("alpha", 1.0)), float(ctx.cfg.get("beta", 0.0))
    keys, weights = common_keys(ctx.task_vectors), task_weights(ctx)
    last = max([i for i in map(layer_index, keys) if i is not None], default=0)
    out = {}
    for key in keys:
        idx = layer_index(key) or 0
        factor = alpha + (beta * idx / last if last > 0 else 0.0)
        out[key] = factor * sum(w * tv[key].float() for w, tv in zip(weights, ctx.task_vectors))
    return out
