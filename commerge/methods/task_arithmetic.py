
from commerge.methods.base import MergeContext
from commerge.methods.ops import common_keys, task_weights
from commerge.models.task_vectors import TaskVector
from commerge.registry import METHODS


@METHODS.register("ta")
def merge(ctx: MergeContext) -> TaskVector:
    weights = task_weights(ctx)
    norm = ctx.num_tasks if ctx.cfg.get("merging_type", "sum") == "mean" else 1
    return {k: sum(w * tv[k].float() for w, tv in zip(weights, ctx.task_vectors)) / norm
            for k in common_keys(ctx.task_vectors)}
