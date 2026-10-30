from commerge.methods.base import MergeContext
from commerge.methods.ops import common_keys, stack_flat, task_weights, ties_merge, ties_merge_streaming
from commerge.models.task_vectors import TaskVector, unflatten
from commerge.registry import METHODS


def _ties(ctx: MergeContext) -> TaskVector:
    cfg, keys = ctx.cfg, common_keys(ctx.task_vectors)
    if float(cfg.get("dare_p", 0.0)) == 0:
        return ties_merge_streaming(ctx.task_vectors, keys, task_weights(ctx), topk=float(cfg.get("topk", 1.0)),
                                    merging_type=cfg.get("merging_type", "mean"), device=ctx.device)
    stack = stack_flat(ctx.task_vectors, keys, "cpu")
    merged = ties_merge(stack, task_weights(ctx), topk=float(cfg.get("topk", 1.0)),
                        merging_type=cfg.get("merging_type", "mean"),
                        dare_p=float(cfg.get("dare_p", 0.0)), seed=int(cfg.get("dare_seed", 0)), device=ctx.device)
    return unflatten(merged.cpu(), ctx.task_vectors[0], keys)


METHODS.register("ties")(_ties)
METHODS.register("dare_ties")(_ties)
