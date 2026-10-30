import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg

from commerge.methods.base import MergeContext
from commerge.methods.fisher import linear_keys, task_fishers
from commerge.methods.ops import average_others, task_weights
from commerge.models.task_vectors import TaskVector
from commerge.registry import METHODS


def cg_diag_solve(diag: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor, num_iterations: int) -> torch.Tensor:
    d = diag.flatten().double().numpy()
    op = LinearOperator((d.size, d.size), matvec=lambda v: d * v, dtype=np.float64)
    x, _ = cg(op, rhs.flatten().double().numpy(), x0=x0.flatten().double().numpy(), maxiter=num_iterations)
    return torch.from_numpy(x).view_as(rhs).float()


@METHODS.register("mats")
def merge(ctx: MergeContext) -> TaskVector:
    keys, weights = linear_keys(ctx), task_weights(ctx)
    fishers = task_fishers(ctx, keys)
    base = dict(ctx.pretrained.named_parameters())
    out = {}
    for k in keys:
        w0 = base[k].detach().float().cpu()
        taus = [w * tv[k].float().cpu() for w, tv in zip(weights, ctx.task_vectors)]
        sum_f = sum(f[k] for f in fishers)
        rhs = sum(f[k] * (w0 + tau) for f, tau in zip(fishers, taus))
        x = cg_diag_solve(sum_f, rhs, w0 + torch.stack(taus).mean(0), int(ctx.cfg["num_iterations"]))
        out[k] = x - w0
    out.update(average_others(ctx.task_vectors, keys))
    return out
