import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import commerge.methods  # noqa: F401
from commerge.methods import fisher, knots, mats, ops
from commerge.methods.base import MergeContext
from commerge.registry import METHODS

CPU = torch.device("cpu")


def ctx_of(tvs, cfg, **kw):
    return MergeContext(pretrained=kw.pop("pretrained", nn.Identity()), task_vectors=tvs, cfg=cfg, device=CPU, **kw)


def run(name, tvs, cfg, **kw):
    return METHODS.get(name)(ctx_of(tvs, cfg, **kw))


def vec(*x):
    return {"w": torch.tensor([x], dtype=torch.float32)}


def test_registered_names():
    for n in ["ta", "ties", "dare_ties", "consensus", "lines", "knots_ties", "tsv", "iso_c", "core_tsv", "core_iso",
              "fisher", "mats", "regmean", "com"]:
        METHODS.get(n)


def test_ta_sum_mean_and_per_task_scaling():
    a, b = vec(1., 2.), vec(3., 0.)
    assert torch.allclose(run("ta", [a, b], {"scale": 0.5})["w"], torch.tensor([[2., 1.]]))
    assert torch.allclose(run("ta", [a, b], {"scale": 0.5, "merging_type": "mean"})["w"], torch.tensor([[1., .5]]))
    out = run("ta", [a, b], {"scale": 99.}, scaling=[1.0, 0.0])["w"]
    assert torch.allclose(out, torch.tensor([[1., 2.]]))


def test_ties_hand_computed():
    t1, t2 = vec(4., -1., .5, 2.), vec(-3., 1., .2, 2.)
    out = run("ties", [t1, t2], {"topk": 0.5, "merging_type": "mean", "scale": 1.0})["w"]
    assert torch.allclose(out, torch.tensor([[4., 0., 0., 2.]]))
    out = run("ties", [t1, t2], {"topk": 0.5, "merging_type": "mean", "scale": 0.5})["w"]
    assert torch.allclose(out, torch.tensor([[2., 0., 0., 1.]]))


def test_sign_election_follows_mass_and_disjoint_mean():
    st = torch.tensor([[3.], [-1.], [-1.]])
    assert ops.elect_sign(st).item() == 1.0
    assert torch.allclose(ops.disjoint_mean(st, ops.elect_sign(st)), torch.tensor([3.]))
    st = torch.tensor([[1.], [-2.], [-2.]])
    assert ops.elect_sign(st).item() == -1.0
    assert torch.allclose(ops.disjoint_mean(st, ops.elect_sign(st)), torch.tensor([-2.]))
    st = torch.tensor([[1., 1., 5.], [-1., 1., 5.]])
    assert ops.elect_sign(st).tolist() == [1., 1., 1.]


def test_topk_prune_keeps_fraction_per_task_over_all_params():
    a = {"x": torch.tensor([[10., 1.]]), "y": torch.tensor([[2., 3.]])}
    b = {"x": torch.tensor([[1., 1.]]), "y": torch.tensor([[1., 9.]])}
    out = run("ties", [a, b], {"topk": 0.25, "merging_type": "sum", "scale": 1.0})
    assert torch.allclose(out["x"], torch.tensor([[10., 0.]]))
    assert torch.allclose(out["y"], torch.tensor([[0., 9.]]))


def test_dare_determinism_and_rescale():
    x = torch.ones(2, 1000)
    a = ops.dare_(x.clone(), 0.5, torch.Generator().manual_seed(3))
    b = ops.dare_(x.clone(), 0.5, torch.Generator().manual_seed(3))
    c = ops.dare_(x.clone(), 0.5, torch.Generator().manual_seed(4))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert set(a.unique().tolist()) == {0.0, 2.0}
    tvs = [{"w": torch.randn(4, 8)} for _ in range(3)]
    cfg = {"topk": 1.0, "dare_p": 0.3, "dare_seed": 7, "scale": 0.5}
    assert torch.equal(run("dare_ties", tvs, cfg)["w"], run("dare_ties", tvs, cfg)["w"])
    assert not torch.equal(run("dare_ties", tvs, cfg)["w"], run("dare_ties", tvs, {**cfg, "dare_seed": 8})["w"])


def test_consensus_hand_computed():
    tvs = [vec(1., 4., 1.), vec(1., .1, 1.), vec(1., .1, -.1)]
    out = run("consensus", tvs, {"tall_threshold": 0.5, "consensus_threshold": 2, "scale": 1.0})["w"]
    assert torch.allclose(out, torch.tensor([[3., 0., 1.9]]))
    out = run("consensus", tvs, {"tall_threshold": 0.5, "consensus_threshold": 2, "scale": 0.5})["w"]
    assert torch.allclose(out, torch.tensor([[1.5, 0., .95]]))


def test_lines_layer_index_from_name_not_order():
    names = [f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight" for i in (2, 0, 1)]
    tv = {n: torch.ones(1, 1) for n in names}
    out = run("lines", [tv, copy.deepcopy(tv)], {"scale": 1.0, "alpha": 0.5, "beta": 1.0})
    assert out[names[1]].item() == pytest.approx(2 * 0.5)
    assert out[names[2]].item() == pytest.approx(2 * 1.0)
    assert out[names[0]].item() == pytest.approx(2 * 1.5)
    assert ops.layer_index("model.layers.31.self_attn.o_proj.weight") == 31
    assert ops.layer_index("visual_projection.weight") is None


def _mats(n, o=6, i=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"a": torch.randn(o, i, generator=g), "b": torch.randn(o, i, generator=g)} for _ in range(n)]


def test_knots_tasks_do_not_collapse_and_reconstruct():
    tvs = _mats(3)
    keys = ["a", "b"]
    bases, coeffs = knots.knots_decompose(tvs, keys)
    for k in keys:
        assert not torch.allclose(coeffs[0][k], coeffs[1][k]) and not torch.allclose(coeffs[1][k], coeffs[2][k])
        for t in range(3):
            assert torch.allclose(bases[k] @ coeffs[t][k], tvs[t][k], atol=1e-4)
    assert coeffs[0]["a"].data_ptr() != coeffs[2]["a"].data_ptr()


def test_knots_single_task_and_identical_tasks_return_the_task():
    tv = _mats(1)[0]
    cfg = {"topk": 1.0, "merging_type": "mean", "scale": 1.0}
    out = run("knots_ties", [tv], cfg)
    assert all(torch.allclose(out[k], tv[k], atol=1e-4) for k in tv)
    out = run("knots_ties", [tv, copy.deepcopy(tv)], cfg)
    assert all(torch.allclose(out[k], tv[k], atol=1e-4) for k in tv)


def test_knots_merge_differs_from_last_task():
    tvs = _mats(3)
    cfg = {"topk": 1.0, "merging_type": "sum", "scale": 1.0}
    out = run("knots_ties", tvs, cfg)
    assert not torch.allclose(out["a"], tvs[2]["a"], atol=1e-3)


def test_knots_non_matrix_params_are_averaged():
    tvs = [{"a": torch.randn(4, 4), "bias": torch.full((4,), float(t))} for t in range(3)]
    out = run("knots_ties", tvs, {"topk": 1.0, "merging_type": "mean", "scale": 1.0})
    assert torch.allclose(out["bias"], torch.ones(4))


def test_tsv_shape_and_spectrum():
    n, d = 2, 8
    tvs = _mats(n, d, d, seed=1)
    out = run("tsv", tvs, {"scale": 1.0})
    assert out["a"].shape == (d, d)
    k = d // n
    expected = torch.cat([torch.linalg.svdvals(tv["a"])[:k] for tv in tvs]).sort(descending=True).values
    got = torch.linalg.svdvals(out["a"])
    assert torch.allclose(got, expected, atol=1e-4)
    assert torch.allclose(run("tsv", tvs, {"scale": 0.5})["a"], 0.5 * out["a"], atol=1e-5)


def test_tsv_non_square():
    out = run("tsv", _mats(2, 6, 10, seed=2), {"scale": 1.0})
    assert out["a"].shape == (6, 10)


def test_iso_c_flat_spectrum_non_square():
    tvs = _mats(3, 6, 10, seed=3)
    out = run("iso_c", tvs, {"scale": 1.0})["a"]
    s = torch.linalg.svdvals(out)
    total = sum(tv["a"] for tv in tvs)
    assert out.shape == (6, 10)
    assert torch.allclose(s, torch.linalg.svdvals(total).mean().expand(6), atol=1e-4)
    out2 = run("iso_c", tvs, {"scale": 2.0})["a"]
    assert torch.allclose(out2, 2 * out, atol=1e-4)


def _lora(n, out_dim, in_dim, rank, seed):
    g = torch.Generator().manual_seed(seed)
    return [{"a": torch.randn(out_dim, rank, generator=g, dtype=torch.float64) @ torch.randn(rank, in_dim, generator=g, dtype=torch.float64)}
            for _ in range(n)]


def _sum_subspace_iso(tvs, rank, scale):
    total = sum(scale * tv["a"] for tv in tvs)
    u, _, vh = torch.linalg.svd(total, full_matrices=False)
    u, v = u[:, :rank], vh[:rank].T
    core = sum(scale * (u.T @ tv["a"] @ v) for tv in tvs)
    p, s, qh = torch.linalg.svd(core, full_matrices=False)
    return s.mean() * (u @ (p @ qh) @ v.T)


def test_core_iso_flat_spectrum_in_the_core_space_and_linear_in_scale():
    tvs = _lora(3, 20, 18, 2, seed=4)
    out = run("core_iso", tvs, {"scale": 1.0, "rank": 2})["a"]
    s = torch.linalg.svdvals(out.double())
    assert out.shape == (20, 18) and (s > 1e-4).sum() == 6 and torch.allclose(s[:6], s[0].expand(6), atol=1e-4)
    assert torch.allclose(run("core_iso", tvs, {"scale": 3.0, "rank": 2})["a"], 3 * out, atol=1e-4)


def test_core_iso_equals_isotropic_merging_in_the_sum_subspace_when_the_ranks_fill_it():
    tvs = _lora(6, 40, 36, 4, seed=5)
    core = run("core_iso", tvs, {"scale": 1.0, "rank": 4})["a"].double()
    assert torch.allclose(core, _sum_subspace_iso(tvs, rank=24, scale=1.0), rtol=1e-3, atol=1e-4)


def test_core_tsv_keeps_rank_r_per_task_in_the_core_space():
    tvs = _lora(3, 20, 18, 2, seed=6)
    out = run("core_tsv", tvs, {"scale": 1.0, "rank": 2})["a"]
    assert out.shape == (20, 18) and (torch.linalg.svdvals(out.double()) > 1e-4).sum() == 6
    assert torch.allclose(run("core_tsv", tvs, {"scale": 0.5, "rank": 2})["a"], 0.5 * out, atol=1e-4)


class Lin(nn.Module):
    def __init__(self, d=4, c=3):
        super().__init__()
        self.fc = nn.Linear(d, c, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_diag_fisher_matches_analytic_gradient():
    torch.manual_seed(0)
    model, x, y = Lin(), torch.randn(5, 4), torch.tensor([0, 1, 2, 1, 0])
    fish, n = fisher.diag_fisher(model, [(x, y)], lambda m, b: m(b[0]), ["fc.weight"])
    expected = torch.zeros(3, 4)
    for xi, yi in zip(x, y):
        p = F.softmax(model(xi), -1)
        grad = (F.one_hot(yi, 3).float() - p).unsqueeze(1) * xi.unsqueeze(0)
        expected += (p[yi].sqrt() * grad) ** 2
    assert n == 5 and torch.allclose(fish["fc.weight"], expected, atol=1e-6)
    assert all(p.requires_grad for p in model.parameters())


def _fisher_ctx(tmp_path=None, sizes=(6, 3), seed=0):
    torch.manual_seed(seed)
    pre = Lin()
    tvs = [{"fc.weight": 0.1 * torch.randn(3, 4)} for _ in sizes]
    calib = [[(torch.randn(m, 4), torch.randint(0, 3, (m,)))] for m in sizes]

    def task_model(t):
        m = copy.deepcopy(pre)
        with torch.no_grad():
            m.fc.weight += tvs[t]["fc.weight"]
        return m

    extras = {"logits_fn": lambda t, m, b: m(b[0])}
    if tmp_path:
        extras["cache_dir"] = str(tmp_path)
    return ctx_of(tvs, {"scale": 1.0, "num_iterations": 200}, pretrained=pre, calib=calib,
                  task_model=task_model, extras=extras)


def test_task_fishers_use_finetuned_model_and_own_sample_count(tmp_path):
    ctx = _fisher_ctx(tmp_path)
    fs = fisher.task_fishers(ctx, ["fc.weight"])
    for t in range(2):
        raw, n = fisher.diag_fisher(ctx.task_model(t), ctx.calib[t], lambda m, b: m(b[0]), ["fc.weight"])
        assert torch.allclose(fs[t]["fc.weight"], raw["fc.weight"] / n)
    assert len(list(tmp_path.glob("fisher_*.pt"))) == 2
    ctx.task_model = lambda t: (_ for _ in ()).throw(AssertionError("cache not used"))
    again = fisher.task_fishers(ctx, ["fc.weight"])
    assert torch.equal(again[1]["fc.weight"], fs[1]["fc.weight"])


def test_fisher_weighting_larger_fisher_dominates(monkeypatch):
    ctx = _fisher_ctx()
    big, small = torch.full((3, 4), 1e3), torch.full((3, 4), 1e-3)
    monkeypatch.setattr(fisher, "task_fishers", lambda c, k: [{"fc.weight": big}, {"fc.weight": small}])
    out = METHODS.get("fisher")(ctx)["fc.weight"]
    assert torch.allclose(out, ctx.task_vectors[0]["fc.weight"], atol=1e-4)
    monkeypatch.setattr(fisher, "task_fishers", lambda c, k: [{"fc.weight": big}, {"fc.weight": big}])
    mean = sum(tv["fc.weight"] for tv in ctx.task_vectors) / 2
    assert torch.allclose(METHODS.get("fisher")(ctx)["fc.weight"], mean, atol=1e-6)


def test_mats_cg_matches_closed_form_for_diagonal_fisher(monkeypatch):
    ctx = _fisher_ctx()
    g = torch.Generator().manual_seed(1)
    fs = [{"fc.weight": torch.rand(3, 4, generator=g) + 0.5} for _ in range(2)]
    monkeypatch.setattr(mats, "task_fishers", lambda c, k: fs)
    out = METHODS.get("mats")(ctx)["fc.weight"]
    w0 = ctx.pretrained.fc.weight.detach()
    taus = [tv["fc.weight"] for tv in ctx.task_vectors]
    closed = sum(f["fc.weight"] * (w0 + t) for f, t in zip(fs, taus)) / sum(f["fc.weight"] for f in fs) - w0
    assert torch.allclose(out, closed, atol=1e-4)
    ctx.cfg["num_iterations"] = 0
    avg = torch.stack(taus).mean(0)
    assert torch.allclose(METHODS.get("mats")(ctx)["fc.weight"], avg, atol=1e-6)


def test_fisher_and_mats_end_to_end_on_tiny_model():
    ctx = _fisher_ctx()
    for name in ("fisher", "mats"):
        out = METHODS.get(name)(ctx)
        assert out["fc.weight"].shape == (3, 4) and torch.isfinite(out["fc.weight"]).all()


def test_disjoint_mean_is_chunk_invariant():
    import torch
    from commerge.methods.ops import disjoint_mean, elect_sign
    torch.manual_seed(0)
    stack = torch.randn(4, 1000)
    stack[torch.rand_like(stack) < 0.3] = 0.0
    sign = elect_sign(stack)
    for kind in ("mean", "sum"):
        full = disjoint_mean(stack, sign, [0.5, 1.0, 2.0, 1.5], kind, chunk=10 ** 9)
        assert torch.equal(disjoint_mean(stack, sign, [0.5, 1.0, 2.0, 1.5], kind, chunk=64), full)


def test_diag_fisher_token_inputs_match_per_sample_autograd():
    torch.manual_seed(1)

    class Tok(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 5, bias=True)
            self.head = nn.Linear(5, 3, bias=False)

        def forward(self, x):
            return self.head(torch.tanh(self.fc(x)).mean(1))

    model = Tok()
    x, y = torch.randn(7, 6, 4), torch.randint(0, 3, (7,))
    fish, n = fisher.diag_fisher(model, [(x[:3], y[:3]), (x[3:], y[3:])], lambda m, b: m(b[0]), ["fc.weight", "head.weight"],
                                 micro_batch=2)
    expected = {k: torch.zeros_like(p) for k, p in model.named_parameters() if k.endswith("weight")}
    for xi, yi in zip(x, y):
        model.zero_grad()
        log_p = F.log_softmax(model(xi[None]), -1)[0, yi]
        (log_p.detach().exp().sqrt() * log_p).backward()
        for k, p in model.named_parameters():
            if k in expected:
                expected[k] += p.grad ** 2
    assert n == 7
    for k in expected:
        assert torch.allclose(fish[k], expected[k], atol=1e-6), k


def test_task_fishers_normalisation_modes_and_dataloader_calibration(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    ctx = _fisher_ctx(tmp_path)
    x, y = torch.randn(9, 4), torch.randint(0, 3, (9,))
    ctx.calib = [DataLoader(TensorDataset(x, y), batch_size=4), ctx.calib[1]]
    ctx.cfg = {**ctx.cfg, "fisher_normalize": "none"}
    raw = fisher.task_fishers(ctx, ["fc.weight"])
    ref, n = fisher.diag_fisher(ctx.task_model(0), ctx.calib[0], lambda m, b: m(b[0]), ["fc.weight"])
    assert n == 9 and torch.allclose(raw[0]["fc.weight"], ref["fc.weight"])
    assert (tmp_path / "fisher_raw_model_0_9.pt").exists()
    ctx.cfg = {**ctx.cfg, "fisher_normalize": "per_task"}
    avg = fisher.task_fishers(ctx, ["fc.weight"])
    assert torch.allclose(avg[0]["fc.weight"], raw[0]["fc.weight"] / 9)


def test_ties_merge_cpu_stack_streamed_to_device_matches_direct():
    import torch
    from commerge.methods.ops import ties_merge, stack_flat
    torch.manual_seed(0)
    tvs = [{"a": torch.randn(7, 5), "b": torch.randn(11)} for _ in range(4)]
    stack = stack_flat(tvs, ["a", "b"])
    assert stack.shape == (4, 46) and torch.equal(stack[2, :35], tvs[2]["a"].reshape(-1))
    ref = ties_merge(stack.clone(), [0.5, 1.0, 2.0, 1.5], topk=0.3, dare_p=0.2, seed=3)
    out = ties_merge(stack.clone(), [0.5, 1.0, 2.0, 1.5], topk=0.3, dare_p=0.2, seed=3, device="cpu")
    assert torch.equal(ref, out)


def test_ties_streaming_equals_stack_version_including_tie_breaks():
    import torch
    from commerge.methods.ops import ties_merge, ties_merge_streaming, stack_flat
    torch.manual_seed(1)
    tvs = [{"a": torch.randn(7, 5), "b": torch.randn(11), "c": torch.randn(3, 3)} for _ in range(4)]
    for tv in tvs[:2]:
        tv["b"][:4] = 0.0
    tvs[0]["b"][:2], tvs[1]["b"][:2] = 1.0, -1.0
    keys = ["a", "b", "c"]
    for topk in (0.25, 1.0):
        for kind in ("mean", "sum"):
            ref = ties_merge(stack_flat(tvs, keys), [0.5, 1.0, 2.0, 1.5], topk=topk, merging_type=kind)
            out = ties_merge_streaming(tvs, keys, [0.5, 1.0, 2.0, 1.5], topk=topk, merging_type=kind)
            flat = torch.cat([out[k].reshape(-1) for k in keys])
            assert torch.allclose(flat, ref, atol=1e-6), (topk, kind)


def test_tsv_and_iso_c_scale_sweeps_use_cached_factors_but_match_direct_computation():
    import torch
    from commerge.methods.tsv import tsv_matrix
    from commerge.methods.iso import isotropize
    tvs = _mats(3, 8, 12, seed=5)
    mats = [tv["a"] for tv in tvs]
    for scale in (0.5, 1.2, 0.5):
        tsv_out = run("tsv", tvs, {"scale": scale})["a"]
        assert torch.allclose(tsv_out, tsv_matrix(mats, [scale] * 3).float(), atol=1e-5)
        iso_out = run("iso_c", tvs, {"scale": scale})["a"]
        assert torch.allclose(iso_out, isotropize(scale * sum(m.double() for m in mats)).float(), atol=1e-5)
