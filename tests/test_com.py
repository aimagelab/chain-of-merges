import copy

import pytest
import torch
import torch.nn as nn

from commerge.methods.base import MergeContext
from commerge.methods.com import (
    LayerwiseConfig,
    LayerwiseMerger,
    capture_stage_input,
    solve_layer,
)
from commerge.models.arch import ArchAdapter, BlockState
from commerge.models.task_vectors import apply_task_vector

D, T, N_TASKS = 8, 5, 3


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D, bias=False) for _ in range(4))

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        att = torch.softmax(q @ k.transpose(1, 2) / D ** 0.5, dim=-1)
        return self.out_proj(att @ v)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.self_attn = Attn()
        self.mlp = nn.Sequential()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(D, 2 * D, bias=False), nn.Linear(2 * D, D, bias=False)

    def forward(self, x):
        x = x + self.self_attn(self.ln(x))
        return x + self.mlp.fc2(torch.relu(self.mlp.fc1(x)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])
        self.emb = nn.Linear(D, D, bias=False)


class ToyArch(ArchAdapter):
    stages = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.out_proj"],
              ["mlp.fc1"], ["mlp.fc2"]]

    def blocks(self, model):
        return model.layers

    def block_prefix(self, i):
        return f"layers.{i}."

    def prepare(self, model, batch):
        return BlockState(hidden=model.emb(batch))

    def run_block(self, block, state):
        return block(state.hidden)

    def forward_batch(self, model, batch):
        h = model.emb(batch)
        for blk in model.layers:
            h = blk(h)


def make_world(seed=0, full_ft=True):
    torch.manual_seed(seed)
    pre = Toy().requires_grad_(False)
    tvs = []
    for _ in range(N_TASKS):
        tv = {}
        for name, p in pre.named_parameters():
            if p.ndim == 2 and (full_ft or "self_attn" in name) and name != "emb.weight":
                tv[name] = 0.3 * torch.randn_like(p)
        tvs.append(tv)
    calib = [[torch.randn(6, T, D) for _ in range(2)] for _ in range(N_TASKS)]
    return pre, tvs, calib


def ctx_for(pre, tvs, calib, **cfg):
    return MergeContext(
        pretrained=pre, task_vectors=tvs, cfg=cfg, device=torch.device("cpu"), arch=ToyArch(), calib=calib,
        task_model=lambda i: apply_task_vector(copy.deepcopy(pre), tvs[i]))


def test_solve_layer_matches_least_squares():
    torch.manual_seed(0)
    xs = [torch.randn(50, D) for _ in range(3)]
    deltas = [torch.randn(D, D) for _ in range(3)]
    grams = [x.T @ x for x in xs]
    cfg = LayerwiseConfig(alpha=1.0, task_weights="none", normalize_by_tokens=False, dtype="float32")
    merged_delta = solve_layer(deltas, grams, [50] * 3, cfg)
    expected = sum(d @ g for d, g in zip(deltas, grams)) @ torch.linalg.inv(sum(grams))
    assert torch.allclose(merged_delta, expected, atol=1e-4)
    grad = sum((merged_delta - d) @ g for d, g in zip(deltas, grams))
    assert grad.abs().max() < 1e-3


def test_alpha_shrinks_off_diagonals_and_scale_multiplies():
    torch.manual_seed(1)
    g = [torch.randn(20, D).T @ torch.randn(20, D) for _ in range(2)]
    g = [x @ x.T + torch.eye(D) for x in g]
    deltas = [torch.randn(D, D) for _ in range(2)]
    base = LayerwiseConfig(alpha=0.0, task_weights="none", normalize_by_tokens=False)
    diag = solve_layer(deltas, g, [1, 1], base)
    expected = sum(d @ torch.diag(torch.diag(x)) for d, x in zip(deltas, g)) @ torch.linalg.pinv(
        sum(torch.diag(torch.diag(x)) for x in g))
    assert torch.allclose(diag, expected, atol=1e-5)
    twice = solve_layer(deltas, g, [1, 1], LayerwiseConfig(alpha=0.0, scale=2.0, task_weights="none",
                                                           normalize_by_tokens=False))
    assert torch.allclose(twice, 2 * diag, atol=1e-5)


def test_task_weights_min_max():
    eye = torch.eye(D)
    grams = [eye * 10 + 0.0, eye * 10 + 1.0 * (1 - eye), eye * 10 + 5.0 * (1 - eye)]
    deltas = [torch.randn(D, D) for _ in range(3)]
    cfg = LayerwiseConfig(alpha=1.0, task_weights="gram_offdiag", normalize_by_tokens=False)
    out = solve_layer(deltas, grams, [1, 1, 1], cfg)
    r = torch.tensor([0.0, 1.0, 5.0])
    w = (r - r.min() + 0.1) / (r.max() - r.min() + 0.1)
    expected = sum(wi * d @ g for wi, d, g in zip(w, deltas, grams)) @ torch.linalg.pinv(
        sum(wi * g for wi, g in zip(w, grams)))
    assert torch.allclose(out, expected, atol=1e-4)


def test_capture_stage_input_matches_full_forward():
    pre, _, calib = make_world()
    arch, block = ToyArch(), pre.layers[0]
    state = arch.prepare(pre, calib[0][0])
    x = state.hidden
    assert torch.allclose(capture_stage_input(arch, block, "self_attn.q_proj", state), block.ln(x))
    z_in = capture_stage_input(arch, block, "self_attn.out_proj", state)
    a = block.ln(x)
    q, k, v = (getattr(block.self_attn, n)(a) for n in ("q_proj", "k_proj", "v_proj"))
    assert torch.allclose(z_in, torch.softmax(q @ k.transpose(1, 2) / D ** 0.5, -1) @ v, atol=1e-6)
    h1 = x + block.self_attn(a)
    assert torch.allclose(capture_stage_input(arch, block, "mlp.fc2", state), torch.relu(block.mlp.fc1(h1)), atol=1e-6)


def test_first_stage_identical_for_chain_and_regmean_but_later_stages_differ():
    pre, tvs, calib = make_world()
    kw = dict(alpha=1.0, task_weights="none", normalize_by_tokens=True, normalize_features=True)
    chain = LayerwiseMerger(ctx_for(pre, tvs, calib, chain=True, **kw)).run()
    plain = LayerwiseMerger(ctx_for(pre, tvs, calib, chain=False, **kw)).run()
    for name in ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight"):
        assert torch.allclose(chain[name], plain[name], atol=1e-5), name
    for name in ("layers.0.self_attn.out_proj.weight", "layers.1.self_attn.q_proj.weight"):
        assert not torch.allclose(chain[name], plain[name], atol=1e-4), name


def test_chain_uses_merged_activations_for_later_layers():
    pre, tvs, calib = make_world()
    kw = dict(alpha=1.0, task_weights="none", normalize_by_tokens=True, normalize_features=True, chain=True)
    delta = LayerwiseMerger(ctx_for(pre, tvs, calib, **kw)).run()

    merged = copy.deepcopy(pre)
    params = dict(merged.named_parameters())
    for k in delta:
        if k.startswith("layers.0."):
            params[k].add_(delta[k])
    arch, cfg = ToyArch(), LayerwiseConfig(alpha=1.0, task_weights="none")
    grams, counts = [], []
    for t in range(N_TASKS):
        g, n = torch.zeros(D, D), 0
        for batch in calib[t]:
            state = arch.prepare(merged, batch)
            state.hidden = arch.run_block(merged.layers[0], state)
            x = capture_stage_input(arch, merged.layers[1], "self_attn.q_proj", state).reshape(-1, D)
            x = x / x.norm(dim=-1, keepdim=True)
            g += x.T @ x
            n += x.shape[0]
        grams.append(g)
        counts.append(n)
    expected = solve_layer([tvs[t]["layers.1.self_attn.q_proj.weight"] for t in range(N_TASKS)], grams, counts, cfg)
    assert torch.allclose(delta["layers.1.self_attn.q_proj.weight"] - sum(
        tv["layers.1.self_attn.q_proj.weight"] for tv in tvs) / N_TASKS * 0, expected, atol=1e-4)


def test_lora_style_attention_only_leaves_mlp_untouched_and_averages_nothing_else():
    pre, tvs, calib = make_world(full_ft=False)
    out = LayerwiseMerger(ctx_for(pre, tvs, calib, alpha=1.0, chain=True, task_weights="none")).run()
    assert set(out) == set(tvs[0])
    assert all("mlp" not in k for k in out)


def test_padding_tokens_are_excluded_from_grams():
    from commerge.methods.com import features
    cfg = LayerwiseConfig()
    x = torch.randn(2, 4, D)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    feats = features(x, mask, cfg)
    assert feats.shape == (5, D)
    x2 = x.clone()
    x2[~mask] = 123.0
    assert torch.equal(features(x2, mask, cfg), feats)


def test_single_task_recovers_that_task():
    pre, tvs, calib = make_world()
    one = LayerwiseMerger(MergeContext(
        pretrained=pre, task_vectors=tvs[:1], cfg=dict(alpha=1.0, chain=True, task_weights="none", pinv_rtol=1e-5),
        device=torch.device("cpu"), arch=ToyArch(), calib=calib[:1],
        task_model=lambda i: apply_task_vector(copy.deepcopy(pre), tvs[i]))).run()
    for name in ("layers.0.mlp.fc1.weight", "layers.1.mlp.fc1.weight", "layers.1.mlp.fc2.weight"):
        assert torch.allclose(one[name], tvs[0][name], atol=1e-3), name
    not_ones = torch.eye(D) - torch.ones(D, D) / D
    q = "layers.0.self_attn.q_proj.weight"
    assert torch.allclose(one[q] @ not_ones, tvs[0][q] @ not_ones, atol=1e-3)


class ConvToy(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, D, kernel_size=2, stride=2, bias=False)
        self.layers = nn.ModuleList([Block()])


class ConvToyArch(ToyArch):
    def stem(self, model):
        from commerge.models.arch import StemSpec
        return StemSpec("patch.weight", 2, 1)

    def stem_inputs(self, batch):
        return torch.nn.functional.unfold(batch, kernel_size=2, stride=2).transpose(1, 2)

    def prepare(self, model, batch):
        return BlockState(hidden=model.patch(batch).flatten(2).transpose(1, 2))


def test_conv_stem_is_merged_with_unfolded_patches():
    torch.manual_seed(0)
    pre = ConvToy().requires_grad_(False)
    tvs = [{"patch.weight": 0.3 * torch.randn_like(pre.patch.weight)} for _ in range(N_TASKS)]
    calib = [[torch.randn(6, 1, 8, 8)] for _ in range(N_TASKS)]
    ctx = MergeContext(pretrained=pre, task_vectors=tvs, cfg=dict(alpha=1.0, task_weights="none"),
                       device=torch.device("cpu"), arch=ConvToyArch(), calib=calib)
    delta = LayerwiseMerger(ctx).run()
    assert delta["patch.weight"].shape == pre.patch.weight.shape
    grams, counts = [], []
    for t in range(N_TASKS):
        x = torch.nn.functional.unfold(calib[t][0], 2, stride=2).transpose(1, 2).reshape(-1, 4)
        x = x / x.norm(dim=-1, keepdim=True)
        grams.append(x.T @ x)
        counts.append(x.shape[0])
    expected = solve_layer([tv["patch.weight"].flatten(1) for tv in tvs], grams, counts,
                           LayerwiseConfig(alpha=1.0, task_weights="none"))
    assert torch.allclose(delta["patch.weight"].flatten(1), expected, atol=1e-5)


def test_streaming_regmean_matches_bruteforce_and_accepts_iterables():
    pre, tvs, calib = make_world()
    kw = dict(alpha=0.9, task_weights="none", normalize_by_tokens=True, normalize_features=True, chain=False, pinv_rtol=1e-6)
    out = LayerwiseMerger(ctx_for(pre, tvs, calib, **kw)).run()
    out_iter = LayerwiseMerger(ctx_for(pre, tvs, [iter(c) for c in calib], **kw)).run()
    for k in out:
        assert torch.allclose(out[k], out_iter[k], atol=1e-6), k

    arch, cfg = ToyArch(), LayerwiseConfig(alpha=0.9, task_weights="none", pinv_rtol=1e-6)
    for name in ("layers.0.self_attn.k_proj.weight", "layers.1.self_attn.out_proj.weight", "layers.1.mlp.fc2.weight"):
        module = name.rsplit(".", 1)[0]
        stage_first = {"k_proj": "q_proj", "out_proj": "out_proj", "fc2": "fc2"}[module.split(".")[-1]]
        first = module.rsplit(".", 1)[0] + "." + stage_first
        grams, counts = [], []
        for t in range(N_TASKS):
            model_t = apply_task_vector(copy.deepcopy(pre), tvs[t])
            rows = []
            handle = model_t.get_submodule(first).register_forward_pre_hook(lambda _, i: rows.append(i[0].detach()))
            for batch in calib[t]:
                arch.forward_batch(model_t, batch)
            handle.remove()
            x = torch.cat([r.reshape(-1, r.shape[-1]) for r in rows])
            x = x / x.norm(dim=-1, keepdim=True)
            grams.append(x.T @ x)
            counts.append(x.shape[0])
        expected = solve_layer([tv[name] for tv in tvs], grams, counts, cfg)
        assert torch.allclose(out[name], expected, atol=1e-4), name


def test_chain_rejects_unmaterialised_calibration():
    pre, tvs, calib = make_world()
    with pytest.raises(ValueError):
        LayerwiseMerger(ctx_for(pre, tvs, [iter(c) for c in calib], chain=True)).run()


class TailToy(Toy):

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D, 3, bias=False)


class TailArch(ToyArch):
    def tail(self, model):
        from commerge.models.arch import TailSpec
        return TailSpec("head.weight", "head")

    def tail_inputs(self, model, hidden):
        return hidden.mean(1)

    def forward_batch(self, model, batch):
        h = model.emb(batch)
        for blk in model.layers:
            h = blk(h)
        model.head(h.mean(1))


def _tail_world(seed=0):
    torch.manual_seed(seed)
    pre = TailToy().requires_grad_(False)
    tvs = [{"head.weight": 0.3 * torch.randn(3, D), "layers.0.mlp.fc1.weight": 0.3 * torch.randn(2 * D, D),
            "layers.1.mlp.fc2.weight": 0.3 * torch.randn(D, 2 * D)} for _ in range(N_TASKS)]
    calib = [[torch.randn(6, T, D) for _ in range(2)] for _ in range(N_TASKS)]
    return pre, tvs, calib


def test_tail_layer_is_solved_from_pooled_features_of_the_merged_network():
    pre, tvs, calib = _tail_world()
    arch = TailArch()
    kw = dict(alpha=1.0, task_weights="none", normalize_by_tokens=True, normalize_features=True, pinv_rtol=1e-6)
    out = LayerwiseMerger(MergeContext(pretrained=pre, task_vectors=tvs, cfg=dict(chain=True, **kw),
                                       device=torch.device("cpu"), arch=arch, calib=calib)).run()
    assert "head.weight" in out
    merged = copy.deepcopy(pre)
    params = dict(merged.named_parameters())
    for k in out:
        if k != "head.weight":
            params[k].add_(out[k])
    grams, counts = [], []
    for t in range(N_TASKS):
        rows = []
        for batch in calib[t]:
            h = merged.emb(batch)
            for blk in merged.layers:
                h = blk(h)
            rows.append(h.mean(1))
        x = torch.cat(rows)
        x = x / x.norm(dim=-1, keepdim=True)
        grams.append(x.T @ x)
        counts.append(x.shape[0])
    expected = solve_layer([tv["head.weight"] for tv in tvs], grams, counts, LayerwiseConfig(**kw))
    assert torch.allclose(out["head.weight"], expected, atol=1e-4)


def test_tail_layer_streaming_regmean_matches_hooked_forward():
    pre, tvs, calib = _tail_world(1)
    arch = TailArch()
    kw = dict(alpha=0.9, task_weights="none", normalize_by_tokens=True, normalize_features=True, pinv_rtol=1e-6)
    out = LayerwiseMerger(MergeContext(pretrained=pre, task_vectors=tvs, cfg=dict(chain=False, **kw),
                                       device=torch.device("cpu"), arch=arch, calib=calib,
                                       task_model=lambda i: apply_task_vector(copy.deepcopy(pre), tvs[i]))).run()
    grams, counts = [], []
    for t in range(N_TASKS):
        model_t = apply_task_vector(copy.deepcopy(pre), tvs[t])
        rows = []
        handle = model_t.head.register_forward_pre_hook(lambda _, i: rows.append(i[0].detach()))
        for batch in calib[t]:
            arch.forward_batch(model_t, batch)
        handle.remove()
        x = torch.cat(rows)
        x = x / x.norm(dim=-1, keepdim=True)
        grams.append(x.T @ x)
        counts.append(x.shape[0])
    expected = solve_layer([tv["head.weight"] for tv in tvs], grams, counts, LayerwiseConfig(**kw))
    assert torch.allclose(out["head.weight"], expected, atol=1e-4)


def test_regmean_with_grams_spilled_to_disk_is_identical(tmp_path):
    pre, tvs, calib = make_world()
    kw = dict(alpha=0.9, task_weights="none", normalize_by_tokens=True, normalize_features=True, chain=False, pinv_rtol=1e-6)
    ref = LayerwiseMerger(ctx_for(pre, tvs, calib, **kw)).run()
    out = LayerwiseMerger(ctx_for(pre, tvs, calib, gram_spill_dir=str(tmp_path), **kw)).run()
    for k in ref:
        assert torch.equal(ref[k], out[k]), k
    assert list(tmp_path.iterdir()) == []


def test_regmean_gram_cache_is_identical_and_reused_across_alpha(tmp_path):
    pre, tvs, calib = make_world()
    kw = dict(task_weights="none", normalize_by_tokens=True, normalize_features=True, chain=False, pinv_rtol=1e-6)
    extras = {"cache_dir": str(tmp_path), "model_id": "toy", "calibration": {"num_samples": 10, "seed": 0},
              "task_names": [f"t{i}" for i in range(len(tvs))], "checkpoints": ["toy"]}

    def cached(calib_, alpha):
        ctx = ctx_for(pre, tvs, calib_, gram_cache=True, alpha=alpha, **kw)
        ctx.extras = extras
        return LayerwiseMerger(ctx).run()

    first = cached(calib, 0.9)
    ref = LayerwiseMerger(ctx_for(pre, tvs, calib, alpha=0.9, **kw)).run()
    for k in ref:
        assert torch.equal(ref[k], first[k]), k

    class Boom:
        def __iter__(self):
            raise AssertionError("calibration data was streamed again although the Grams are cached")

    second = cached([Boom() for _ in calib], 0.7)
    ref7 = LayerwiseMerger(ctx_for(pre, tvs, calib, alpha=0.7, **kw)).run()
    for k in ref7:
        assert torch.equal(ref7[k], second[k]), k
    assert len(list((tmp_path / "grams").iterdir())) == 1
