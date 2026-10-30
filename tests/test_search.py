from commerge.search import coordinate_search, grid_search, parse_space, _with


def test_parse_space():
    assert parse_space(["method.scale=0.1,0.2", "method.merging_type=sum,mean"]) == {
        "method.scale": [0.1, 0.2], "method.merging_type": ["sum", "mean"]}


def test_grid_search_finds_the_maximum_and_tries_everything():
    calls = []

    def score(p):
        calls.append(p)
        return -(p["a"] - 2) ** 2 - (p["b"] - 30) ** 2

    best, history = grid_search({"a": [1, 2, 3], "b": [10, 30, 50]}, score)
    assert best == {"a": 2, "b": 30} and len(history) == len(calls) == 9


def test_coordinate_search_scans_in_order_and_stops_at_the_first_decrease():
    calls = []

    def score(p):
        calls.append(dict(p))
        return -(p["a"] - 0.3) ** 2 - (p["b"] - 4) ** 2 / 100

    space = {"a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "b": [2, 3, 4, 5, 6]}
    best, history = coordinate_search(space, score, {"a": 0.9, "b": 3})
    assert best == {"a": 0.3, "b": 4}
    assert len(history) == 7
    assert all(c["b"] == 3 for c in calls[:4])
    assert calls[4]["a"] == 0.3


def test_coordinate_search_never_reevaluates_a_setting():
    seen = []

    def score(p):
        seen.append(tuple(sorted(p.items())))
        return float(p["a"])

    coordinate_search({"a": [1, 2], "b": [1, 2]}, score, {"a": 1, "b": 1})
    assert len(seen) == len(set(seen))


def test_with_builds_a_nested_config():
    assert _with({}, {"method.scale": 0.3, "method.topk": 0.2}) == {"method": {"scale": 0.3, "topk": 0.2}}


def test_search_selects_on_val_and_reports_test(monkeypatch):
    import torch.nn as nn
    import commerge.search as search_mod
    from commerge.setup import World

    seen = []

    def fake_evaluate(model, i, split):
        seen.append(split)
        a = model.a
        return 0.5 - 0.01 * abs(a - 2) if split == "val" else 0.6 - 0.01 * abs(a - 3)

    class Toy(nn.Module):
        a = 0

    world = World(pretrained=Toy(), arch=None, task_vectors=[], tasks=["t"], task_data=[], calib=[], evaluate=fake_evaluate,
                  logits_fn=None, heads=[])
    monkeypatch.setattr(search_mod, "build_world", lambda cfg, device: world)

    def fake_merge(cfg, world, device):
        m = Toy()
        m.a = cfg["method"]["a"]
        return m

    import commerge.run as run_mod
    monkeypatch.setattr(run_mod, "merge", fake_merge)
    cfg = {"method": {"name": "toy", "a": 1}, "eval": {"split": "test", "reference_acc": {"t": 50.0}}, "seed": 0,
           "_config_path": "experiments/toy/toy.yaml"}
    import torch
    for mode in ("grid", "coordinate"):
        seen.clear()
        result = search_mod.search(cfg, {"method.a": [1, 2, 3]}, mode, torch.device("cpu"))
        assert result["best_params"] == {"method.a": 2}
        assert seen[-1] == "test" and seen.count("test") == 1 and set(seen[:-1]) == {"val"}
        assert abs(result["test"]["avg/acc"] - 100 * (0.6 - 0.01)) < 1e-9


def test_coordinate_search_returns_the_best_setting_seen_even_after_a_valley():
    def score(p):
        topk = {0.1: 78.0, 0.2: 77.0, 0.3: 79.0, 1.0: 80.5}[p["topk"]]
        return topk - (p["scale"] - 1.0) ** 2

    space = {"scale": [0.5, 1.0, 1.5], "topk": [0.1, 0.2, 0.3, 1.0]}
    best, history = coordinate_search(space, score, {"scale": 0.5, "topk": 1.0})
    assert best == {"scale": 1.0, "topk": 1.0}
    assert max(h["score"] for h in history) == score(best)
