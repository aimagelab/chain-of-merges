from commerge.config import apply_override, deep_merge, load_config


def test_deep_merge_recurses_and_replaces_leaves():
    out = deep_merge({"a": {"b": 1, "c": 2}, "d": [1]}, {"a": {"b": 3}, "d": [2]})
    assert out == {"a": {"b": 3, "c": 2}, "d": [2]}


def test_overrides_parse_types():
    cfg = apply_override({"method": {"alpha": 1.0}}, "method.alpha=0.95")
    assert cfg["method"]["alpha"] == 0.95
    assert apply_override(cfg, "method.pinv_rtol=1e-6")["method"]["pinv_rtol"] == 1e-6
    assert apply_override(cfg, "method.chain=false")["method"]["chain"] is False
    assert apply_override(cfg, "data.calibration.num_samples=5")["data"]["calibration"]["num_samples"] == 5
    assert apply_override(cfg, "method.name=zeroshot")["method"]["name"] == "zeroshot"


def test_defaults_are_composed_and_experiment_wins():
    cfg = load_config("experiments/vit_b32_full/com.yaml")
    assert cfg["method"]["name"] == "com" and cfg["method"]["scale"] == 1.25
    assert cfg["model"]["finetune"] == "openclip" and len(cfg["model"]["checkpoints"]) == 8
    assert cfg["method"]["normalize_features"] is True


def test_every_experiment_config_loads_and_names_a_registered_method():
    import glob
    import commerge.methods  # noqa: F401
    from commerge.config import CONFIG_DIR
    from commerge.registry import METHODS
    files = sorted(glob.glob(str(CONFIG_DIR / "experiments" / "*" / "*.yaml")))
    assert len(files) > 60
    for f in files:
        cfg = load_config(f)
        METHODS.get(cfg["method"]["name"])
        for key in (cfg.get("search") or {}).get("params", {}):
            assert key.split(".", 1)[1] in cfg["method"], (f, key)
