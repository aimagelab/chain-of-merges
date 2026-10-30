import json

import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast

from commerge.data import nli
from dataclasses import replace

from commerge.eval.nli import evaluate_nli, nli_logits
from commerge.models import llama
from commerge.models.arch import BlockState


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=100, max_position_embeddings=64, num_labels=3,
                      pad_token_id=99, eos_token_id=99, use_cache=False)
    return LlamaForSequenceClassification(cfg).eval().float()


def tiny_tokenizer():
    words = ["a", "b", "c", "d", "e", "f", "g", "h"]
    vocab = {"<unk>": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>")


def _padded_batch(side):
    torch.manual_seed(1)
    lens = [7, 4, 5, 2]
    T = max(lens)
    ids = torch.full((4, T), 99)
    mask = torch.zeros(4, T, dtype=torch.long)
    for i, n in enumerate(lens):
        sl = slice(0, n) if side == "right" else slice(T - n, T)
        ids[i, sl] = torch.randint(0, 90, (n,))
        mask[i, sl] = 1
    return {"input_ids": ids, "attention_mask": mask}


@pytest.mark.parametrize("side", ["right", "left"])
def test_llama_arch_matches_hf(side):
    model = tiny_model()
    batch = _padded_batch(side)
    ref = model.model(**batch).last_hidden_state
    arch = llama.LlamaArch()
    state = arch.prepare(model, batch)
    assert state.mask.dtype == torch.bool and state.mask.equal(batch["attention_mask"].bool())
    h = state.hidden
    for i, block in enumerate(arch.blocks(model)):
        h = arch.run_block(block, BlockState(hidden=h, kwargs=state.kwargs, mask=state.mask))
    out = model.model.norm(h)
    assert torch.equal(out, ref)
    assert arch.block_prefix(1) == "model.layers.1."
    assert dict(model.named_parameters())[arch.block_prefix(1) + "self_attn.q_proj.weight"] is not None


def _raw(task, n=40):
    k1, k2 = nli.TEXT_KEYS[task]
    def split(labels):
        return Dataset.from_dict({k1: ["a b c"] * len(labels), k2: ["d e"] * len(labels), "label": labels})
    if task == "scitail":
        labs = ["entails", "neutral"] * (n // 2)
        return DatasetDict(train=split(labs), validation=split(labs), test=split(labs))
    labs = [0, 1, 2, -1] * (n // 4)
    return DatasetDict(train=split(labs), validation=split(labs), test=split(labs))


def test_remap_labels():
    r = nli.remap_labels(_raw("qnli")["train"], "qnli")
    assert sorted(set(r["label"])) == [0, 2]
    assert len(r) == 30 and 1 not in r["label"] and -1 not in r["label"]
    r = nli.remap_labels(_raw("snli")["train"], "snli")
    assert sorted(set(r["label"])) == [0, 1, 2] and len(r) == 30
    r = nli.remap_labels(_raw("scitail")["train"], "scitail")
    assert sorted(set(r["label"])) == [0, 1]


def test_split_rule():
    raw = _raw("mnli", 80)
    sp = nli.make_splits(raw, "mnli")
    filtered = nli.remap_labels(raw["validation"], "mnli")
    n = len(filtered)
    assert len(sp["test"]) + len(sp["val"]) == n
    ref = filtered.train_test_split(test_size=0.2, shuffle=True, seed=42)
    assert sp["test"].to_dict() == ref["train"].to_dict() and sp["val"].to_dict() == ref["test"].to_dict()
    assert len(sp["train"]) == 60
    sp = nli.make_splits(_raw("snli"), "snli")
    assert len(sp["test"]) == len(sp["val"]) == len(sp["train"]) == 30
    assert len(nli.make_splits(_raw("rte", 80), "rte")["val"]) == 30


def test_build_task_data():
    tok = tiny_tokenizer()
    td = nli.build_task_data(_raw("scitail"), "scitail", tok)
    assert td.masked_class == 2 and nli.MASKED_CLASS["qnli"] == 1 and nli.MASKED_CLASS["snli"] is None
    assert td.labels("test") == [0, 1] * 20 and td.num_classes == 3
    batch = td.collate_fn([td.train[0], td.train[1]])
    assert set(batch) >= {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 2 and batch["labels"].tolist() == [0, 1]
    assert tok.pad_token_id == tok.eos_token_id and tok.padding_side == "right"


def test_masked_class_eval():
    model = tiny_model()
    tok = tiny_tokenizer()
    td = nli.build_task_data(_raw("snli", 40), "snli", tok)
    torch.manual_seed(3)
    head = torch.randn(3, 32)
    with torch.no_grad():
        model.score.weight.copy_(head)
    acc = evaluate_nli(model, head, td, batch_size=7, device="cpu")
    from torch.utils.data import DataLoader
    ok = n = 0
    for b in DataLoader(td.test, batch_size=7, collate_fn=td.collate_fn):
        ok += (model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.argmax(-1) == b["labels"]).sum().item()
        n += len(b["labels"])
    assert acc == pytest.approx(ok / n)
    bias_head = torch.zeros(3, 32)
    model2 = tiny_model()
    model2.model.norm.weight.data.fill_(0)
    assert evaluate_nli(model2, bias_head, td, 8, "cpu") == pytest.approx(
        sum(l == 0 for l in td.labels("test")) / len(td.labels("test")))
    masked = replace(td, masked_class=0)
    assert evaluate_nli(model2, bias_head, masked, 8, "cpu") == pytest.approx(
        sum(l == 1 for l in td.labels("test")) / len(td.labels("test")))
    batch = next(iter(DataLoader(masked.test, batch_size=4, collate_fn=masked.collate_fn)))
    logits = nli_logits(model2, bias_head, masked, batch, "cpu")
    assert torch.isinf(logits[:, 0]).all()


def test_task_vectors_from_local_adapter(tmp_path):
    torch.manual_seed(0)
    r, alpha = 4, 8
    state, expect = {}, {}
    for layer in range(2):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            A, B = torch.randn(r, 32), torch.randn(16 if proj in ("k_proj", "v_proj") else 32, r)
            state[f"{base}.lora_A.weight"], state[f"{base}.lora_B.weight"] = A, B
            expect[f"model.layers.{layer}.self_attn.{proj}.weight"] = B @ A * alpha / r
    head = torch.randn(3, 32)
    state["base_model.model.score.modules_to_save.default.weight"] = head
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))
    torch.save(state, d / "adapter_model.bin")
    cfg = {"finetune": "lora", "checkpoints": [str(d), str(d)]}
    tvs = llama.load_task_vectors(cfg, None)
    assert len(tvs) == 2 and set(tvs[0]) == set(expect)
    for k, v in expect.items():
        assert torch.allclose(tvs[0][k], v, atol=1e-5)
    heads = llama.load_task_heads(cfg)
    assert torch.equal(heads[1], head)
    from commerge.models.task_vectors import apply_task_vector
    apply_task_vector(tiny_model(), tvs[0])


def test_nli_finetuning_runs_and_keeps_the_merged_model_when_nothing_improves():
    from commerge.finetune import finetune
    from commerge.setup import World
    tok = tiny_tokenizer()
    tasks = [nli.build_task_data(_raw("snli", 40), "snli", tok), nli.build_task_data(_raw("snli", 40), "snli", tok)]
    heads = [torch.randn(3, 32), torch.randn(3, 32)]
    model = tiny_model()
    world = World(pretrained=model, arch=None, task_vectors=[], tasks=["a", "b"], task_data=tasks, calib=[],
                  evaluate=lambda m, i, split: evaluate_nli(m, heads[i], tasks[i], 8, "cpu", split),
                  logits_fn=lambda i, m, b: nli_logits(m, heads[i], tasks[i], b, "cpu"), heads=heads)
    cfg = {"model": {"family": "llama"}, "data": {"calibration": {"num_samples": 10}},
           "eval": {"split": "test", "reference_acc": {"a": 50.0, "b": 50.0}}, "_config_path": "experiments/tiny/x.yaml"}
    proto = {"lr_grid": [0.0, 1e-3], "lr_start": [0.0], "max_steps": 4, "eval_every": 2, "patience": 1, "batch_per_task": {"tiny": 2}}
    out = finetune(cfg, world, tiny_model(), "tiny", 0, proto, torch.device("cpu"))
    assert [r["lr"] for r in out["runs"]] == [0.0, 1e-3]
    zero = next(r for r in out["runs"] if r["lr"] == 0.0)
    assert zero["es_best"] == zero["es_step0"]
    if out["selected_lr"] == 0.0:
        assert out["kept_merged_model"]
