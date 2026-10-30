import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from commerge.data.nli import load_nli_task, prepare_tokenizer

BASE = "meta-llama/Llama-3.2-1B"
CKPT = False


@torch.no_grad()
def val_accuracy(model, data, split, device, batch_size=64):
    model.eval()
    correct = total = 0
    for batch in DataLoader(data.split(split), batch_size=batch_size, shuffle=False, collate_fn=data.collate_fn):
        ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        if data.masked_class is not None:
            logits[:, data.masked_class] = float("-inf")
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / total


def train_one_lr(task, data, tok, lr, max_epochs, seed, micro_bs, accum, device, log):
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3, torch_dtype=torch.float32).to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    if CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(data.train, batch_size=micro_bs, shuffle=True, generator=g, collate_fn=data.collate_fn, drop_last=True)
    steps_per_epoch = len(loader) // accum
    total = steps_per_epoch * max_epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    eval_every = max(1, steps_per_epoch // 2)
    best = {"val": -1.0, "step": 0, "state": None}
    bad, step, micro, t0 = 0, 0, 0, time.time()
    stop = False
    for epoch in range(max_epochs):
        for batch in loader:
            ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits.float()
            loss = F.cross_entropy(logits, labels) / accum
            loss.backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                log(f"  lr {lr:g} step {step}/{total} loss {loss.item() * accum:.4f} ({time.time() - t0:.0f}s, {step / (time.time() - t0):.2f} it/s)")
            if step % eval_every == 0 or step == total:
                v = val_accuracy(model, data, "val", device)
                log(f"  lr {lr:g} step {step} (epoch {step / steps_per_epoch:.2f}) val acc {v:.4f}")
                if v > best["val"]:
                    best.update(val=v, step=step, state={k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()})
                    bad = 0
                else:
                    bad += 1
                    if bad >= 2 and step >= steps_per_epoch:
                        log(f"  early stop at step {step}")
                        stop = True
                        break
        if stop:
            break
    return best, step, steps_per_epoch, time.time() - t0, model


def main():
    ap = argparse.ArgumentParser(description="Fully fine-tune Llama-3.2-1B on one NLI task; the learning rate and the stopping point are selected on validation.")
    ap.add_argument("task")
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-6, 1e-5, 2e-5])
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (slower, less memory)")
    ap.add_argument("--train-subset", type=int, default=0, help="debug only: use the first N training examples")
    args = ap.parse_args()
    global CKPT
    CKPT = args.ckpt
    from transformers import AutoTokenizer
    device = torch.device("cuda")
    logf = open(Path(args.out + "_train.log"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(BASE))
    data = load_nli_task(args.task, tok, root=os.environ.get("COMMERGE_DATA", "data"), max_length=args.max_length)
    if args.train_subset:
        data.train = data.train.select(range(args.train_subset))
    accum = 32 // args.micro_bs
    log(f"task {args.task}: {len(data.train)} train / {len(data.val)} val examples; GPU {torch.cuda.get_device_name(0)}; micro-batch {args.micro_bs} x accum {accum} = 32")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"task": args.task, "base": BASE, "lrs": {}, "max_epochs": args.max_epochs, "seed": args.seed, "max_length": args.max_length,
               "gpu": torch.cuda.get_device_name(0), "recipe": "AdamW wd0, cosine, 3% warmup, eff batch 32, bf16 autocast, fp32 master weights, early stop patience 2 (half-epoch evals)"}
    best_overall = {"val": -1.0, "lr": None}
    for lr in args.lrs:
        best, steps, spe, secs, model = train_one_lr(args.task, data, tok, lr, args.max_epochs, args.seed, args.micro_bs, accum, device, log)
        summary["lrs"][str(lr)] = {"best_val": best["val"], "best_step": best["step"], "best_epoch": best["step"] / spe, "steps_run": steps, "epochs_run": steps / spe, "seconds": secs}
        log(f"lr {lr:g}: best val {best['val']:.4f} at epoch {best['step'] / spe:.2f} ({secs / 60:.1f} min)")
        if best["val"] > best_overall["val"]:
            model.load_state_dict(best["state"])
            model.save_pretrained(out, safe_serialization=True)
            best_overall = {"val": best["val"], "lr": lr}
            log(f"  -> new best (lr {lr:g}); checkpoint written to {out}")
        del model, best
        torch.cuda.empty_cache()
    summary.update(selected_lr=best_overall["lr"], selected_val_acc=best_overall["val"])
    json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
    log(f"DONE {args.task}: selected lr {best_overall['lr']:g}, val acc {best_overall['val']:.4f}")


if __name__ == "__main__":
    main()
