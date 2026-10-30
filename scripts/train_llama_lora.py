import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from commerge.data.nli import load_nli_task, prepare_tokenizer


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a rank-16 LoRA adapter (q, k, v, o projections + score head) of a Llama model on one NLI task.")
    ap.add_argument("base")
    ap.add_argument("task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=6000, help="random training subset size")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (faster; fp32 master weights)")
    ap.add_argument("--data-root", default=os.environ.get("COMMERGE_DATA", "data"))
    args = ap.parse_args(argv)

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(args.base))
    data = load_nli_task(args.task, tok, root=args.data_root, max_length=args.max_length)
    train = data.train.shuffle(seed=args.seed).select(range(min(args.samples, len(data.train))))
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=3, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=data.collate_fn)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start, step = time.time(), 0
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 25 == 0 or step == total:
                print(f"step {step}/{total} loss {loss.item():.4f} ({time.time() - start:.0f}s)", flush=True)

    model.eval()
    val = data.val.select(range(min(1000, len(data.val))))
    correct = 0
    with torch.no_grad():
        for batch in DataLoader(val, batch_size=32, collate_fn=data.collate_fn):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            if data.masked_class is not None:
                logits[:, data.masked_class] = float("-inf")
            correct += (logits.argmax(-1) == batch["labels"]).sum().item()
    acc = correct / len(val)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    json.dump({"base": args.base, "task": args.task, "val_acc": acc, "samples": len(train), "epochs": args.epochs, "lr": args.lr,
               "seconds": time.time() - start}, open(Path(args.out) / "train_info.json", "w"), indent=1)
    print(f"{args.task}: val acc {acc:.4f}; adapter saved to {args.out}")


if __name__ == "__main__":
    main()
