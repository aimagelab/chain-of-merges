import functools
import gc
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from commerge.data.calibration import random_indices
from commerge.run import evaluate_model


def _split(n: int) -> int:
    return min(max(1, int(round(0.8 * n))), n - 1)


class VisionTrainer:
    start_competes = False

    def __init__(self, world, batch_size, seed, device):
        self.world, self.bs, self.seed, self.dev = world, batch_size, seed, device
        self.data = []
        for batches in world.calib:
            x, y = torch.cat([b[0] for b in batches]).half(), torch.cat([b[1] for b in batches])
            k = _split(len(y))
            self.data.append({"x_tr": x[:k], "y_tr": y[:k], "x_es": x[k:], "y_es": y[k:]})
        self.n_train = [len(d["y_tr"]) for d in self.data]

    def logits(self, net, x, i):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.world.logits_fn(i, net, (x.to(self.dev),)).float()

    @torch.no_grad()
    def es_accuracy(self, net):
        net.eval()
        accs = []
        for i, d in enumerate(self.data):
            correct = sum((self.logits(net, d["x_es"][s:s + 50], i).argmax(-1).cpu() == d["y_es"][s:s + 50]).sum().item()
                          for s in range(0, len(d["y_es"]), 50))
            accs.append(correct / len(d["y_es"]))
        net.train()
        return float(np.mean(accs))

    def network(self, start_state):
        net = deepcopy(self.world.pretrained)
        net.load_state_dict(start_state)
        net = net.to(self.dev).float().requires_grad_(True)
        encoder = net.vision_model.encoder
        encoder.gradient_checkpointing = True
        encoder._gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        return net, dict(net.named_parameters())

    def seed_all(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def shuffle(self, n):
        return torch.randperm(n).tolist()

    def step(self, net, opt, scaler, picks):
        for i, (d, idx) in enumerate(zip(self.data, picks)):
            loss = F.cross_entropy(self.logits(net, d["x_tr"][idx], i), d["y_tr"][idx].to(self.dev)) / len(self.data)
            scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def make_scaler(self):
        return torch.amp.GradScaler()


class NLITrainer:
    start_competes = True

    def __init__(self, world, batch_size, seed, device, n):
        self.world, self.bs, self.seed, self.dev = world, batch_size, seed, device
        self.data = []
        for task in world.task_data:
            idx = random_indices(task.labels("train"), n, seed)
            k = _split(len(idx))
            ds = task.split("train")
            es = list(DataLoader(Subset(ds, idx[k:]), batch_size=32, shuffle=False, collate_fn=task.collate_fn))
            self.data.append({"ds": ds, "tr": idx[:k], "es": es, "n_es": len(idx) - k, "task": task})
        self.n_train = [len(d["tr"]) for d in self.data]
        self.rng = random.Random(seed)

    def logits(self, net, i, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.world.logits_fn(i, net, batch)

    @torch.no_grad()
    def es_accuracy(self, net):
        net.eval()
        accs = [sum((self.logits(net, i, b).argmax(-1).cpu() == b["labels"]).sum().item() for b in d["es"]) / d["n_es"]
                for i, d in enumerate(self.data)]
        net.train()
        return float(np.mean(accs))

    def network(self, start_state):
        net = deepcopy(self.world.pretrained)
        net.load_state_dict(start_state)
        net = net.to(self.dev).float().requires_grad_(False)
        trainable = {n: p for n, p in net.named_parameters() if n.startswith("model.layers.") or n.startswith("model.norm.")}
        for p in trainable.values():
            p.requires_grad_(True)
        net.config.use_cache = False
        net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return net, trainable

    def seed_all(self):
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def shuffle(self, n):
        return self.rng.sample(range(n), n)

    def step(self, net, opt, scaler, picks):
        for i, (d, idx) in enumerate(zip(self.data, picks)):
            batch = d["task"].collate_fn([d["ds"][d["tr"][j]] for j in idx])
            loss = F.cross_entropy(self.logits(net, i, batch), batch["labels"].to(self.dev)) / len(self.data)
            loss.backward()
        opt.step()

    def make_scaler(self):
        return None


def train_one_lr(trainer, start_state, lr, proto):
    net, params = trainer.network(start_state)
    opt = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
    scaler = trainer.make_scaler()
    trainer.seed_all()
    net.train()
    step0 = trainer.es_accuracy(net)
    best, best_step, best_state, bad, stopped = (step0 if trainer.start_competes else -1), 0, None, 0, 0
    perms = [trainer.shuffle(n) for n in trainer.n_train]
    ptr = [0] * len(trainer.n_train)
    start = time.time()
    for step in range(1, proto["max_steps"] + 1):
        opt.zero_grad(set_to_none=True)
        picks = []
        for i, n in enumerate(trainer.n_train):
            bs = min(trainer.bs, n)
            if ptr[i] + bs > n:
                perms[i], ptr[i] = trainer.shuffle(n), 0
            picks.append(perms[i][ptr[i]:ptr[i] + bs])
            ptr[i] += bs
        trainer.step(net, opt, scaler, picks)
        if step % proto["eval_every"] == 0:
            stopped = step
            acc = trainer.es_accuracy(net)
            if acc > best:
                best, best_step, bad = acc, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in params.items()}
            else:
                bad += 1
                if bad >= proto["patience"]:
                    break
    del net, opt
    gc.collect()
    torch.cuda.empty_cache()
    if best_state is None:
        best = step0
    return {"lr": lr, "es_best": best, "best_step": best_step, "es_step0": step0, "stopped_at": stopped, "seconds": time.time() - start, "state": best_state}


def finetune(cfg, world, start_model, setting, seed, proto, device):
    start_state = {k: v.detach().cpu().clone() for k, v in start_model.state_dict().items()}
    del start_model
    bs = proto["batch_per_task"][setting]
    if cfg["model"]["family"] == "clip":
        trainer = VisionTrainer(world, bs, seed, device)
    else:
        trainer = NLITrainer(world, bs, seed, device, cfg["data"]["calibration"]["num_samples"])
        world.task_vectors = None
        gc.collect()
    grid, runs = proto["lr_grid"], {}

    def selected():
        return max(runs, key=lambda lr: (runs[lr]["es_best"], lr))

    def run(lr):
        if lr not in runs:
            runs[lr] = train_one_lr(trainer, start_state, lr, proto)
            for k, r in runs.items():
                if k != selected():
                    r["state"] = None

    for lr in proto["lr_start"]:
        run(lr)
    while True:
        i = grid.index(selected())
        todo = [grid[j] for j in (i + 1, i - 1) if 0 <= j < len(grid) and grid[j] not in runs]
        if not todo:
            break
        for lr in todo:
            run(lr)
    best = runs[selected()]
    keep_merged = best["state"] is None or best["es_step0"] >= best["es_best"]
    net = deepcopy(world.pretrained)
    net.load_state_dict(start_state)
    if not keep_merged:
        net.load_state_dict(best["state"], strict=False)
    report = evaluate_model(cfg, world, net.to(device).float(), "test")
    return {"selected_lr": best["lr"], "kept_merged_model": keep_merged, "test": report,
            "runs": [{k: v for k, v in r.items() if k != "state"} for r in runs.values()]}
