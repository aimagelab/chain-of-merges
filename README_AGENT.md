# Operating manual for coding agents

## Layout

```
commerge/
  run.py          python -m commerge.run <experiment.yaml> [key.sub=value ...]       merge + evaluate, writes outputs/runs/...
  search.py       python -m commerge.search <experiment.yaml> [key=v1,v2 ...]        select on val, report test once
  matched.py      python -m commerge.matched {merge,finetune} <setting>              Tab. 8 (same n examples for every method)
  finetune.py     backbone fine-tuning used by matched.py (vision and NLI trainers)
  prepare.py      python -m commerge.prepare --vision/--nli/--models [--gated]/--llama1b-adapters/--synthetic
  prepare_vision.py   rebuilds the 8 vision datasets from public sources, verified against data/assets/manifests
                      (SVHN / GTSRB fall back to pinned Hugging Face copies; GTSRB test labels are bundled in data/assets)
  config.py       YAML loading: `defaults: [models/x, methods/y]` merged in order, then the file, then CLI overrides
  setup.py        build_world(cfg) -> World(pretrained, arch, task_vectors, tasks, task_data, calib, evaluate, logits_fn, heads)
  data/           vision.py (splits), nli.py (splits, label remaps), calibration.py (sampling), synthetic.py
  models/         clip.py, openclip.py (Ilharco checkpoints), llama.py (LoRA + full fine-tuning), arch.py (ArchAdapter)
  methods/        one file per family; every method is `merge(ctx: MergeContext) -> TaskVector` registered in METHODS
  eval/           vision.py, nli.py (last-token pooling through the task head), metrics.py (normalised accuracy)
configs/
  models/ methods/ experiments/<setting>/<method>.yaml   paper hyper-parameters + the `search:` grid they came from
  matched/{vision,llama}.yaml                            Tab. 8 grids and fine-tuning recipe
  synthetic.yaml                                         Tab. 7 grid points
scripts/          one script per table/figure (see README.md), tables.py collects outputs/ into outputs/RESULTS.md
tests/            CPU unit tests (tiny models), no downloads
```

## Commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q        # ~2 s, 13 tests skip without cached tiny HF models
python -m pyflakes commerge scripts
python -m commerge.run experiments/vit_b32_lora/com.yaml data.calibration.num_samples=50     # smoke run, ~2 min on an 11 GB GPU
```

## Methods and their keys

| name | file | keys |
|---|---|---|
| `com`, `regmean` | methods/com.py | `alpha` (off-diagonal Gram shrinkage), `scale`, `chain` (true = CoM), `task_weights` (`gram_offdiag` / `none`), `normalize_features`, `normalize_by_tokens`, `pinv_rtol`, `gram_cache`, `gram_spill_dir` |
| `ta` | task_arithmetic.py | `scale`, `merging_type` (`sum` / `mean`) |
| `ties`, `dare_ties` | ties.py | `topk`, `scale`, `merging_type`, `dare_p`, `dare_seed` |
| `consensus` | consensus.py | `tall_threshold`, `consensus_threshold`, `scale` |
| `lines` | lines.py | `scale`, `alpha`, `beta` |
| `fisher`, `mats` | fisher.py, mats.py | `scale`, `fisher_normalize` (`per_task` / `none`), `num_iterations` (MaTS) |
| `tsv`, `iso_c` | tsv.py, iso.py | `scale` |
| `knots_ties` | knots.py | `topk`, `scale`, `merging_type` |
| `core_tsv`, `core_iso` | core.py | `scale`, `rank` (LoRA rank, 16) |
| `zeroshot`, `individual` | run.py | pseudo-methods: pretrained model / each task's own fine-tuned model |

`regmean` is `com` with `chain: false, task_weights: none, normalize_features: false`. RegMean in its original setting (Tab. 1, 2, 12) also sets
`normalize_by_tokens: false` and `data.calibration.num_samples: all` on the training split.

## Calibration keys (`data.calibration`)

`split` (`train` / `val`), `num_samples` (int or `all`), `seed`, `batch_size` (vision), `sampling` (`balanced` / `random`), `top_up` (split used to complete a
split with fewer than `num_samples` examples), `noise` (Gaussian noise: vision = std on the preprocessed images, Llama = multiple of the RMS of the calibration
token embeddings, applied only while merging), `max_class_fraction` (calibrate on at most this fraction of the classes), `source: synthetic`
(Stable Diffusion images under `data/synthetic/<task>`). The draws match the scripts that produced the paper's numbers. Vision draws the few-class subset with
`Random(seed)` per task, Llama with `Random(1000 * seed + task_index)`.

## Invariants

- Select on validation, report test once. Never tune anything on the test split.
- The calibration examples of CoM and RegMean are the only data they see. Labels are used only for class-balanced sampling.
- A merge returns deltas with respect to the pretrained weights. Parameters that are not solved in closed form (norms, biases, embeddings) are averaged.
- Padding tokens never enter a Gram matrix (`features` applies the token mask).
- Normalised accuracy divides by `eval.reference_acc`: a table in the model config, or `measured`, cached in `outputs/reference/<setting>_<split>.json`.
- Llama code depends on `transformers==4.44.2` (`models/llama.py: block_kwargs` rebuilds the decoder-layer arguments).
- YAML 1.1 reads `1e-6` as a string: write `1.0e-6` in configs (CLI overrides are coerced to float).

## Memory

- RegMean / Fisher over whole training sets stream the data (`num_samples: all` gives a DataLoader). CoM caches every task's block inputs on the CPU, so it needs a
  finite `num_samples`.
- ViT-L/14 full fine-tuning RegMean holds ~15 GB of Grams: `gram_spill_dir` keeps them on disk.
- Llama-3-8B: bf16 weights, float32 Grams and solves. RegMean on the whole training set uses `gram_cache: true` (Grams persisted under `outputs/cache/grams`).
- TIES on full fine-tuning streams per tensor (`ties_merge_streaming`) instead of stacking all task vectors.

## Adding a method

Create `commerge/methods/<name>.py` with `@METHODS.register("<name>") def merge(ctx) -> TaskVector`, import it in `commerge/methods/__init__.py`,
add `configs/methods/<name>.yaml` with its defaults and an experiment file per setting, and a CPU test in `tests/test_methods.py`.
