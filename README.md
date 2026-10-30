# Chain of Merges (CoM)

Code for **Rethinking Layer-wise Model Merging through Chain of Merges** (NeurIPS 2026),
Pietro Buzzega, Riccardo Salami, Angelo Porrello, Simone Calderara.

Activation-matching merges such as RegMean solve every layer independently, using the activations of the
individual fine-tuned models. Once the earlier layers are merged, those activations are no longer the ones the
merged network produces, so the statistics used for the later layers are stale. We call this *merging covariate shift*.
CoM merges layer by layer and recomputes every layer's input with the partially merged network, so each closed-form
solve sees the activations it will actually receive. It also weights every task-layer pair by the off-diagonal mass of
its feature-correlation matrix, which prioritises the tasks that moved farthest from the pretrained model.

The repository contains CoM, every baseline of the paper (Task Arithmetic, TIES, DARE-TIES, Consensus TA, LiNeS,
FisherAVG, RegMean, MaTS, TSV, Iso-C, KnOTS-TIES, CoreTSV, CoreISO) and the scripts that reproduce every table and figure
listed under [Reproducing the paper](#reproducing-the-paper).

## Install

The environment is pinned (Python 3.11, PyTorch 2.11.0 with CUDA 12.8, `transformers` 4.44.2, and the rest in `uv.lock`; the CUDA 12.8 build runs on GPUs from Turing, e.g. RTX 2080 Ti, to Blackwell). With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aimagelab/chain-of-merges && cd chain-of-merges
uv sync --extra dev --extra figures          # creates .venv from uv.lock
uv run python -m commerge.run experiments/vit_b32_lora/com.yaml
```

Without uv, in a Python 3.11 environment: `pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[dev,figures]"` installs the same
pinned versions. Add `--extra synthetic` for the Stable Diffusion generation script. The Llama code relies on the
`transformers` 4.44 decoder-layer interface, so do not upgrade it.

## Data and checkpoints

Everything is stored under `./data` (or `$COMMERGE_DATA`) and **downloaded automatically the first time an experiment needs it**:

| what | source | size |
|---|---|---|
| 8 vision datasets (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN) with the paper's splits | official sources and pinned Hugging Face datasets, checked by md5 / pixel hash; SVHN and GTSRB fall back to Hugging Face copies of the original files when their hosts are unreachable | ~20 GB download, ~40 GB free disk while building |
| CLIP zero-shot classification heads | TALL-masks repository on Hugging Face | small |
| CLIP ViT-B/32 and ViT-L/14, LoRA checkpoints (KnOTS), full fine-tuning checkpoints (Ilharco et al.) | Hugging Face | ~15 GB |
| 6 NLI datasets | Hugging Face `datasets` | ~1 GB |
| Llama-3.2-1B LoRA adapters of the paper | GitHub release asset `llama32_1b_lora_adapters.tar.gz` | 76 MB |
| Stable Diffusion calibration images of the paper | GitHub release asset `synthetic_sd15.tar.gz` | 280 MB |

The release assets are not part of the git history, so `git clone` stays small. They are fetched only when an experiment uses them.
To download things in advance:

```bash
python -m commerge.prepare --vision            # datasets + zero-shot heads (add --tasks mnist svhn ... for a subset)
python -m commerge.prepare --nli --models      # NLI datasets, CLIP models and public checkpoints
python -m commerge.prepare --models --gated    # also Llama-3-8B and Llama-3.2-1B (accept their licences on the Hub, then `huggingface-cli login`)
python -m commerge.prepare --llama1b-adapters --synthetic
```

The fully fine-tuned Llama-3.2-1B checkpoints of Tab. 12 and 13 are not released (6 × 5 GB). Train them with
`scripts/train_llama1b.sh full`, which uses the paper's recipe (see [Limitations](#limitations-and-known-differences)).

## Usage

```bash
python -m commerge.run experiments/vit_b32_lora/com.yaml                          # merge + evaluate one setting
python -m commerge.run experiments/vit_b32_lora/com.yaml data.calibration.num_samples=100 method.alpha=0.95
python -m commerge.search experiments/vit_b32_lora/ties.yaml                      # tune on validation, report test once
```

A run writes `config.yaml`, `log.txt` and `metrics.json` to `outputs/runs/<experiment>/<timestamp>/`. `metrics.json` contains, for every task,
the absolute accuracy `<task>/acc` and the normalised accuracy `<task>/norm_acc` (merged accuracy divided by the accuracy of the
task's own fine-tuned model), plus their averages `avg/acc` and `avg/norm_acc`.

Every experiment is one YAML file, `configs/experiments/<setting>/<method>.yaml`. It composes a model file (`configs/models/`)
and a method file (`configs/methods/`), and stores the hyper-parameters used in the paper. Its `search:` block holds the grid they
were selected from on the validation split. Any key can be overridden on the command line (`key.sub=value`).

Settings: `vit_b32_lora`, `vit_l14_lora`, `llama3_8b_lora` (LoRA, rank 16), `vit_b32_full`, `vit_l14_full`, `llama32_1b_full`
(full fine-tuning), and `llama32_1b_lora` (only used by the matched-data study).

## Reproducing the paper

Run the commands from the repository root, inside the environment (`source .venv/bin/activate`, or prefix them with `uv run`). All scripts write under `outputs/`;
`python scripts/tables.py` then collects everything into `outputs/RESULTS.md`.

| paper | command |
|---|---|
| Tab. 1, 2, 9, 10, 11 (main results, per-dataset accuracy) | `scripts/main_tables.sh vit_b32_lora vit_l14_lora llama3_8b_lora vit_b32_full vit_l14_full` |
| Tab. 12, 13 (Llama-3.2-1B, full fine-tuning) | `scripts/train_llama1b.sh full`, then `scripts/main_tables.sh llama32_1b_full` |
| Tab. 3, 4 (MCS / Norm / FC ablation) | `scripts/ablation.sh vit_b32_lora vit_l14_lora llama3_8b_lora` |
| Tab. 5 (calibration set) | `scripts/calibration_sensitivity.sh vit_b32_lora vit_l14_lora llama3_8b_lora` |
| Tab. 6, Fig. 3 (number of calibration examples) | `scripts/sample_sweep.sh vit_b32_lora vit_l14_lora llama3_8b_lora`, then `python scripts/figures/samples.py fig3.pdf fig3.png` |
| Tab. 7, Fig. 5 (synthetic calibration data) | `python scripts/synthetic.py vit_b32_lora` (and `vit_l14_lora`) |
| Tab. 8, Fig. 4 (merging vs fine-tuning on the same n examples) | `scripts/matched_data.sh`, then `python scripts/figures/matched.py fig4.pdf fig4.png` |
| Fig. 6, 10 (task importance vs accuracy gap) | `python scripts/correlation.py vit_b32_lora` (and `vit_l14_lora`, `llama3_8b_lora`) |
| Fig. 7, 8 (memory and FLOPs) | `python scripts/figures/complexity.py <out_dir>` (analytic cost model) |

Calibration-based methods (CoM, the ablation, Tab. 5, 6, 8) are run with calibration seeds 420, 1 and 2, and the tables report the mean.
Data-free methods are deterministic and run once, except DARE-TIES on Llama, which is averaged over three drop masks.

### Hardware

ViT-B/32 experiments run on an 11 GB GPU in minutes. ViT-L/14 runs in tens of minutes; RegMean and Fisher over the whole training set,
and the full fine-tuning settings, need about 30 GB of host RAM. Llama-3-8B needs a ≥ 40 GB GPU and ≥ 64 GB of host RAM. The Llama-3.2-1B merges of
Tab. 8 fit an 11 GB GPU with about 25 GB of host RAM; fine-tuning Llama-3.2-1B (Tab. 8, and the full fine-tuning of Tab. 12) needs a ≥ 40 GB GPU.

## Limitations and known differences

- **Llama-3.2-1B full fine-tuning (Tab. 12, 13).** The six checkpoints are not released. `scripts/train_llama1b.sh full` uses the paper's recipe:
  AdamW without weight decay, cosine schedule with 3% warm-up, effective batch 32, bf16 autocast over fp32 weights, learning rate in {5e-6, 1e-5, 2e-5}, and
  early stopping on validation (at most 2 epochs for SNLI and MNLI, 3 otherwise). The paper's runs selected 5e-6 for SNLI, MNLI, QNLI and SciTail, 1e-5 for RTE, and 2e-5 for SICK. Retrained checkpoints will differ slightly,
  and so will the numbers. The configs keep the hyper-parameters selected on validation for the paper's checkpoints; re-select them with `commerge.search` for new checkpoints.
- **Llama-3.2-1B LoRA adapters.** They were trained by us and are released as a GitHub asset. `scripts/train_llama1b.sh lora` retrains them, but fp16 training is not
  bit-reproducible, so use the released adapters to reproduce Tab. 8.
- **Synthetic images.** `scripts/synthetic_generation/` contains the Stable Diffusion v1.5 generation and downsampling code. The class of every image was
  drawn with an unseeded random choice, so regenerating does not give the same images. Use the released ones (their labels, derived from the prompts, are in
  `data/synthetic/<task>/labels.json`).
- **Not included:** the MCS curves of Fig. 2 and 11, the zero-shot-reference bars of Fig. 9, and the Localize-and-Stitch, AdaMerging++ and ProDistill rows of Tab. 2
  (taken from their official implementations or papers).
- **Calibration split of Llama.** CoM on Llama calibrates on class-balanced *training* examples (the validation splits of RTE and SICK
  have fewer than 500 examples). The validation rows of Tab. 5 complete those two tasks with training examples (`top_up: train`).
- **Hyper-parameter grids.** For a few Llama-3-8B baselines, the final value was selected on a finer grid around the first winner. The config stores the selected
  value and the first grid.
- **Determinism.** On a given GPU, merges, evaluation and the fine-tuning of Tab. 8 are deterministic: rerunning gives the same digits. Across GPU models,
  float rounding flips a few test predictions. Re-running the Llama-3.2-1B merges of Tab. 8 on a different GPU changed the normalised accuracy by up to 0.13 points.

## Citation

```bibtex
@inproceedings{buzzega2026chain,
  title     = {Rethinking Layer-wise Model Merging through Chain of Merges},
  author    = {Buzzega, Pietro and Salami, Riccardo and Porrello, Angelo and Calderara, Simone},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
