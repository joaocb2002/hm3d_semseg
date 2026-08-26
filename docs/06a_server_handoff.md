# 6a. Workstation-to-server handoff and return

This guide is the operational bridge between offline HM3D rendering, GPU
training, and ObjectNav deployment. The physical machine may change; the
scientific inputs must not. Reproducibility comes from an exact source commit,
immutable dataset manifests, frozen scene lists, a pinned pretrained-model
revision, resolved configurations, and preserved run artifacts.

## Which machine owns each phase

| Phase | Recommended machine | Portable boundary |
|---|---|---|
| HM3D audit, camera resolution, rendering, dataset validation | Licensed workstation with Habitat and HM3D | Versioned offline dataset directories |
| Tiny-overfit plus baseline/balanced smoke checks | Workstation training environment | Local diagnostic runs; no calibration and no transfer required |
| Repeated tiny-overfit acceptance, full recipe development, final training, evaluation, calibration | One dedicated GPU on the server | Complete run and calibrated checkpoint directories |
| ObjectNav inference and rollout testing | Workstation or deployment host | Calibrated checkpoint plus the exact repository version |

The training server does not need Habitat-Lab, Habitat-Sim, HM3D meshes,
navmeshes, semantic descriptors, or `object-nav-v2`. It needs the repository,
offline datasets, a model snapshot, and the training environment. Do not copy
licensed HM3D source assets unless the server is an approved storage and compute
location under the applicable licenses.

## Two transfer channels, not one repository round trip

Keep source control and generated artifacts separate throughout the workflow:

| Channel | Workstation to server | Server to workstation |
|---|---|---|
| Git | Commit/push source, checked YAML, splits, tests, and docs; clone the exact SHA on the server | Usually nothing: training should not edit source. Pull only intentional server-side commits that were reviewed and pushed. |
| Artifact transfer | Copy the six complete offline dataset directories; optionally copy the pinned model cache | Copy complete development/final runs, evaluation reports, scheduler logs, and the calibrated checkpoint using `rsync` or managed storage. |

The trained model is never committed to this repository. `git pull` cannot
return it because `runs/`, checkpoints, `*.safetensors`, and `*.pt` are ignored.
The normal final workstation state is:

```text
~/projects/hm3d-semseg/                         # source at recorded Git SHA
~/hm3d-semseg-data/generated/                   # original rendered datasets
~/hm3d-semseg-data/runs/server/<final-run>/     # copied back from server
└── checkpoints/calibrated/                     # loaded by ObjectNav
```

Treat the server clone as read-only during scientific runs. If a source or
experiment change becomes necessary, make it deliberately, commit it, put both
machines on the new SHA, and restart or label the affected experiment. Never
copy checkpoint files into the Git working tree.

## 1. Freeze the workstation state

Finish the pilot, audits, scene split, camera contract, dataset generation, and
dataset validation in the `habitat` environment. Then check the repository:

```bash
cd ~/projects/hm3d-semseg
git status --short
git diff --check
python -m pytest -m unit
ruff check .
git rev-parse HEAD
```

Review and commit every intended source, configuration, scene-list, and
documentation change before training. Do not commit `configs/local.yaml`,
generated datasets, caches, runs, or weights. Record the resulting commit SHA in
the server job notes. A Git tag for the frozen experiment protocol is optional
but useful.

If the repository has no server-accessible remote, create and transfer a Git
bundle rather than copying an unrecorded working tree. Whichever mechanism is
used, the server must check out the recorded commit.

## 2. Decide which offline datasets to transfer

The authoritative purpose and chronology of every root are in the
[dataset lifecycle table](05_sampling_and_generation.md#dataset-lifecycle-why-six-roots-exist).
At this point Stage A is complete and validated:

- `train-v1`: the 130-scene fit dataset generated from `fit.txt`;
- `development-v1`: the disjoint 15-scene development dataset.

Complete Stage B before the recommended single transfer:

- `train-all-v1`: all 145 official training scenes, generated only after the
  130/15 protocol has been frozen;
- `official-val-v1`: the 36 official validation scenes;
- `calibration-fit-v1` and `calibration-evaluation-v1`: disjoint subsets of
  official validation used for temperature fitting and probability evaluation.

The smoothest workflow is to render and validate all of these on the
workstation before the first transfer. This is data preparation, not model
evaluation. Possessing `official-val-v1` on the server does not authorize using
its results for recipe selection: keep it embargoed until the
baseline/balanced recipe and duration are frozen.

The boundaries are strict: candidate-model gradients come only from
`train-v1`; candidate selection comes only from `development-v1`; final-model
gradients come only from `train-all-v1`; official metrics come from
`official-val-v1`; and temperature fitting/evaluation use their respective
disjoint calibration roots. The server workflow in sections 7 and 8 follows
these roles without resplitting any manifest.

To create a distinct all-training-scenes dataset, never extend or overwrite the
130-scene `train-v1` directory:

```bash
conda activate habitat
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg

hm3d-semseg generate-dataset \
  --config configs/data/train.yaml \
  --local-config configs/local.yaml \
  --dataset-name train-all-v1

hm3d-semseg validate-dataset \
  --dataset /absolute/generated/root/train-all-v1
```

The calibration scene lists are already frozen and checked in as
`configs/data/splits/calibration_fit.txt` and
`configs/data/splits/calibration_evaluation.txt`; do not regenerate them during
an ordinary run. Still in the local `habitat` environment, render official
validation and the two calibration roots:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --official-split val

hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --official-split val \
  --dataset-name calibration-fit-v1 \
  --split-list configs/data/splits/calibration_fit.txt

hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --official-split val \
  --dataset-name calibration-evaluation-v1 \
  --split-list configs/data/splits/calibration_evaluation.txt
```

Run each command with `--dry-run` first when checking storage plans. Then
validate every newly rendered root:

```bash
hm3d-semseg validate-dataset \
  --dataset /absolute/generated/root/official-val-v1
hm3d-semseg validate-dataset \
  --dataset /absolute/generated/root/calibration-fit-v1
hm3d-semseg validate-dataset \
  --dataset /absolute/generated/root/calibration-evaluation-v1
```

`official-val-v1` and the two calibration roots intentionally contain two
rendered copies of the same 36 validation scenes partitioned in different
ways. The evaluator consumes one self-contained dataset root at a time: the
full root provides official hard-segmentation metrics, while the disjoint roots
prevent temperature-fit leakage into probability evaluation.

It is also valid to transfer only fit/development initially and transfer the
final datasets later; that saves temporary storage but introduces a second
handoff. Temperature fitting is not performed now. Only the calibration images
and masks are prepared now; `hm3d-semseg calibrate` runs on the server after the
final model is frozen.

Every directory is self-contained. Transfer the whole directory, including
`dataset.yaml`, `resolved_config.yaml`, `provenance.json`,
`camera_profile.yaml`, `manifest.jsonl`, scene files, and validation artifacts.
Do not copy only the RGB and mask subdirectories.

## 2a. Workstation pre-handoff training tests

After all six roots validate, run three bounded checks locally. These commands
train model weights; they do not calibrate probabilities. Dataset names and all
optional training/evaluation fields are explicit in the checked YAML files and
resolve below `paths.generated_data_root` from `configs/local.yaml`.

| Test | Train | Development | Purpose |
|---|---:|---:|---|
| Tiny overfit | 4 images, 50 epochs | `null` | Verify memorization and core training mechanics. |
| Baseline smoke | 512 images, 10 epochs | 256 images after each epoch | Exercise unweighted training and held-out metrics. |
| Balanced smoke | 512 images, 10 epochs | Same 256 images | Exercise inverse-square-root class weights under otherwise identical settings. |

Use the training environment:

```bash
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg

hm3d-semseg train \
  --config configs/experiments/overfit_tiny.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline_smoke.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer_b2_moderately_balanced_smoke.yaml \
  --local-config configs/local.yaml
```

The limited selections are deterministic, scene-diverse, and recorded by exact
ID in `provenance.json`. Edit the two `max_*_samples` values or `epochs` in a
smoke YAML when deliberately changing its cost; `null` means the complete
manifest. Accept a run only when it completes with finite values, no scene
overlap, and populated reports, plots, and `checkpoints/{best,last}`.

Artifacts are written only below
`/home/joaocb2002/hm3d-semseg-data/runs/<run_name>/`. Existing names receive a
safe numeric suffix and are never overwritten. Smoke metrics verify plumbing;
they are too sample-limited for recipe selection. The local smoke runs do not
need to move to the server. See the concise
[experiment matrix](../configs/experiments/README.md) and
[artifact map](losses_and_metrics.md#artifact-map-after-train).

After the checks pass, rerun the source checks in section 1 and commit/push any
intentional code, YAML, test, or documentation changes. The server must use that
final recorded SHA, not an earlier pre-smoke commit.

## 3. Transfer data without changing it

Create a private project location on persistent server storage. Prefer
node-local NVMe or fast scratch for training I/O, but keep a second copy or a
recoverable source when scratch is ephemeral. A representative resumable copy
is:

```bash
rsync -a --partial --info=progress2 --checksum \
  /absolute/generated/root/train-v1 \
  /absolute/generated/root/development-v1 \
  /absolute/generated/root/train-all-v1 \
  /absolute/generated/root/official-val-v1 \
  /absolute/generated/root/calibration-fit-v1 \
  /absolute/generated/root/calibration-evaluation-v1 \
  <user>@<server>:/absolute/server/data/
```

The six-directory command is the recommended single-transfer workflow. Omit
the final four directories only when deliberately choosing the two-transfer
workflow. Do not use `--delete` for a handoff. If the site provides managed
object storage or a data-transfer node, prefer that mechanism while retaining
the same complete-directory rule.

Do not blindly copy the entire `hm3d-semseg-data` directory:

- `generated/<dataset-name>` directories listed above are the required data;
- `cache/` is optional and may instead be reconstructed from the pinned model
  revision;
- existing local `runs/` are not needed unless resuming one of those exact
  runs;
- audits, inspection images, and the pilot remain useful local records but are
  not training-server inputs.

Transfer repository source through Git at the exact committed SHA rather than
copying the repository working directory with the data.

The Hugging Face cache is optional to transfer. On a connected server, run
`download-model` against the already recorded revision. On an offline server,
copy the complete pinned snapshot/cache rather than a loose weight file.

## 4. Recreate the training environment on the compute host

Clone the repository and check out the frozen SHA. Do not copy the workstation
Conda environment: binary environments are host-dependent.

```bash
git clone <repository-location> ~/projects/hm3d-semseg
cd ~/projects/hm3d-semseg
git checkout <frozen-commit-sha>

conda create -n hm3d-semseg-train python=3.10
conda activate hm3d-semseg-train
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
python -m pip install -e .

hm3d-semseg install-training-env
hm3d-semseg install-training-env --apply --run-tests
```

Run host detection on a GPU compute node, not a restricted login node, because
the automatic profile probes a real CUDA kernel. The code is single-process and
single-GPU; request exactly one GPU even when the host contains several.

After installation, preserve the environment for all recipe comparisons. Each
run records package and hardware provenance. Changing package versions between
the baseline and balanced runs invalidates a controlled comparison.

## 5. Create a server-local configuration

Create a fresh ignored `configs/local.yaml`. Do not copy workstation absolute
paths. A training-only server does not need the Habitat-related path keys:

```yaml
paths:
  generated_data_root: /absolute/server/data
  runs_root: /absolute/server/runs
  cache_root: /absolute/server/cache

camera:
  profile: /absolute/server/data/train-v1/camera_profile.yaml
  allow_mismatch: false

model:
  revision: <resolved-hugging-face-commit>
  local_files_only: true

training:
  device: auto
```

Keep scientific settings—seed, augmentation, learning rates, weighting,
epochs, dataset names, subset limits, and batch semantics—in checked experiment
YAML files. `local.yaml` contains host paths and the device override. Worker
count may be tuned after measuring input throughput, but record every change in
the resolved configuration.

Download or confirm the pinned model snapshot:

```bash
hm3d-semseg download-model \
  --local-config configs/local.yaml \
  --model-id nvidia/segformer-b2-finetuned-ade-512-512 \
  --revision <resolved-hugging-face-commit>
```

## 6. Verify the handoff before a long job

Validate each transferred dataset once on the server before its first use. This
fully decodes the files and catches a truncated transfer rather than discovering
it during epoch ten. The initial development phase requires:

```bash
hm3d-semseg validate-dataset --dataset /absolute/server/data/train-v1
hm3d-semseg validate-dataset --dataset /absolute/server/data/development-v1
```

Before final training and evaluation, similarly validate `train-all-v1`,
`official-val-v1`, `calibration-fit-v1`, and
`calibration-evaluation-v1` if they were included in the handoff.

Then repeat the configured tiny-overfit experiment on the server:

```bash
hm3d-semseg train \
  --config configs/experiments/overfit_tiny.yaml \
  --local-config configs/local.yaml
```

Check loss reduction, the final selected-subset metrics, qualitative alignment,
GPU memory, and throughput. This is the server acceptance test. It does not
replace held-out development evaluation.

## 7. Run recipe development as a durable job

First identify the server's execution policy:

```bash
command -v sbatch
command -v qsub
command -v bsub
```

Use Slurm, PBS, LSF, or the site's equivalent when one is provided. Ask the
administrator which host accepts long GPU jobs; never assume the login node is
also a compute node. Request one GPU, sufficient CPU workers and memory,
persistent stdout and stderr, and a wall-time compatible with checkpoint
cadence. A minimal Slurm body, only for a site that actually uses Slurm, is:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=hm3d-segformer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/absolute/server/logs/%x-%j.log

set -euo pipefail
source /absolute/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
cd /absolute/server/repository/hm3d-semseg

hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml
```

Adapt only the scheduler directives and absolute installation path to the
site. If no scheduler is installed and the administrator confirms that the
machine is a directly managed standalone server, use `tmux` so an SSH
disconnect does not terminate training:

```bash
mkdir -p /absolute/server/logs
tmux new -s hm3d-baseline

conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
cd /absolute/server/repository/hm3d-semseg
hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml \
  2>&1 | tee /absolute/server/logs/segformer_b2_baseline.log
```

Detach with `Ctrl-b`, then `d`; reconnect with
`tmux attach -t hm3d-baseline`. Do not use this fallback to bypass a cluster
scheduler or local usage policy.

Submit or run the baseline and moderately balanced experiments separately using
the same commit, environment, fit dataset, development dataset, and hardware
class. Do not choose a recipe from training loss alone. Use development
known-class mIoU as the primary selection metric, supported by per-class,
ObjectNav-six, scene-macro, confusion, and qualitative results.

For remote TensorBoard, bind it to the server loopback interface and use SSH
port forwarding; never expose it publicly. The exact tunnel depends on whether
the service runs on a login or allocated compute node.

If a job is preempted, preserve its run directory and resume explicitly from
`checkpoints/last` using the same requested run name. Do not start a nominally
identical fresh run and mix its metrics manually.

## 8. Freeze and execute the final protocol

After recipe development:

1. Select baseline or moderate balancing using development data only.
2. Freeze all scientific hyperparameters and the number of epochs.
3. Create a checked final experiment YAML with a new, unambiguous run name.
4. Set `training.datasets.train: train-all-v1`.
5. Set `training.datasets.development: null` so the 15 development scenes rejoin
   the final training population and are no longer used for model selection.
6. Train for the frozen duration. The protocol checkpoint is
   `checkpoints/last`.
7. Evaluate that checkpoint once on `official-val-v1`.
8. Fit temperature only on `calibration-fit-v1`; evaluate probability quality
   only on `calibration-evaluation-v1`.
9. Benchmark the calibrated checkpoint on the server and later on the actual
   ObjectNav host.

Do not tune the recipe after viewing official-validation results. Any such
change starts a new explicitly labeled research iteration.

## 9. Preserve and return the results

The complete final run is the research record. Copy it back to a new local
destination without overwriting previous runs. It contains resolved config,
provenance, JSONL metrics, summaries, plots, TensorBoard events, diagnostics,
evaluation reports, and recoverable checkpoints.

This is an artifact return, not a Git pull. If `git status --short` is clean on
the server—as it normally should be—the workstation repository already has the
correct source. If intentional source commits were made on the server, push and
review them through the normal Git remote, then check out that exact revision on
the workstation separately from copying the run.

Before copying, create a small checksum file inside the calibrated checkpoint:

```bash
cd /absolute/server/runs/<final-run>/checkpoints/calibrated
sha256sum \
  model.safetensors \
  config.json \
  checkpoint.json \
  camera_profile.yaml \
  calibration.json > SHA256SUMS
```

Then copy the complete final run and the development-run summaries needed to
justify recipe selection:

```bash
rsync -a --partial --info=progress2 \
  <user>@<server>:/absolute/server/runs/<final-run>/ \
  /absolute/local/runs/server/<final-run>/

cd /absolute/local/runs/server/<final-run>/checkpoints/calibrated
sha256sum -c SHA256SUMS
```

For a complete research archive, also return the full selected development run,
the competing recipe's `summary.json`, `metrics_summary.json`, development
evaluation reports and plots, plus scheduler or `tmux` logs. The calibrated
checkpoint inside the final run is the deployment artifact; the rest explains
how it was produced.

Keep the server copy until the local checksum, inference test, and institutional
backup have succeeded. Do not copy the Conda environment back. The workstation
already owns the rendered datasets; the model cache can be reconstructed from
the pinned revision.

For long-term reproducibility, archive:

- the exact Git SHA or tag;
- selected development run summaries and plots;
- the complete final run, including `training_state.pt` for recovery;
- official evaluation and calibration reports;
- the calibrated checkpoint and its checksums;
- scheduler logs and any environment export required by local policy.

## 10. Verify locally and integrate with ObjectNav

Use the repository version recorded in the returned run's `provenance.json`.
In the ObjectNav environment, install this package without changing the sibling
ObjectNav or Habitat repositories, then run single-image inference:

```bash
conda activate habitat
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg
python -m pip install -e ".[inference]"

hm3d-semseg infer \
  --checkpoint /absolute/local/runs/server/<final-run>/checkpoints/calibrated \
  --image /absolute/path/to/a/representative/rgb.png \
  --output /absolute/local/inference-check
```

Inspect the class IDs, overlay, confidence, entropy, and metadata. Resolve the
actual ObjectNav runtime configuration and require
`segmenter.assert_camera(runtime_camera)` before rollouts. Benchmark inference
on this target GPU because server throughput is not deployment throughput.

For deployment, the calibrated checkpoint directory is the unit of transfer:

```text
calibrated/
├── model.safetensors
├── config.json
├── checkpoint.json
├── camera_profile.yaml
├── calibration.json
└── SHA256SUMS
```

`training_state.pt` is needed to resume optimization but not for inference; keep
it in the archived full run. Never deploy only `model.safetensors`, because the
architecture, label mapping, camera contract, model provenance, and calibrated
temperature would be lost.

Next: [training](07_training.md).
