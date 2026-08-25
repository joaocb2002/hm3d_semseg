# HM3D semantic segmentation

`hm3d-semseg` is a standalone, inspectable pipeline for rendering
HM3D-Semantics v0.2 and fine-tuning SegFormer-B2 as the RGB perception component
of an ObjectNav system. It produces one categorical distribution over 41 classes
at every pixel: learnable `unknown` plus the 40 MPCAT40 classes. Invalid ground
truth remains the separate target value `255` and is never a model output.

The initial scope is deliberately narrow: aligned RGB/semantic/depth rendering,
offline datasets, an RGB SegFormer baseline, streamed evaluation, scalar
temperature calibration, and native-resolution inference. It does not implement
instance segmentation, RGB-D fusion, online training renders, or changes to
`object-nav-v2`.

## Architecture

```text
composed ObjectNav YAML ──> frozen camera profile ─┐
HM3D meshes/navmeshes ──> deterministic poses ────┼─> versioned offline dataset
semantic IDs ─> scene objects ─> Matterport TSV ──┘        │
                                                           v
RGB ─> SegFormer-B2 ─> 41 logits ─> softmax/calibration ─> ObjectNav API
                              │
                              └─> streamed global + scene-macro evaluation
```

Core code uses a `src/` layout. `camera/` resolves projection/extrinsics;
`taxonomy/` owns the only class list and mapping policy; `scenes/`, `sampling/`,
and `rendering/` generate aligned observations; `data/` owns the immutable
manifest contract; `models/`, `training/`, `evaluation/`, `calibration/`, and
`inference/` implement the baseline. The CLI is always `hm3d-semseg`.

## Physical workflow and artifact ownership

```text
WORKSTATION                          GPU SERVER
Git working tree --commit/push----> exact Git clone
validated datasets -------rsync----> server data root
                                     train/evaluate/calibrate
local source remains unchanged <---- complete run via rsync
ObjectNav loads calibrated checkpoint from the returned run
```

Git and model artifacts are separate channels. Git tracks source, checked
configuration, scene lists, tests, and documentation. Generated datasets,
TensorBoard events, run reports, and checkpoints are ignored by Git. Training
normally does not modify the server clone, so the workstation does not pull a
model with `git pull`: it copies the complete final run back into its external
`hm3d-semseg-data/runs` tree and loads the calibrated checkpoint from there.

## Local implementation findings

The read-only implementation audit found:

- `habitat-lab` at commit `9a1ddcb1c8b94b8555610303ec80c420b9faeb63`,
  clean and detached at `stable-1-g9a1ddcb1c`;
- `object-nav-v2` at commit `2493e72d24e28a217c232be6d53d45549e44f641`,
  clean on `main`;
- the `habitat` Conda environment uses Python 3.9.25, Habitat-Lab 0.3.3,
  Habitat-Sim 0.3.3, PyTorch 2.5.1+cu118, and torchvision 0.20.1+cu118;
- HM3D contains 145 annotated train, 36 annotated validation, and four annotated
  minival scenes;
- `object-nav-v2` actually uses Habitat-Lab's
  `benchmark/nav/objectnav/objectnav_hm3d.yaml`. Composing it yields landscape
  640×480 RGB/depth, 79° HFOV, `[0, 0.88, 0]` sensor position, agent height
  0.88 m, radius 0.18 m, normalized depth from 0.5–5.0 m, and 30° look actions;
- CUDA training compatibility is host-dependent: the local GTX 1070 is CC 6.1
  and requires the CUDA 12.6 rather than CUDA 13 PyTorch profile; headless EGL is
  checked independently in the render environment;
- training dependencies remain isolated from the Habitat environment.

These findings are implementation notes, not duplicated runtime constants. The
composed configuration always wins.

## Prerequisites

Obtain HM3D and HM3D-Semantics under their respective licenses. Install
Habitat-Lab/Habitat-Sim at versions compatible with the local assets. Do not let
this repository upgrade them. Obtain the authoritative
`matterport_category_mappings.tsv` from Habitat-Sim and record its checksum.

The current machine should use two environments until driver and dependency
compatibility are confirmed:

- render environment: existing `habitat` environment, plus `.[dev,render]`;
- training environment: a separate environment installed by the host-aware
  `install-training-env` command. It selects a tested CPU/CUDA wheel from GPU
  compute capability and driver support, and keeps `~/.local` disabled.

Both environments install this same editable package and exchange only the
offline dataset.

## Five-minute setup

```bash
conda activate habitat
cd ~/projects/hm3d-semseg
python -m pip install -e ".[dev,render]"
cp configs/local.example.yaml configs/local.yaml
```

Edit only `configs/local.yaml`. Every external path must be absolute. Then run:

```bash
hm3d-semseg doctor --local-config configs/local.yaml
hm3d-semseg resolve-camera \
  --local-config configs/local.yaml \
  --output /absolute/generated/root/camera/objectnav_resolved.yaml
pytest -m "unit"
```

Do not continue to full generation until the printed camera values match the
camera used by the ObjectNav agent and pitch values have been chosen explicitly.

For the separate training environment:

```bash
conda create -n hm3d-semseg-train python=3.10
conda activate hm3d-semseg-train
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg
python -m pip install -e .
hm3d-semseg install-training-env
hm3d-semseg install-training-env --apply --run-tests
```

The first installer invocation is read-only. On this computer it selects CUDA
12.6 for the Pascal GTX 1070; the same `auto` command can select CUDA 13.0 on a
compatible server or a CPU wheel on a machine without NVIDIA hardware. Every
long model workflow additionally runs a real CUDA kernel before loading the
model and auto-selects the working GPU with the most free memory.

## Workflow

Follow the numbered guide in order:

1. [Installation](docs/01_installation.md)
2. [Paths and configuration](docs/02_paths_and_configuration.md)
3. [Camera contract](docs/03_camera_contract.md)
4. [HM3D and taxonomy](docs/04_hm3d_and_taxonomy.md)
5. [Sampling and generation](docs/05_sampling_and_generation.md)
6. [Dataset format](docs/06_dataset_format.md)
6a. [Workstation-to-server handoff and return](docs/06a_server_handoff.md)
7. [Training](docs/07_training.md)
8. [Evaluation and calibration](docs/08_evaluation_and_calibration.md)
9. [Inference and ObjectNav integration](docs/09_inference_and_objectnav_integration.md)
10. [Testing](docs/10_testing.md)
11. [Troubleshooting](docs/11_troubleshooting.md)
12. [CLI reference](docs/12_cli_reference.md)

Reference: [losses, metrics, and run artifacts](docs/losses_and_metrics.md).

The machine boundary is deliberate: resolve the ObjectNav camera and render
licensed HM3D data in the Habitat workstation environment; transfer complete,
validated offline dataset directories to a host-matched single-GPU training
environment; then return the complete final run and calibrated checkpoint for
local verification and ObjectNav deployment. Never transfer an uncommitted
working tree or deploy a loose `model.safetensors` file. The handoff guide lists
the exact portable artifacts, integrity checks, server acceptance test,
scheduler workflow, final-protocol boundary, and return procedure.

At the end, the workstation owns two linked but separate records:

```text
~/projects/hm3d-semseg/                         # exact Git source revision
~/hm3d-semseg-data/runs/server/<final-run>/     # returned research artifacts
└── checkpoints/calibrated/                     # ObjectNav deployment input
```

The complete execution sequence is: install; configure local paths; run
`doctor`; freeze the camera; run unit tests; inspect minival; run Habitat tests;
audit minival; generate and validate the pilot; explicitly download the model;
run smoke and tiny-overfit checks; audit train/val; freeze the internal split;
generate and validate fit/development data; freeze the source commit; transfer
complete datasets; recreate and accept the server environment; compare recipes
on development; freeze the recipe; train on all 145 scenes; evaluate the fixed
36-scene official validation set; calibrate on disjoint scenes; return and
checksum the complete run; verify inference locally; integrate through the
Python API; benchmark the actual ObjectNav host.

## Known limitations

- `smoke-test` requires a functioning headless renderer, a locally cached pinned
  SegFormer snapshot, and PyTorch/Transformers; it fails explicitly when any is
  unavailable.
- Class-aware supplementary pose selection is represented in configuration and
  provenance but the initial generator currently implements only the unbiased
  spatial sampler. Leave `class_aware_fraction: 0.0`.
- Pitch increments can be resolved locally, but controller bounds cannot. Full
  configs therefore refuse to generate until `camera.pitch_degrees` is explicit.
- Primary evaluation is fixed-manifest segmentation, not policy-rollout
  distribution evaluation.
- There is no private-test evaluation, distributed training, or challenge
  submission packaging.

## Sources, citation, and licenses

The pipeline follows the primary [Habitat-Lab](https://github.com/facebookresearch/habitat-lab),
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim),
[HM3D-Semantics](https://aihabitat.org/datasets/hm3d-semantics/),
[SegFormer paper](https://arxiv.org/abs/2105.15203), and
[NVIDIA SegFormer-B2 checkpoint](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512).
The model download command records the resolved revision; weights are never
redistributed here. Cite and comply with the HM3D, HM3D-Semantics, checkpoint,
and source-code licenses in downstream work.
