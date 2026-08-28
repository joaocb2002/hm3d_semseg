# Reference: troubleshooting

## Conda uses packages from `~/.local`

Set `PYTHONNOUSERSITE=1`, reactivate, and reapply the host-aware installer:

```bash
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
hm3d-semseg install-training-env --apply
```

Package paths must point inside the active Conda prefix.

## CUDA is visible but kernels are unsupported

`torch.cuda.is_available()` proves visibility, not architecture compatibility.
Run the dry plan, inspect it, then apply:

```bash
hm3d-semseg install-training-env
hm3d-semseg install-training-env --apply --run-tests
```

For example, a GTX 1070 requires the supported Pascal/CUDA 12.6 profile rather
than a CUDA 13 wheel. Never suppress the real-kernel failure.

## The server has multiple GPUs but only one may be used

Use the scheduler's one-GPU request when a scheduler exists. On an authorized
direct-use server, inspect `nvidia-smi` and expose only the assigned device:

```bash
export CUDA_VISIBLE_DEVICES=GPU_INDEX
```

Current `knuth` example when physical GPU 0 is assigned:

```bash
export CUDA_VISIBLE_DEVICES=0
```

The selected physical GPU then appears to PyTorch as `cuda:0`.

## `rsync: command not found` from the remote shell

Both ends need `rsync`. On `knuth` it is isolated in `hm3d-transfer`; pass:

```bash
--rsync-path=/workspace/miniconda/envs/hm3d-transfer/bin/rsync
```

An unbracketed IPv6 address is also parsed incorrectly by `rsync`. Prefer the
working SSH alias `joao_branco@knuth`; otherwise bracket the literal IPv6.

## SSH disconnects during a run

Use `tmux`: create with `tmux new -s NAME`, detach with `Ctrl-b` then `d`, list
with `tmux ls`, and reattach with `tmux attach -t NAME`. A detached session keeps
training or TensorBoard alive.

## Server storage is filling

Current storage is shared under `/workspace`:

```bash
df -h /workspace
du -sh /workspace/data /workspace/cache /workspace/runs /workspace/logs
```

Do not delete a final or resumable run merely to make room. First identify safe
duplicate smoke artifacts or request more persistent storage. Keep at least the
accepted source provenance, best/last checkpoints, optimizer state needed for
resume, metrics, and final calibrated run.

## VS Code Remote SSH becomes slow

Open `/workspace/repository/hm3d-semseg` and `/workspace/runs` as workspace
folders. Avoid adding `/workspace/data`, whose thousands of images can trigger
expensive indexing. Preview JSON/YAML/PNG statically and use a forwarded
loopback-only TensorBoard port for live curves.

## EGL/OpenGL or NVIDIA rendering fails

Run `nvidia-smi` and workstation `hm3d-semseg doctor`. Repair driver/library,
device-access, or EGL problems outside this repository. Training servers do not
need Habitat rendering.

## Too few semantic IDs or mostly unknown

Confirm the annotated scene-dataset config, semantic GLB/TXT, and
`load_semantic_mesh`. Run `inspect-scene` and inspect
`semantic_id_decisions`; do not fuzzy-map or convert every failure to unknown.

## Missing meshes or descriptors

Discovery must report 145 complete train scenes, 36 complete validation scenes,
and four annotated minival scenes. A base HM3D directory is not necessarily an
annotated HM3D-Semantics scene.

## Swapped or mismatched camera images

The resolved local camera is 640×480, 79° HFOV. The official comparison fixture
is 480×640 and is not the source of truth. Compare frozen camera hashes. Never
bilinearly resize masks or transform RGB without the mask.

## Invalid targets or dominant unknown

Run dataset validation and taxonomy audit. Targets may only be 0--40 or 255.
Class 0 is learnable unknown; 255 is ignored and has no model logit.

## Hugging Face label reduction appears

This project does not delegate resizing or labels to the image processor.
`reduce_labels` remains false; class 0 stays learnable; cross-entropy receives
raw native-shape logits.

## GPU out of memory

Reduce batch size, enable AMP, or raise gradient accumulation while preserving
effective batch semantics. Record every change in a checked experiment. Do not
silently change deployment resolution.

## Loading is slow

Measure decoding and GPU utilization before changing anything. Tune workers to
the server CPU/storage, and use recorded JPEG/WebP only after the PNG
correctness protocol. Do not invent dataset shards without a measured need.

## Tiny overfit fails

Disable augmentation; inspect paired RGB/masks; verify class 0 and ignore 255;
confirm nearest mask geometry, raw-logit loss, decode-head learning rate, and
gradients. Full recipe runs are blocked until four selected images nearly
memorize.

## Generation is interrupted

Rerun the identical generation command. Contract changes refuse resume. Never
edit a manifest manually; validate after completion.

## Training is interrupted

Set `training.resume` to the prior `checkpoints/last`. Model, optimizer,
scheduler, scaler, epoch, and global step restore together. If source,
configuration, camera, or data changed, start a new named run.

Return to the [execution guide](README.md) or consult the
[CLI reference](reference_cli.md).
