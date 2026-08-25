# 11. Troubleshooting

## Conda environment uses `~/.local` packages

If pip says requirements are satisfied under `~/.local/lib/python...`, the
environment is contaminated by the Python user site. Run
`conda env config vars set PYTHONNOUSERSITE=1`, reactivate, and run
`hm3d-semseg install-training-env --apply`. Package paths in the installer
report should point inside the active Conda prefix.

## CUDA is true but the GPU architecture is unsupported

`torch.cuda.is_available()` only proves driver/runtime visibility. If the
installer or runtime kernel probe reports an unsupported `sm_XX`, run:

```bash
hm3d-semseg install-training-env
hm3d-semseg install-training-env --apply
```

For example, CUDA 13 wheels do not contain GTX 1070 (Pascal, `sm_61`) kernels;
the automatic plan selects the CUDA 12.6 wheel. Do not suppress this warning or
start training merely because CUDA reports true.

## Viewer name

Habitat's executable is normally `habitat-viewer`, not a generic `viewer`.
Viewer success does not prove headless EGL works.

## EGL/OpenGL or NVIDIA failures

Run `nvidia-smi` and `hm3d-semseg doctor`. A driver/library mismatch, inaccessible
device, or bad `gpu_device_id` must be repaired outside this repository. Use the
host-aware installer for training PyTorch; do not mutate the Habitat environment.
Verify headless rendering in the render environment before generation.

## Few semantic IDs or mostly zero

Confirm the annotated scene-dataset config, `load_semantic_mesh`, semantic GLB,
and semantic TXT. A non-annotated basis config can render geometry but not useful
labels. Run `inspect-scene` and review `semantic_id_decisions`.

## Missing meshes/descriptors

`doctor` and discovery report incomplete annotated scenes before long work.
HM3D-Semantics has 145 train/36 val/four annotated minival scenes, not every base
HM3D directory.

## Swapped or mismatched images

The local camera is width 640, height 480. The official fallback is width 480,
height 640. Compare frozen hashes. Never use bilinear interpolation for masks or
apply geometry to RGB alone.

## Invalid targets or dominant unknown

Run `validate-dataset` and `audit-taxonomy`. Targets may only be 0–40 or 255.
Inspect unmapped raw names rather than converting all failures to unknown. Freeze
the exact Matterport mapping SHA.

## Hugging Face label reduction

This project never delegates resizing/labels to the image processor.
`reduce_labels` is false and class zero stays learnable. Cross-entropy receives
raw upsampled logits.

## Out of memory

Reduce batch size, enable AMP, or increase accumulation. Keep native aspect
ratio. Record every change. Do not silently downscale deployment resolution.

## Slow loading

Measure decoding separately, increase workers/pinned memory, and prefer recorded
JPEG/WebP only after PNG correctness. Profile before adding shards.

## Tiny overfit fails

Disable augmentation; inspect paired images/masks; verify label 0 and ignore 255;
confirm nearest mask geometry and raw-logit loss; inspect classifier LR and
gradients. Full training is blocked until memorization works.

## Interrupted generation

Rerun the identical command. Contract changes refuse resume. Never edit the
manifest manually. Validate when complete.

## Interrupted training

Set `training.resume` to `checkpoints/last`. Model, optimizer, scheduler, scaler,
epoch, and global step are restored. If config or camera/data changed, start a
new named run.

Next: [CLI reference](12_cli_reference.md).
