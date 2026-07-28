# 2. Paths and configuration

Configuration is strict. Unknown keys fail. Precedence is CLI overrides, then
`configs/local.yaml`, then the command/experiment YAML, then dataclass defaults.
Every command saves or prints its resolved values.

```bash
cp configs/local.example.yaml configs/local.yaml
```

For this machine, the discovered inputs are:

```yaml
paths:
  habitat_lab_root: /home/joaocb2002/projects/habitat-lab
  hm3d_root: /home/joaocb2002/projects/habitat-lab/data/scene_datasets/hm3d_v0.2
  scene_dataset_config: /home/joaocb2002/projects/habitat-lab/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json
  objectnav_config: /home/joaocb2002/projects/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml
  taxonomy_mapping: /home/joaocb2002/Downloads/matterport_category_mappings.tsv
```

Choose absolute, writable locations outside Git for `generated_data_root`,
`runs_root`, and `cache_root`. The locally inspected mapping hash is
`36e40c25cbe32c8bf34ef55f199f194671045106914dd09b1581aeedcf051a05`;
it is the checked default for `taxonomy.expected_mapping_sha256`. Override it
only as a deliberate taxonomy-asset revision after auditing the new mapping.

Run:

```bash
hm3d-semseg doctor --local-config configs/local.yaml
```

Expected output includes package versions, all paths, annotated counts
train=145/val=36/minival=4, CUDA state, and driver diagnostics. A failed required
path blocks later commands. A failed CUDA check explains why rendering/training
is unavailable but does not invalidate pure taxonomy/unit work.

Generated data paths never default to the current working directory.
`configs/local.yaml`, data, runs, weights, and caches are ignored by Git.

Next: [freeze the camera](03_camera_contract.md).
