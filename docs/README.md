# HM3D semantic-segmentation execution guide

This directory is the operational interface for the project. Follow the ten
execution stages in order. Open a reference only when a stage links to it; the
references are not additional pipeline steps.

## Execution sequence

| Step | Document | Machine | Exit condition |
|---:|---|---|---|
| 1 | [Install the environments](01_installation.md) | Workstation | Separate render and training environments pass their checks. |
| 2 | [Configure workstation paths](02_paths_and_configuration.md) | Workstation | `configs/local.yaml` resolves the local Habitat and data roots. |
| 3 | [Freeze the camera contract](03_camera_contract.md) | Workstation | The resolved ObjectNav camera profile is accepted. |
| 4 | [Audit HM3D and the taxonomy](04_hm3d_and_taxonomy.md) | Workstation | Semantic IDs map reproducibly to model IDs 0--40 or ignore 255. |
| 5 | [Render the six datasets](05_sampling_and_generation.md) | Workstation | All dataset roles and scene partitions are frozen. |
| 6 | [Validate the dataset contract](06_dataset_format.md) | Workstation | Every portable dataset root validates. |
| 7 | [Move from workstation to GPU server](07_workstation_to_server.md) | Both | Exact source, data, model snapshot, and server environment pass tiny overfit. |
| 8 | [Develop and refit the model](08_server_training.md) | GPU server | A recipe is selected on development data and refit on all training scenes. |
| 9 | [Evaluate and calibrate the final model](09_server_evaluation_and_calibration.md) | GPU server | Official metrics, temperature, calibration metrics, and benchmark are frozen. |
| 10 | [Return artifacts and integrate with ObjectNav](10_return_and_objectnav.md) | Server to workstation | The complete calibrated checkpoint passes local inference and camera checks. |

The current machine-specific roots used by this project are:

| Purpose | Workstation | GPU server (`knuth`) |
|---|---|---|
| Repository | `/home/joaocb2002/projects/hm3d-semseg` | `/workspace/repository/hm3d-semseg` |
| Datasets | `/home/joaocb2002/hm3d-semseg-data/generated` | `/workspace/data` |
| Runs | `/home/joaocb2002/hm3d-semseg-data/runs` | `/workspace/runs` |
| Model cache | `/home/joaocb2002/hm3d-semseg-data/cache` | `/workspace/cache` |
| Server logs | not applicable | `/workspace/logs` |

Generic commands retain placeholders so the repository remains reusable. Every
such command is followed immediately by the concrete form for the current
workstation/server layout. A value that does not exist yet, such as the final
run name, is explicitly labeled **planned** rather than presented as an existing
artifact.

## References

- [Losses, metrics, model selection, and artifacts](reference_losses_metrics_and_artifacts.md)
- [Testing and quality gates](reference_testing.md)
- [Troubleshooting](reference_troubleshooting.md)
- [CLI reference](reference_cli.md)

Generated datasets, weights, caches, and runs remain outside Git. Scientific
runs are tied to the Git commit recorded in their `provenance.json`; editing
these Markdown guides does not alter an already running server process.
