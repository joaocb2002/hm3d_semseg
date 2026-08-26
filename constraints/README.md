# Dependency constraints

`training.txt` pins the tested hardware-independent Python 3.10 training stack.
The host-aware installer applies it automatically. PyTorch and torchvision are
excluded because `install-training-env` selects and pins their exact CPU/CUDA
builds from the detected driver and GPU capability.

`quality.txt` pins repository-only lint, typing, and test tools for local checks
and CI. The host-aware installer also applies it unless `--without-dev` is used.
The project metadata retains compatible version ranges for ordinary editable
installs; these files make protocol runs and quality gates repeatable.

Update a pin only in a dedicated change that runs `pip check`, unit tests, Ruff,
Mypy, a real GPU kernel probe, and the end-to-end smoke workflow. The resolved
package versions are also stored in every run's `provenance.json`.
