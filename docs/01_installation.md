# 1. Installation

This step installs one standalone CLI without changing Habitat or ObjectNav.

## Render environment

The current `habitat` environment already pins Habitat-Lab/Sim 0.3.3. Preserve
those packages:

```bash
conda activate habitat
cd ~/projects/hm3d-semseg
python -m pip install -e ".[dev,render]"
hm3d-semseg --help
```

Success means the CLI help lists all subcommands and
`python -c "import habitat, habitat_sim"` succeeds. If pip proposes replacing
Habitat, CUDA, or PyTorch, stop and install only the missing dev/render packages
individually.

## Training environment

Create a separate environment instead of mutating Habitat:

```bash
conda create -n hm3d-semseg-train python=3.10
conda activate hm3d-semseg-train
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg
python -m pip install -e .
```

The Conda variable persists after the next activation; the export protects this
first session immediately. It prevents packages under `~/.local` from leaking
into the environment.

```bash
hm3d-semseg install-training-env
```

This is a read-only plan. It queries `nvidia-smi`, records every visible GPU's
compute capability and memory, inspects the existing PyTorch build in an isolated
process, and selects:

| Host | Selected profile |
|---|---|
| no NVIDIA GPU, or explicit `--profile cpu` | PyTorch 2.13 CPU |
| NVIDIA CC 6.x with a driver supporting CUDA 12.6+ | PyTorch 2.13 CUDA 12.6 |
| NVIDIA CC 7.5+ with a driver supporting CUDA 13.0+ | PyTorch 2.13 CUDA 13.0 |
| supported NVIDIA GPU with driver support from CUDA 12.6 to 12.x | PyTorch 2.13 CUDA 12.6 |
| supported NVIDIA GPU with driver support from CUDA 11.8 to 12.5 | PyTorch 2.7.1 CUDA 11.8 |

CUDA 13 removed Pascal offline compilation and library support, so the GTX 1070
(CC 6.1) deliberately receives CUDA 12.6 even when its driver advertises CUDA
13. Review the printed commands, then apply and verify:

```bash
hm3d-semseg install-training-env --apply --run-tests
```

The command installs PyTorch from the selected official wheel index before the
hardware-independent `train` extra, runs `pip check`, runs the unit suite when
requested, and executes a real matrix operation on every visible CUDA device.
An existing wrong wheel is replaced automatically. Success requires both the
expected CUDA build and at least one successful GPU kernel; CUDA availability
alone is insufficient.

Useful explicit modes:

```bash
# Laptop checks without CUDA
hm3d-semseg install-training-env --profile cpu --apply --run-tests

# Provision on a login node for a different CUDA 13 worker
hm3d-semseg install-training-env \
  --profile cu130 \
  --allow-unsupported-host
```

The second example remains a dry run until `--apply` is added. Bypassing host
validation is intended only when the installation host is not the execution
host. On a GPU execution host, always use `auto`. CPU training is valid for
small checks but not practical for the full baseline.

## Downstream inference environment

After a trained checkpoint returns from the server, install the minimal
inference extra in the ObjectNav/Habitat environment:

```bash
conda activate habitat
cd ~/projects/hm3d-semseg
python -m pip install -e ".[inference]"
```

This adds the checkpoint-loading dependencies but does not install or replace
PyTorch. If pip proposes changing Habitat, Habitat-Sim, CUDA, or PyTorch, stop
and resolve the environment before integration. The training extra remains
isolated in `hm3d-semseg-train`.

Profile versions and indexes follow the official
[PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/);
the Pascal boundary follows NVIDIA's
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

Next: [configure absolute paths](02_paths_and_configuration.md).
