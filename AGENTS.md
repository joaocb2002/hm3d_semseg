# Repository conventions

This repository is standalone. Never edit sibling `habitat-lab` or `object-nav-v2`
repositories, add them to `sys.path`, or commit generated datasets and weights.

- Support Python 3.9 because the local Habitat environment uses it.
- Keep Habitat and PyTorch/Transformers imports lazy so unit tests need neither.
- Treat the composed ObjectNav configuration as the camera source of truth.
- Keep `unknown` (model ID 0) distinct from `ignore_index` (target value 255).
- Apply every geometric transform identically to RGB and masks; masks use nearest
  interpolation.
- Do not download assets at import time or in unit tests.
- Preserve output manifests and provenance. Do not overwrite incompatible datasets.

Safe fast checks:

```bash
python -m pytest -m unit
ruff check .
```

Optional local checks:

```bash
python -m pytest -m habitat
python -m pytest -m smoke
```

