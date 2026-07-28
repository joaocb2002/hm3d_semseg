# Frozen HM3D-Semantics v0.2 scene splits

These lists were generated from the complete local descriptor audits with the
repository's exact SHA-256 ranking and seed 2027.

- `fit.txt`: 130 of the 145 official training scenes.
- `development.txt`: the remaining 15 official training scenes.
- `calibration_fit.txt`: 12 of the 36 official validation scenes, used only to
  fit scalar temperature.
- `calibration_evaluation.txt`: the other 24 official validation scenes, used
  for calibration metrics.

The paired lists are disjoint. Regenerate them only deliberately with
`make-dev-split` or `make-calibration-split`, review coverage, and commit the
changed protocol as a research decision.

The descriptor audit used mapping SHA-256
`36e40c25cbe32c8bf34ef55f199f194671045106914dd09b1581aeedcf051a05`.
Both development and calibration partitions contain object-instance support for
all 41 project output classes; pixel support must still be rechecked after the
fixed rendering manifests are generated.

List-file SHA-256 values:

- `fit.txt`: `f35a0fc14b1a6ae8449812a62081b3ab969ed96f808a1dc5fb827bd1cd88447e`
- `development.txt`: `6084d8d2bf1c6f420059ab10c7fee5bb7c87f5fb279b8b392e4ccaf786f971fa`
- `calibration_fit.txt`: `493d333fa677f21771bb971fe7f62ae8bfb3bc5c49e7c8f10a974eeaf4241dd1`
- `calibration_evaluation.txt`: `2e340f1c63ea6ad911628ed5a654f12abed532dbe0f3efbda16e6e3541ce5068`
