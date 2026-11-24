# Native `season_average` compatibility contract

`season_average` is implemented entirely in C++ and is available through both
`luw season_average` and the compatibility wrapper. It does not start Python or
import NumPy/Matplotlib.

## Inputs and selection

- Accepts `.luw`, `.luwdg`, and `.luwpf` decks.
- Reads `wind_bc/profile.dat` and requires exactly one
  `wind_bc/windrose_*m.csv`.
- Windrose CSV decoding follows the legacy order: UTF-8 with optional BOM,
  UTF-8, GBK/CP936, then Latin-1. Direction aliases, percentage/fraction
  detection, finite and infinite speed-bin targets, probability normalization,
  and profile interpolation/extrapolation retain the legacy formulas.
- `--vtk-dir` may be absolute or relative to the case directory. Without it,
  `RESULTS/crop/cropped_vtk` is preferred over `RESULTS/vtk`. Cropped files,
  then the largest average step, then modification time and filename determine
  the preferred file for a direction.
- Legacy `STRUCTURED_POINTS` inputs may use a vector field or independent
  `u/v/w` scalars and a recognized TKE scalar. Both BINARY and ASCII inputs are
  accepted by the shared native reader.

## Numerical behavior and outputs

For every direction and speed bin, velocity uses the probability-weighted
speed ratio while TKE uses the squared ratio. Accumulation is performed in
`float32`, one z plane at a time, matching the legacy computation order.
Velocity magnitude is weighted per directional source field rather than being
recomputed from the final mean vector.

The command writes:

- `RESULTS/season_average/season_average.vtk`, with `u_avg`, `u`, `v`, `w`,
  `vm`, and `tke`;
- `RESULTS/season_average/season_average_summary.txt`;
- unless `--skip-figures` is used, five `season_average_*_9layers.png` figures.

`--output-spacing` accepts one value or three values. Grid endpoints are kept
fixed, so the actual spacing is recomputed from the integer output dimensions,
and interpolation uses the same z-then-x/y `float32` order as the legacy tool.

## Large-volume strategy

The implementation uses reusable open VTK streams, plane-bounded input memory,
OpenMP point loops, and chunked endian conversion. When the five accumulator
volumes exceed 512 MiB, they are backed by memory-mapped temporary files under
the output directory. Resampled component caches are always disk-backed. The
temporary directories are owned by the command and are cleaned only after the
mapped files close.

PNG figures retain the legacy filenames, 18-by-16-inch pixel dimensions at the
configured DPI (with the same 80-DPI minimum), 3-by-3 target-height layout,
2nd/98th-percentile color limits, symmetric component scales, and field-specific
color-map semantics. The native rasterizer does not use Matplotlib, so PNG
compression bytes, font glyphs, and exact anti-aliasing are intentionally not
byte-identical to legacy figures.
