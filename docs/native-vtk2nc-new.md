# Native `vtk2nc_new` compatibility contract

`vtk2nc_new` is a C++20 port of `core/tools_core/vtk2nc_new.py`. It is available
through `luw vtk2nc_new <case.luw>` and the `vtk2nc_new` compatibility wrapper;
neither entry starts Python.

## Inputs and discovery

The command requires `casename`, `cut_lon_manual`, and `cut_lat_manual` in the
deck. `datetime`, `utm_crs`, and `rotate_deg` remain optional. A missing UTM CRS
is inferred from the centre of the configured geographic bounds, and a missing
rotation is recomputed from their lower edge.

Discovery first checks `proj_temp/vtk/*.vtk`. When a deck datetime is present,
matching names are preferred and the original warning/fallback to all files is
retained. `RESULTS/*.vtk` is used only when the first location supplies no
candidate. Files are resolved, de-duplicated, and processed in deterministic
order.

The strict production path accepts legacy BINARY `STRUCTURED_POINTS` with
`SCALARS` arrays using all types supported by the former script: signed and
unsigned 8-, 16-, and 32-bit integers (`long` remains legacy 32-bit), float32,
and float64. The shared reader also permits ASCII files as an additive
compatibility extension; this does not change BINARY parsing or output.

## Coordinates and interpolation

The native transform follows the same chain as the Python implementation:

1. project the four geographic configuration corners into UTM;
2. rotate them around their mean pivot and use the rotated minima as the local
   origin;
3. derive the largest complete axis-aligned longitude/latitude bounds from all
   four source-grid edges;
4. build a regular target grid, retrying with the same margins and 12-attempt
   limit until every target point maps inside the source domain;
5. interpolate every field component and height level with cubic B-splines.

The cubic implementation reproduces
`scipy.ndimage.map_coordinates(order=3, mode="nearest", prefilter=True)`: it
uses the same 12-cell nearest-edge pre-padding, float64 separable spline
prefilter, and float32 output conversion. Work is bounded to at most 12 OpenMP
threads and NetCDF writes remain ordered by height.

## Output contract

Each input `<name>.vtk` produces `RESULTS/<name>.nc` with dimensions `height`,
`lat`, and `lon`. Coordinate types are float32/float64/float64. Three-component
main velocity fields map to `u`, `v`, and `w`; scalar, secondary vector, and
arbitrary multi-component names follow the original sanitising, collision, and
`data`/`rho` rules.

Coordinate and data variables use shuffle plus zlib level 3. Data chunks are
`(1, min(ny, 256), min(nx, 256))` and their fill value is float32 NaN. Variable
metadata and the global source/CRS/rotation/pivot/origin/note attributes match
the legacy schema.

Configuration and structural validation failures stop immediately with the
legacy error code. Ordinary per-file I/O, projection, or NetCDF failures are
reported, later files continue, and a mixed batch ends with status 2 after the
success/failure summary.
