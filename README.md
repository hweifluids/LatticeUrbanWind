# LatticeUrbanWind (LUW)

**Meter-level, hundred-KM-scale, Minute-computation, Billion-grid Winds - Supporting next-gen real-time low-altitude applications**

LatticeUrbanWind (LUW) is a native C++20 workflow toolkit for fast urban wind simulation. It couples mesoscale wind fields (for example WRF NetCDF or data from national centers) to lattice Boltzmann CFD and provides data inspection, boundary-condition construction, geometry preparation, voxelization, solver orchestration, and postprocessing. The production CLI, preprocessing, postprocessing, and Qt GUI do not start or embed a Python runtime.
On the other hand, it also supports the fast generation of wind field dataset (also considering terrain and buildings) as well as profile-based inflow simulations for research purposes, i.e., pre-defined batch run.

The application potentials include but not limited to: urban air mobility (UAM), buidling ventilation, pedestrian wind environment, machine learning prediction of wind fields, real-time low-altitude support of events, observation data assimilation.

LUW supports three run modes with text-based configuration files and reproducible project layouts:

- **NWP-LBMLES** (standard `*.luw`)
- **Dataset generation** (`*.luwdg`)
- **Profile research** (`*.luwpf`)

Just leave terrain and building dataset in the correct folder, edit the configuration to setup the case, and enjoy the wind field results in shorter than a cup of coffee time.

The active default solver is **CULBM**, the CUDA LBM module in this repository. **FluidX3D** is retained as a separate, frozen OpenCL backend that is exposed only by explicit opt-in.

It is designed for both Windows and Linux with GPU acceleration.

## Highlights

- End-to-end native C++20 pipeline: NetCDF + GIS + terrain to CFD-ready domain.
- Native command-line tools and Qt GUI for inspection, preprocessing, voxelization, orchestration, and postprocessing.
- Three run modes: NWP-LBMLES, Dataset generation, and Profile research.
- Built-in helpers for postprocessing and exporting results to VTK/NetCDF.
- CULBM is the default CUDA backend; frozen FluidX3D is disabled by default and remains a separate explicit OpenCL option.
- Optional CULBM SurfaceIBM support for static hydrodynamic surfaces; it is disabled by default.

## System Requirements

- OS: Windows 10/11 or Linux.
- CMake 3.24+ and a C++20 compiler.
- Qt 6.8+, VTK 9.x, GDAL/OGR/GEOS, NetCDF, and their native development libraries for the CLI and GUI.
- CUDA Toolkit 12.0+ and a compatible NVIDIA driver for the default CULBM backend.
- An OpenCL runtime and suitable device driver for optional preprocessing acceleration; it is also required when the separate FluidX3D backend is enabled.
- Visual Studio C++ Build Tools on Windows, or GCC/Clang and the usual native build tools on Linux.

The native dependency manifest is [`vcpkg.json`](vcpkg.json). If its packages
are not installed system-wide, set `VCPKG_ROOT` to a bootstrapped vcpkg checkout
before running the installer. Set `LUW_CMAKE_PREFIX_PATH` to a quoted,
semicolon-separated CMake list of Qt and VTK installation prefixes, or use the
standard `CMAKE_PREFIX_PATH` environment variable with its platform-native list
syntax. The Windows stage-2 script also detects a conventional
`C:\Qt\*\msvc*_64` prefix and a sibling `deps\VTK-install` directory. It prints
the final prefixes and vcpkg root before configuration, and CMake stops with an
explicit missing-package error rather than silently disabling production paths.

For example, in a Windows CMD session with an existing vcpkg checkout:

```bat
set VCPKG_ROOT=C:\tools\vcpkg
set LUW_CMAKE_PREFIX_PATH=C:\Qt\6.8.0\msvc2022_64;C:\native\VTK-install
install_win.cmd
```

Python is not a production dependency. Four Python scripts retained under `core/cfd_core/CULBM/` are optional, offline validation/reporting utilities only; the installer, `luw`, `runluw`, and LUW Studio never invoke them.

## Installation

### Windows

1. Open **PowerShell or CMD as Administrator**.
2. Run the installer orchestrator:

```bat
install_win.cmd
```

This runs the staged native installer scripts in `installer/`:

1. `0_detect_env.cmd` - checks the native toolchain and OpenCL runtime.
2. `1_env_var.cmd` - sets `LUW_HOME` and adds `%LUW_HOME%\bin` to `PATH`.
3. `2_build_native.cmd` - configures, builds, and installs the C++20 CLI and GUI.
4. `3_compile_cfdcore.cmd` - builds the default CULBM solver; FluidX3D is built only when `LUW_ENABLE_FLUIDX3D=ON`.
5. `4_testrun.cmd` - checks the installed launcher, GUI runtime, and CULBM; it checks FluidX3D only when explicitly enabled.

### Linux

1. Open a terminal.
2. Run the installer:

```bash
./install_linux.sh
```

This executes the numbered scripts in `installer/`:

1. `0_detect_env.sh` - checks the native toolchain and OpenCL runtime.
2. `1_env_var.sh` - sets `LUW_HOME` and updates `PATH` in shell profiles.
3. `2_build_native.sh` - configures, builds, and installs the C++20 CLI and GUI.
4. `3_compile_cfdcore.sh` - builds the default CULBM solver; FluidX3D is built only when `LUW_ENABLE_FLUIDX3D=ON`.
5. `3_detect_opencl.sh` - runs the native OpenCL device diagnostic.
6. `4_testrun.sh` - checks the installed launcher, GUI runtime, and CULBM; it checks FluidX3D only when explicitly enabled.

### Manual staged build

Set `LUW_HOME` to the repository root and add its `bin` directory to `PATH`, then run both build stages. Stage 2 builds the native CLI/GUI; stage 3 builds the self-contained CULBM package. Neither stage reads or builds FluidX3D by default.

Windows:

```bat
installer\2_build_native.cmd
installer\3_compile_cfdcore.cmd
```

Linux:

```bash
bash installer/2_build_native.sh
bash installer/3_compile_cfdcore.sh
```

To expose, package, build, and validate the separate FluidX3D backend as well,
set `LUW_ENABLE_FLUIDX3D=ON` before running the stages. For example:

```bat
set LUW_ENABLE_FLUIDX3D=ON
installer\2_build_native.cmd
installer\3_compile_cfdcore.cmd
```

```bash
LUW_ENABLE_FLUIDX3D=ON bash installer/2_build_native.sh
LUW_ENABLE_FLUIDX3D=ON bash installer/3_compile_cfdcore.sh
```

## Quick Start

### 1) Create a project folder

Copy the template:

```bash
cp -r project_template my_case
cd my_case
```

Or start from the example:

```bash
cp -r examples/example_ProfileResearch my_case
cd my_case
```

### 2) Prepare inputs

Required folders:

- `wind_bc/` - NetCDF wind files.
- `building_db/` - building footprints (`.shp`) or STL geometry.
- `terrain_db/` - DEM or terrain data (if used).

Wind file naming convention (important for time-dependent runs):

```
<casename>_yyyymmddhhmmss.nc
```

If the naming rule is not followed, the tools will fall back to the only `.nc` file found, and time-dependent batch runs will be disabled.

### 3) Edit a configuration file

Common configuration files:

- `conf.luw`   - NWP-LBMLES (standard mode)
- `conf.luwdg` - Dataset generation
- `conf.luwpf` - Profile research

Example (from `project_template/conf.luw`):

```txt
// Project
casename = example
datetime = 20251010120000

// Domain
cut_lon_manual = [121.3, 121.7]
cut_lat_manual = [31.1, 31.4]
base_height = 50
z_limit = 500
midmesh_basesize = 10

// CFD Controls
cfd_solver = "culbm"
ibm_enabler = false
n_gpu = [2, 1, 1]
mesh_control = "gpu_memory"
gpu_memory = 20000
cell_size =
high_order = true
flux_correction = true

// Physics
coriolis_term = true
```

Deck files are sectioned and order-insensitive. During parsing, common bool-like values such as `y/yes/t/true/1` and quoted variants are accepted and rewritten back to canonical `true/false`.

Solver selection is explicit and stable:

- A missing `cfd_solver` field, or `cfd_solver = "culbm"`, runs CULBM.
- `cfd_solver = "fluidx3d"` selects the frozen FluidX3D backend only when that separate executable was installed through explicit opt-in; otherwise startup fails without falling back to CULBM.
- SurfaceIBM is implemented only by CULBM. `ibm_enabler` defaults to `false`; setting it to `true` while selecting FluidX3D is rejected before solver launch.

### 4) Run the preprocessing pipeline

From inside your project folder:

```bash
makeluw conf.luw
```

`makeluw` runs these steps in order:

1. `cdfinspect` - inspect NetCDF wind data
2. `shpinspect` - inspect building shapefiles
3. `luwbc` - build boundary conditions
4. `luwcut` - crop/prepare building geometry
5. `luwvox` - voxelization
6. `luwval` - pre-run validation

### 5) Run CFD

On Windows or Linux:

```bash
runluw conf.luw
```

`runluw` reads `cfd_solver` from the deck, defaults to CULBM when the field is absent, locates the selected executable, and forwards the deck path and any remaining solver arguments. FluidX3D is used only when the deck explicitly selects it and its opt-in executable exists; a missing selected backend is an error and never causes solver fallback.

### 6) Postprocess (optional)

- Translate VTKs to a common origin:

```bash
transluw conf.luw
```

- Generate section plots and optional NetCDF:

```bash
visluw conf.luw
```

- Convert VTK outputs to NetCDF:

```bash
vtk2nc conf.luw
```

## Configuration File Types

| File type | Run mode | Typical use |
| --- | --- | --- |
| `.luw` | NWP-LBMLES | Full end-to-end CFD run |
| `.luwdg` | Dataset generation | Automated dataset creation |
| `.luwpf` | Profile research | Profile research workflow |

Key fields (common across modes):

- `casename` - case identifier (used in filenames)
- `datetime` - timestamp `yyyymmddhhmmss`
- `cut_lon_manual`, `cut_lat_manual` - geographic crop window
- `utm_crs` / `utm_epsg` - target projection
- `base_height`, `z_limit` - vertical domain setup
- `cfd_solver` - `culbm` (default, including when omitted) or explicitly `fluidx3d`
- `ibm_enabler` - optional CULBM SurfaceIBM master switch; default `false`
- `n_gpu`, `mesh_control`, `gpu_memory`, `cell_size` - CFD controls

## CLI Tools

| Command | Description |
| --- | --- |
| `cdfinspect` | Inspect NetCDF wind fields |
| `shpinspect` | Inspect building shapefiles and CRS |
| `luwbc` | Build wind boundary conditions |
| `luwcut` | Cut/crop geometry |
| `luwvox` | Voxelize geometry for CFD |
| `luwval` | Pre-run validation |
| `makeluw` | Run the full preprocessing chain |
| `runluw` | Run the deck-selected CFD backend (CULBM by default) |
| `transluw` | Translate VTK outputs to a common origin |
| `visluw` | Section plots + optional NetCDF export |
| `vtk2nc` | Convert VTK to NetCDF |
| `cleanluw` | Clean the temporary and result files of a LUW case |

## Python Boundary

The production source and runtime path are Python-free. The following four files are deliberately retained for offline CULBM validation, comparison, and report generation; they are not installed as production commands and are not called by the CLI, GUI, preprocessing, postprocessing, or solver launcher:

- `core/cfd_core/CULBM/tests/compare_backends.py`
- `core/cfd_core/CULBM/tests/compare_physical_vtk.py`
- `core/cfd_core/CULBM/validation/gea/analyze_results.py`
- `core/cfd_core/CULBM/validation/gea/report/build_report.py`

Using those optional analysis programs requires a separately managed Python environment, but ordinary LUW installation and operation do not.

## Project Layout

```
LatticeUrbanWind/
  bin/                native CLI and compatibility wrappers
  src/                C++20 CLI, orchestration, preprocessing, and postprocessing
  gui/                native Qt/VTK LUW Studio
  core/cfd_core/      active CULBM + separately opt-in FluidX3D
  installer/          staged native build and runtime checks
  examples/           working examples
  project_template/   starter project layout
  run/                local datasets and experiments (optional)
```

Inside a project folder:

```
my_case/
  building_db/        shapefiles or STL
  terrain_db/         DEM/terrain data
  wind_bc/            NetCDF wind files
  proj_temp/          intermediate files
  RESULTS/            outputs and VTKs
  conf.luw            case configuration
```

## Examples

Each run mode has a corresponding example under `examples/`. These examples are intended as complete, working references:

- **NWP-LBMLES** example (standard `*.luw`)
- **Dataset generation** example (`*.luwdg`)
- **Profile research** example (`*.luwpf`)

## Maintainers

- Huanxia Wei (Tongji University/The University of Manchester)
- Prof. Qing Jia (Tongji University)
- Dr. Shuai Han (NMIC China Meteorological Administration)

## Contributors

- Yue Cao (Tongji University/UC Berkeley)
- Shixiong Zhou (HKUSTGZ)
- Zheng Jian (Shensi Lab)
- Wenguang Liang (NMIC China Meteorological Administration)


## Contact

If you have questions, suggestions, or want to collaborate or contribute (then as a paper co-author), feel free to reach out (Huanxia Wei).

- Email: huanxia.wei@u.nus.edu
- WeChat: TJerCZ

## Acknowledgements

This project is developed with funding and support from the National Meteorological Center of China Meteorological Administration.

## License

LatticeUrbanWind is released under the custom terms in `LICENSE.md`. CULBM is independently licensed by `core/cfd_core/CULBM/LICENSE.md`; use or distribution of LatticeUrbanWind with CULBM must satisfy both licenses. If the separate optional FluidX3D backend is explicitly enabled, used or distributed, `core/cfd_core/FluidX3D/LICENSE.md` additionally applies.
