# LatticeUrbanWind (LUW)

LatticeUrbanWind (LUW) is a workflow toolkit for fast urban wind simulation that couples mesoscale wind fields (e.g., WRF NetCDF) with the FluidX3D lattice Boltzmann solver. It provides data inspection, boundary-condition building, geometry preparation, voxelization, and validation utilities so you can go from regional wind data and city geometry to a ready-to-run CFD case.

LUW supports three run modes with lightweight text-based configuration files and reproducible project layouts:

- **NWP-LBMLES** (standard `*.luw`)
- **Dataset generation** (`*.luwdg`)
- **Profile research** (`*.luwpf`)

It is designed for Windows and Linux, with GPU acceleration via OpenCL.

## Highlights

- End-to-end pipeline: NetCDF + GIS + terrain to CFD-ready domain.
- Command-line tools for inspection, preprocessing, voxelization, and validation.
- Three run modes: NWP-LBMLES, Dataset generation, and Profile research.
- Built-in helpers for postprocessing and exporting results to VTK/NetCDF.
- Compatible with multi-GPU FluidX3D runs.

## System Requirements

- OS: Windows 10/11 or Linux (tested via installer scripts).
- Python 3.x with `venv`.
- OpenCL runtime (GPU drivers for AMD/NVIDIA/Intel).
- Visual Studio Build Tools (Windows) or a C++ toolchain (Linux) to compile FluidX3D.

## Installation

### Windows (recommended)

1. Open **PowerShell or CMD as Administrator**.
2. Run the installer orchestrator:

```bat
install_win.cmd
```

This runs the staged scripts in `installer/`:

1. `0_detect_env.cmd` - checks Python, pip, OpenCL.
2. `1_env_var.cmd` - sets `LUW_HOME` and adds `%LUW_HOME%\bin` to PATH.
3. `2_setup_python_venv.cmd` - creates `.venv` and installs Python dependencies.
4. `3_compile_cfdcore.cmd` - builds FluidX3D.
5. `4_testrun.cmd` - placeholder (currently empty).

### Linux

1. Open a terminal.
2. Run the installer:

```bash
./install_linux.sh
```

This executes the numbered scripts in `installer/`:

1. `0_detect_env.sh` - checks Python, pip, OpenCL.
2. `1_env_var.sh` - sets `LUW_HOME` and updates PATH in shell profiles.
3. `2_setup_python.sh` - creates `.venv` and installs dependencies.
4. `3_compile_cfdcore.sh` - builds FluidX3D (calls `core/cfd_core/FluidX3D/make.sh`).

### Manual setup (if you prefer)

- Set `LUW_HOME` to the repo root and add `$LUW_HOME/bin` to PATH.
- Create a virtual environment and install:

```bash
python -m venv .venv
python -m pip install -r installer/requirements.txt
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
casename = example
datetime = 20251010120000
cut_lon_manual = [121.3, 121.7]
cut_lat_manual = [31.1, 31.4]
utm_crs = "EPSG:32651"
base_height = 50
z_limit = 500
n_gpu = [2, 1, 1]
mesh_control = "gpu_memory"
gpu_memory = 20000
```

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

### 5) Run FluidX3D

Windows:

```powershell
runluw.ps1 0
```

Linux:

```bash
cd core/cfd_core/FluidX3D
./make.sh 0
```

You can pass multiple GPU indices as needed (e.g., `0 1 2`).

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
| `transluw` | Translate VTK outputs to origin (**legacy**, will be removed in a future release) |
| `visluw` | Section plots + optional NetCDF export |
| `vtk2nc` | Convert VTK to NetCDF |
| `cleanluw` | Clean LUW text files |

## Project Layout

```
LatticeUrbanWind/
  bin/                CLI wrappers
  core/               main pipeline + FluidX3D
  installer/          installation scripts + requirements.txt
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

## Troubleshooting

- **`LUW_HOME is not set`**: ensure the environment variable is exported and `$LUW_HOME/bin` is on PATH.
- **OpenCL not found**: install the correct GPU driver package and verify with `clinfo` (Linux) or the Windows env check.
- **Python deps fail**: activate `.venv` and reinstall `installer/requirements.txt`.
- **Shapefile CRS issues**: ensure your building data is in EPSG:4326 or provide correct CRS metadata.

## Acknowledgements

This project is developed with cooperation and support from the National Meteorological Center of China Meteorological Administration.

## Contact

If you have questions, suggestions, or want to collaborate, feel free to reach out. I am Huanxia Wei.

- Email: huanxia.wei@u.nus.edu
- WeChat: TJerCZ

## License

This repository is released under a custom, non-commercial, non-military, no-AI-training license. Please read `LICENSE.md` carefully before use. FluidX3D is included under its own license and terms.
