"""
cdfInspect.py
A self-contained CLI tool that inspects a target NetCDF file and reports an overview,
coordinate ranges and variable metadata.
"""

import os
import sys
import glob
import warnings
import xarray as xr

def resolve_nc_path(input_path: str) -> str:
    """
    Resolve the actual NetCDF path according to the rules described above.
    Returns the resolved .nc path if successful, otherwise raises RuntimeError.
    """
    input_path = os.path.abspath(input_path)

    # Case A: the input is itself a .nc path
    if input_path.lower().endswith(".nc"):
        if os.path.isfile(input_path):
            return input_path
        raise RuntimeError(f"Input .nc does not exist: {input_path}")

    # Case B: the input has a non-.nc extension
    if not os.path.isfile(input_path):
        raise RuntimeError(f"Input file does not exist: {input_path}")

    parent_dir = os.path.dirname(input_path)
    bc_dir = os.path.join(parent_dir, "wind_bc")

    if not os.path.isdir(bc_dir):
        raise RuntimeError(f"Derived folder 'wind_bc' does not exist: {bc_dir}")

    nc_candidates = sorted(glob.glob(os.path.join(bc_dir, "*.nc")))
    if len(nc_candidates) == 0:
        raise RuntimeError(f"No .nc files found in folder: {bc_dir}")
    if len(nc_candidates) > 1:
        warnings.warn(
            "Multiple .nc files found under 'wind_bc'. "
            "The first NetCDF in alphabetical order will be used."
        )
        print("All detected NetCDF files:")
        for p in nc_candidates:
            print(f"  {os.path.basename(p)}")

    chosen = nc_candidates[0]
    print(f"Using NetCDF: {chosen}")
    return chosen


def print_dataset_info(ds: xr.Dataset) -> None:
    """
    Print basic info and detailed variable info for the dataset.
    """
    print("\n===== Dataset Overview =====")
    # The __repr__ of an xarray Dataset already shows dimensions, coordinates and variables
    print(ds)

    # Coordinate ranges for common longitude and latitude names
    coord_candidates = ["lon", "longitude", "lat", "latitude"]
    printed_any = False
    for coord_name in coord_candidates:
        if coord_name in ds.variables:
            try:
                data = ds[coord_name].values
                data_min = float(data.min())
                data_max = float(data.max())
                print(f"\n*** {coord_name} range: min = {data_min}, max = {data_max}")
                printed_any = True
            except Exception as e:
                print(f"\n*** Failed to compute range for '{coord_name}': {e}")
    if not printed_any:
        print("\nNo standard lon/lat coordinate variable was found in the dataset.")

    print("\n===== Variable Details =====")
    for var in ds.data_vars:
        da = ds[var]
        print(f"\nVariable: {var}")
        print(f"  Dimensions: {da.dims}")
        print(f"  Shape: {tuple(da.sizes[d] for d in da.dims)}")
        print(f"  Data type: {da.dtype}")
        try:
            preview = da.values.ravel()[:2]
            print(f"  Preview first two values: {preview}")
        except Exception:
            print("  Unable to preview data values.")
        if da.attrs:
            for attr_name, attr_val in da.attrs.items():
                print(f"  Attribute {attr_name}: {attr_val}")

    if ds.attrs:
        print("\n===== Global Attributes =====")
        for attr_name, attr_val in ds.attrs.items():
            print(f"{attr_name}: {attr_val}")


def main() -> None:
    # Enforce exactly one argument which is the input file path
    if len(sys.argv) != 2:
        script = os.path.basename(sys.argv[0])
        print(f"Usage: python {script} <input_file_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    try:
        nc_path = resolve_nc_path(input_path)
        print("========================================")
        print("Loading NetCDF")
        print(f"Path: {nc_path}")
        print("========================================")
        with xr.open_dataset(nc_path) as ds:
            print_dataset_info(ds)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
