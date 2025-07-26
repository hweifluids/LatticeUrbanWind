"""
tool_cdfInspect.py
"""
import os
import sys
import glob
import xarray as xr


def get_nc_path():
    """
    NC file checker by Huanxia, V0714.
    """

    # 1) Prompt user to input the case name
    if len(sys.argv) == 2:              # input parameter: python load_shp.py my_case
        user_input = sys.argv[1]
    elif len(sys.argv) == 1:            # no parameter, send to interaction
        user_input = input("Enter case name or .nc filename: ").strip()
    else:                               # too many params, exit
        script = os.path.basename(sys.argv[0])
        print(f"Usage: {script} [case_name]")
        sys.exit(1)

    #user_input = input("Enter case name or .nc filename: ").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If input ends with .nc, treat it as a filename in the current script directory
    if user_input.lower().endswith('.nc'):
        nc_path = os.path.join(script_dir, user_input)
        if not os.path.isfile(nc_path):
            sys.exit(f"Error: File '{user_input}' not found in script directory.")
        return nc_path

    # Otherwise, treat input as a case name under ../wrfInput
    parent_dir = os.path.dirname(script_dir)
    wrf_input_dir = os.path.join(parent_dir, 'wrfInput')
    case_dir = os.path.join(wrf_input_dir, user_input)
    if not os.path.isdir(case_dir):
        sys.exit(f"Error: Case directory '{case_dir}' does not exist.")

    # Find .nc files in the case directory
    nc_files = glob.glob(os.path.join(case_dir, '*.nc'))
    if not nc_files:
        sys.exit(f"Error: No .nc files found in '{case_dir}'.")
    if len(nc_files) > 1:
        sys.exit(f"Error: Multiple .nc files found in '{case_dir}'. Please ensure only one .nc file is present.")

    return nc_files[0]


def print_dataset_info(ds):
    """
    Print basic info and detailed variable info for the dataset.
    """
    print("\n===== Dataset Overview =====")
    print(ds)
    # === New Code: Print lon/lat min/max ===
    for coord_name in ['lon', 'latitude', 'lat', 'longitude']:
        if coord_name in ds.variables:
            data = ds[coord_name].values
            print(f"\n*** {coord_name} range: min = {data.min()}, max = {data.max()}")

    print("\n===== Variable Details =====")
    for var in ds.data_vars:
        da = ds[var]
        print(f"\nVariable: {var}")
        print(f"  Dimensions: {da.dims}")
        print(f"  Shape: {da.shape}")
        print(f"  Data type: {da.dtype}")
        try:
            preview = da.values.flatten()[:2]
            print(f"  Preview first two values: {preview}")
        except Exception:
            print("  Unable to preview data.")
        for attr_name, attr_val in da.attrs.items():
            print(f"  Attribute - {attr_name}: {attr_val}")

    print("\n===== Global Attributes =====")
    for attr_name, attr_val in ds.attrs.items():
        print(f"{attr_name}: {attr_val}")

def main():
    nc_path = get_nc_path()
    ds = xr.open_dataset(nc_path)
    print_dataset_info(ds)
    ds.close()


if __name__ == '__main__':
    main()
