Installation:

## 1. Install GPU Drivers and OpenCL Runtime


- **Windows**
  <details><summary>GPUs</summary>

  - Download and install the [AMD](https://www.amd.com/en/support/download/drivers.html)/[Intel](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)/[Nvidia](https://www.nvidia.com/Download/index.aspx) GPU Drivers, which contain the OpenCL Runtime.
  - Reboot.

  </details>
  <details><summary>CPUs</summary>

  - Download and install the [Intel CPU Runtime for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html) (works for both AMD/Intel CPUs).
  - Reboot.

  </details>
- **Linux**
  <details><summary>AMD GPUs</summary>

  - Download and install [AMD GPU Drivers](https://www.amd.com/en/support/download/linux-drivers.html), which contain the OpenCL Runtime, with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev
    mkdir -p ~/amdgpu
    wget -P ~/amdgpu https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
    sudo apt install -y ~/amdgpu/amdgpu-install*.deb
    sudo amdgpu-install -y --usecase=graphics,rocm,opencl --opencl=rocr
    sudo usermod -a -G render,video $(whoami)
    rm -r ~/amdgpu
    sudo shutdown -r now
    ```

  </details>
  <details><summary>Intel GPUs</summary>

  - Intel GPU Drivers come already installed since Linux Kernel 6.2, but they don't contain the OpenCL Runtime.
  - The the [OpenCL Runtime](https://github.com/intel/compute-runtime/releases) has to be installed separately with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev intel-opencl-icd
    sudo usermod -a -G render $(whoami)
    sudo shutdown -r now
    ```

  </details>
  <details><summary>Nvidia GPUs</summary>

  - Download and install [Nvidia GPU Drivers](https://www.nvidia.com/Download/index.aspx), which contain the OpenCL Runtime, with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev nvidia-driver-580
    sudo shutdown -r now
    ```

  </details>
  <details><summary>CPUs</summary>

  - Option 1: Download and install the [oneAPI DPC++ Compiler](https://github.com/intel/llvm/releases?q=%22oneAPI+DPC%2B%2B+Compiler+dependencies%22) and [oneTBB](https://github.com/uxlfoundation/oneTBB/releases) with:
    ```bash
    export OCLV="oclcpuexp-2025.20.6.0.04_224945_rel"
    export TBBV="oneapi-tbb-2022.2.0"
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev
    sudo mkdir -p ~/cpurt /opt/intel/${OCLV} /etc/OpenCL/vendors /etc/ld.so.conf.d
    sudo wget -P ~/cpurt https://github.com/intel/llvm/releases/download/2025-WW27/${OCLV}.tar.gz
    sudo wget -P ~/cpurt https://github.com/uxlfoundation/oneTBB/releases/download/v2022.2.0/${TBBV}-lin.tgz
    sudo tar -zxvf ~/cpurt/${OCLV}.tar.gz -C /opt/intel/${OCLV}
    sudo tar -zxvf ~/cpurt/${TBBV}-lin.tgz -C /opt/intel
    echo /opt/intel/${OCLV}/x64/libintelocl.so | sudo tee /etc/OpenCL/vendors/intel_expcpu.icd
    echo /opt/intel/${OCLV}/x64 | sudo tee /etc/ld.so.conf.d/libintelopenclexp.conf
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbb.so /opt/intel/${OCLV}/x64
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbbmalloc.so /opt/intel/${OCLV}/x64
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbb.so.12 /opt/intel/${OCLV}/x64
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbbmalloc.so.2 /opt/intel/${OCLV}/x64
    sudo ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
    sudo rm -r ~/cpurt
    ```
  - Option 2: Download and install [PoCL](https://portablecl.org/) with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev pocl-opencl-icd
    ```
  </details>

- **Android**
  <details><summary>ARM GPUs</summary>

  - Download the [Termux `.apk`](https://github.com/termux/termux-app/releases) and install it.
  - In the Termux app, run:
    ```bash
    apt update && apt upgrade -y
    apt install -y clang git make
    ```

  </details>

<br>


Usage:

The $project_path$ mentioned below is the parent folder of deck file (*.luw, the configuration file). You can actually place it anywhere you would like, and name the folder as you like (casename is suggested).

Project folder $project_path$ includes: xxx.luw (configuration deck), building_db (folder for building database), terrain_db (folder for terrain database, under development), wind_bc (netcdf boundary data), proj_temp (intermedia files), and RESULTS folder for output.

Commands:

The following commands are provided by this package. The deck file could be automatically detected, if "conf.luw" exists in current folder of console.

"luwbc <path-to-deck>" to build the boundary condition. The netcdf file (.nc) should be placed in $project_path$/wind_bc, with the name of "casename_datetime.nc". If the netcdf file name is incorrect, the only .nc file in the wind_bc folder will be treated as input, which is not recommended. It is supposed to be called for each datetime code under batch mode.

"luwcut <path-to-deck>" to cut out the building data within the given lat-lon range, indicated in the deck. The input is $project_path$/building_db. Only one *.shp file is allowed to be in this folder, and it is insensitive to the *.shp file name.

"luwvox <path-to-deck>" to voxelize the buildings (build the *.stl 3D file), preparing for CFD, based on the cutted *.shp file by a previous run of luwcut.

"shpinspect <path-to-deck or path-to-shp>" inspect the metadata of input *.shp file (buildings database), or the *.shp file pointed by the deck.

"cdfinspect <path-to-deck or path-to-nc>" inspect the metadata of input *.nc netcdf file, or the *.nc file pointed by the deck.

"luwval <path-to-deck>" run post alignment degree check and fulfill mandatory fields to deck, ensure it is suitable for cfd run.

"makeluw <path-to-deck>" globally run inspects, build the bc, voxelize the buildings, in one-click, and dump the logfile at the same time.

"cleanluw <path-to-deck>" will clean the autogenerated deck parameters and deleted the temp files, preparing for the next run of makeluw.

"transluw <path-to-deck> <(optional tailname)>" will perform post-computation transformation on the raw cfd results, recover the coordinates to orginal.

"visluw" translate vtk file from utm to lat/lon, save to *.npz and *.nc, and generate sectional views of velocity.

(Under development) "visluw <path-to-vtk>" will compute vorticity and, if possible, pressure, from a vtk result file.  