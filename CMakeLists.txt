cmake_minimum_required(VERSION 3.24)

project(LatticeUrbanWindNative VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(LUW_BUILD_GUI "Build LUW Studio" ON)
option(LUW_BUILD_TESTS "Build native compatibility tests" ON)
option(LUW_ENABLE_GEOSPATIAL "Build GDAL/OGR-backed GIS commands" ON)
option(LUW_ENABLE_FLUIDX3D
    "Expose and install the separate optional FluidX3D backend" OFF)
option(LUW_REQUIRE_FULL_DEPENDENCIES "Fail configure when an enabled production dependency is missing" ON)
option(LUW_INSTALL_RUNTIME_DEPENDENCIES
    "Bundle non-system runtime libraries and Qt plugins during installation" ON)
set(LUW_RUNTIME_SEARCH_DIRECTORIES "" CACHE STRING
    "Additional directories searched while collecting install-time runtime libraries")
set(LUW_GDAL_DATA_DIRECTORY "" CACHE PATH
    "Directory containing GDAL runtime data (gdalvrt.xsd)")
set(LUW_PROJ_DATA_DIRECTORY "" CACHE PATH
    "Directory containing the PROJ runtime database (proj.db)")
set(LUW_VTK_PROJ_DATA_DIRECTORY "" CACHE PATH
    "Directory containing VTK's bundled PROJ runtime database (proj.db)")

if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LuwDependencies)

add_subdirectory(src/common)
add_subdirectory(src/post)
add_subdirectory(src/cli)

if(LUW_BUILD_GUI)
    add_subdirectory(gui)
endif()

include(LuwInstallRuntime)
luw_configure_install()

if(LUW_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests/native)
    add_test(
        NAME native.python_free_tree
        COMMAND "${CMAKE_COMMAND}"
            "-DLUW_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyPythonFree.cmake"
    )
endif()

# CFD solvers are independently built native modules.  The default product
# source view contains CULBM only; expose FluidX3D build entry points solely
# when that separate optional backend is requested explicitly.
set(LUW_CFD_CORE_SOURCE_FILES
    core/cfd_core/CULBM/CMakeLists.txt
    core/cfd_core/CULBM/build.ps1
    core/cfd_core/CULBM/build.sh)
if(LUW_ENABLE_FLUIDX3D)
    list(APPEND LUW_CFD_CORE_SOURCE_FILES
        core/cfd_core/FluidX3D/FluidX3D.vcxproj
        core/cfd_core/FluidX3D/makefile
        core/cfd_core/FluidX3D/make.sh)
endif()
add_custom_target(cfd_core_sources
    SOURCES ${LUW_CFD_CORE_SOURCE_FILES}
)
