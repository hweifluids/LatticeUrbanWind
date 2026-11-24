if(LUW_BUILD_GUI)
    find_package(Qt6 6.8 REQUIRED COMPONENTS Core Concurrent Gui Widgets OpenGL OpenGLWidgets)
    find_package(OpenGL REQUIRED)
else()
    find_package(Qt6 6.5 REQUIRED COMPONENTS Core Concurrent)
endif()

find_package(VTK 9.2 REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    CommonExecutionModel
    CommonMath
    CommonTransforms
    FiltersCore
    FiltersGeneral
    FiltersGeometry
    FiltersModeling
    FiltersPoints
    FiltersSources
    ImagingCore
    IOGeometry
    IOImage
    IOLegacy
    IONetCDF
    IOXML
    libproj
)

find_package(manifold CONFIG QUIET)
if(NOT TARGET manifold::manifold AND LUW_REQUIRE_FULL_DEPENDENCIES)
    message(FATAL_ERROR
        "Manifold is required for the native dgprepare boolean-union path. "
        "Install the manifest dependency, or configure an inspection-only "
        "build with -DLUW_REQUIRE_FULL_DEPENDENCIES=OFF; that build retains "
        "the legacy concatenation fallback.")
endif()

if(LUW_ENABLE_GEOSPATIAL)
    find_package(GDAL CONFIG QUIET)
    if(NOT GDAL_FOUND)
        find_package(GDAL QUIET)
    endif()
    if(NOT GDAL_FOUND AND LUW_REQUIRE_FULL_DEPENDENCIES)
        message(FATAL_ERROR
            "GDAL/OGR is required for the native GIS pipeline. Install GDAL "
            "with C++ headers, or configure an inspection-only build with "
            "-DLUW_ENABLE_GEOSPATIAL=OFF -DLUW_REQUIRE_FULL_DEPENDENCIES=OFF.")
    endif()

    find_package(JPEG QUIET)
    if((NOT JPEG_FOUND OR NOT TARGET JPEG::JPEG) AND
       LUW_REQUIRE_FULL_DEPENDENCIES)
        message(FATAL_ERROR
            "libjpeg-turbo is required for bounded-memory 4800-DPI luwcut "
            "preview output. Install libjpeg-turbo with C headers, or "
            "configure an inspection-only build with "
            "-DLUW_ENABLE_GEOSPATIAL=OFF -DLUW_REQUIRE_FULL_DEPENDENCIES=OFF.")
    endif()

    find_package(GEOS CONFIG QUIET)
    if(NOT TARGET GEOS::geos_c AND LUW_REQUIRE_FULL_DEPENDENCIES)
        message(FATAL_ERROR
            "GEOS C is required for Shapely-compatible geometry validity "
            "reasons in native shptester. Install GEOS, or configure an "
            "inspection-only build with -DLUW_ENABLE_GEOSPATIAL=OFF "
            "-DLUW_REQUIRE_FULL_DEPENDENCIES=OFF.")
    endif()

    find_path(LUW_EARCUT_INCLUDE_DIR NAMES mapbox/earcut.hpp)
    if(LUW_EARCUT_INCLUDE_DIR)
        add_library(luw_earcut_headers INTERFACE)
        add_library(luw::earcut ALIAS luw_earcut_headers)
        target_include_directories(luw_earcut_headers INTERFACE
            "${LUW_EARCUT_INCLUDE_DIR}"
        )
    elseif(LUW_REQUIRE_FULL_DEPENDENCIES)
        message(FATAL_ERROR
            "mapbox/earcut.hpp is required by native luwvox polygon extrusion. "
            "Install earcut-hpp (the vcpkg manifest dependency), or configure "
            "an inspection-only build with -DLUW_ENABLE_GEOSPATIAL=OFF "
            "-DLUW_REQUIRE_FULL_DEPENDENCIES=OFF.")
    endif()
    mark_as_advanced(LUW_EARCUT_INCLUDE_DIR)
endif()

find_package(OpenMP QUIET COMPONENTS CXX)
find_package(OpenCL QUIET)
find_package(Threads REQUIRED)
