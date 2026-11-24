set(LUW_POST_SOURCES
    legacy_vtk.cpp
    les_spectra_command.cpp
    tke_viz_core.cpp
    tke_viz_command.cpp
    transluw_command.cpp
    visluw_command.cpp
    vtk2nc_command.cpp
    vtk2nc_new_command.cpp
    vtk_avg_to_utm_asl_nc_command.cpp
)

if(MSVC)
    set_property(SOURCE vtk2nc_new_command.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:precise)
else()
    set_property(SOURCE vtk2nc_new_command.cpp APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()

# Keep the shared target configurable while a new post-processing command is
# being developed in the shared workspace.  Once both translation units are
# present they are included together, so the public command cannot be linked
# against a half-written implementation.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/season_average_command.cpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/season_average_figures.cpp")
    list(APPEND LUW_POST_SOURCES
        season_average_command.cpp
        season_average_figures.cpp
    )
    if(MSVC)
        set_property(SOURCE season_average_command.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:precise)
    else()
        set_property(SOURCE season_average_command.cpp APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
    endif()
endif()

add_library(luw_post STATIC ${LUW_POST_SOURCES})

# The production LES spectra backend uses the same header-only FFT family as
# SciPy and transforms whole axes/batches in one call.  Keep VTK's bundled
# kissfft as a dependency-light fallback for the ordinary no-vcpkg build.
find_path(LUW_POCKETFFT_INCLUDE_DIR NAMES pocketfft_hdronly.h)
if(LUW_POCKETFFT_INCLUDE_DIR)
    target_include_directories(luw_post SYSTEM PRIVATE ${LUW_POCKETFFT_INCLUDE_DIR})
    target_compile_definitions(luw_post PRIVATE LUW_HAS_POCKETFFT=1)
else()
    target_compile_definitions(luw_post PRIVATE LUW_HAS_POCKETFFT=0)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(luw_post PRIVATE OpenMP::OpenMP_CXX)
endif()

if(NOT WIN32)
    find_package(Iconv QUIET)
    if(Iconv_FOUND)
        target_link_libraries(luw_post PRIVATE Iconv::Iconv)
    endif()
endif()

target_include_directories(luw_post
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(luw_post
    PUBLIC
        luw_common
    PRIVATE
        VTK::FiltersCore
        VTK::IOLegacy
        VTK::IOImage
        VTK::CommonMath
        VTK::zlib
        VTK::netcdf
        VTK::libproj
)

if(MSVC)
    target_compile_options(luw_post PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_post PRIVATE -Wall -Wextra -Wpedantic)
endif()
