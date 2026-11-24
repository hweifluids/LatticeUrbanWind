add_library(luw_cli_dispatch STATIC
    command_registry.cpp
)
target_include_directories(luw_cli_dispatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(luw_cli_dispatch PUBLIC luw_common Qt6::Core)

set(LUW_CLI_SOURCES
    main.cpp
    builtin_commands.cpp
    commands/accelerator_report_command.cpp
    commands/detect_opencl_command.cpp
    commands/inspect_commands.cpp
    commands/shptester_command.cpp
    commands/visdem_command.cpp
    commands/cleanluw_command.cpp
    commands/dem_tif_cache.cpp
    commands/buildingscale_command.cpp
    commands/luwbc_command.cpp
    commands/luwbc_opencl.cpp
    commands/luwcut_command.cpp
    commands/luwvox_command.cpp
    commands/luwvox_opencl.cpp
    commands/validate_command.cpp
    commands/dgprepare_command.cpp
    commands/makeluw.cpp
    commands/runluw.cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/commands/deck_command.cpp")
    list(APPEND LUW_CLI_SOURCES commands/deck_command.cpp)
endif()

add_executable(luw ${LUW_CLI_SOURCES})
target_include_directories(luw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(luw PRIVATE
    luw_cli_dispatch
    luw_common
    luw_post
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::FiltersCore
    VTK::FiltersGeneral
    VTK::IOGeometry
    VTK::IOLegacy
    VTK::IOXML
    VTK::libproj
    VTK::netcdf
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(luw PRIVATE OpenMP::OpenMP_CXX)
endif()

if(OpenCL_FOUND)
    target_compile_definitions(luw PRIVATE LUW_HAS_OPENCL=1)
    target_link_libraries(luw PRIVATE OpenCL::OpenCL)
else()
    target_compile_definitions(luw PRIVATE LUW_HAS_OPENCL=0)
endif()

if(TARGET manifold::manifold)
    target_compile_definitions(luw PRIVATE LUW_HAS_MANIFOLD=1)
    target_link_libraries(luw PRIVATE manifold::manifold)
else()
    target_compile_definitions(luw PRIVATE LUW_HAS_MANIFOLD=0)
endif()

if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET luw::earcut)
    target_compile_definitions(luw PRIVATE LUW_HAS_GDAL=1)
    target_link_libraries(luw PRIVATE luw::earcut)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw PRIVATE LUW_HAS_GDAL=0)
endif()

if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND JPEG_FOUND AND
   TARGET JPEG::JPEG)
    target_compile_definitions(luw PRIVATE LUW_HAS_JPEG=1)
    target_link_libraries(luw PRIVATE JPEG::JPEG)
else()
    target_compile_definitions(luw PRIVATE LUW_HAS_JPEG=0)
endif()

if(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND AND TARGET GEOS::geos_c)
    target_compile_definitions(luw PRIVATE LUW_HAS_SHPTESTER_GIS=1)
    target_link_libraries(luw PRIVATE GEOS::geos_c)
    if(TARGET GDAL::GDAL)
        target_link_libraries(luw PRIVATE GDAL::GDAL)
    else()
        target_include_directories(luw PRIVATE ${GDAL_INCLUDE_DIRS})
        target_link_libraries(luw PRIVATE ${GDAL_LIBRARIES})
    endif()
else()
    target_compile_definitions(luw PRIVATE LUW_HAS_SHPTESTER_GIS=0)
endif()

if(COMMAND vtk_module_autoinit)
    vtk_module_autoinit(
        TARGETS luw
        MODULES
            VTK::CommonCore
            VTK::CommonDataModel
            VTK::FiltersCore
            VTK::FiltersGeneral
            VTK::IOGeometry
            VTK::IOLegacy
            VTK::IOXML
    )
endif()

if(MSVC)
    target_compile_options(luw_cli_dispatch PRIVATE /W4 /permissive- /utf-8)
    target_compile_options(luw PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_cli_dispatch PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(luw PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(UNIX AND NOT APPLE AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    # DT_RPATH is intentionally transitive: copied VTK/Qt libraries can find
    # their own copied dependencies without LD_LIBRARY_PATH.
    target_link_options(luw PRIVATE "LINKER:--disable-new-dtags")
endif()

# Compatibility command names are native copies of the same dispatcher.  A
# post-build copy works on Windows where symbolic links often require elevated
# privileges; source-controlled bin wrappers remain the install-time entry.
set(LUW_COMPAT_COMMANDS
    accelerator_report
    cdfinspect
    detect_opencl
    shpinspect
    shptester
    visdem
    luwbc
    luwcut
    luwvox
    luwval
    makeluw
    runluw
    visluw
    transluw
    vtk2nc
    vtk2nc_new
    cleanluw
    season_average
    les_spectra
    tke_viz
    cut_vis
    buildingscale
    batch_tke_geo_viz
    vtk_avg_to_utm_asl_nc
    dgprepare
)

foreach(command_name IN LISTS LUW_COMPAT_COMMANDS)
    add_custom_command(TARGET luw POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:luw>"
            "$<TARGET_FILE_DIR:luw>/${command_name}$<TARGET_FILE_SUFFIX:luw>"
        VERBATIM
    )
endforeach()

install(TARGETS luw
    RUNTIME_DEPENDENCY_SET luw_runtime_dependencies
    RUNTIME DESTINATION bin)

set(LUW_POSIX_WRAPPERS
    "${PROJECT_SOURCE_DIR}/bin/.luw-wrapper"
    "${PROJECT_SOURCE_DIR}/bin/accelerator_report"
    "${PROJECT_SOURCE_DIR}/bin/cdfinspect"
    "${PROJECT_SOURCE_DIR}/bin/detect_opencl"
    "${PROJECT_SOURCE_DIR}/bin/shpinspect"
    "${PROJECT_SOURCE_DIR}/bin/shptester"
    "${PROJECT_SOURCE_DIR}/bin/visdem"
    "${PROJECT_SOURCE_DIR}/bin/luwbc"
    "${PROJECT_SOURCE_DIR}/bin/luwcut"
    "${PROJECT_SOURCE_DIR}/bin/luwvox"
    "${PROJECT_SOURCE_DIR}/bin/luwval"
    "${PROJECT_SOURCE_DIR}/bin/makeluw"
    "${PROJECT_SOURCE_DIR}/bin/runluw"
    "${PROJECT_SOURCE_DIR}/bin/visluw"
    "${PROJECT_SOURCE_DIR}/bin/transluw"
    "${PROJECT_SOURCE_DIR}/bin/vtk2nc"
    "${PROJECT_SOURCE_DIR}/bin/vtk2nc_new"
    "${PROJECT_SOURCE_DIR}/bin/cleanluw"
    "${PROJECT_SOURCE_DIR}/bin/dgprepare"
    "${PROJECT_SOURCE_DIR}/bin/season_average"
    "${PROJECT_SOURCE_DIR}/bin/les_spectra"
    "${PROJECT_SOURCE_DIR}/bin/tke_viz"
    "${PROJECT_SOURCE_DIR}/bin/cut_vis"
    "${PROJECT_SOURCE_DIR}/bin/buildingscale"
    "${PROJECT_SOURCE_DIR}/bin/batch_tke_geo_viz"
    "${PROJECT_SOURCE_DIR}/bin/vtk_avg_to_utm_asl_nc"
)
install(PROGRAMS ${LUW_POSIX_WRAPPERS} DESTINATION bin)

set(LUW_POWERSHELL_WRAPPERS
    .luw-wrapper.ps1
    accelerator_report.ps1
    cdfinspect.ps1
    detect_opencl.ps1
    shpinspect.ps1
    shptester.ps1
    visdem.ps1
    luwbc.ps1
    luwcut.ps1
    luwvox.ps1
    luwval.ps1
    makeluw.ps1
    runluw.ps1
    visluw.ps1
    transluw.ps1
    vtk2nc.ps1
    vtk2nc_new.ps1
    cleanluw.ps1
    dgprepare.ps1
    season_average.ps1
    les_spectra.ps1
    tke_viz.ps1
    cut_vis.ps1
    buildingscale.ps1
    batch_tke_geo_viz.ps1
    vtk_avg_to_utm_asl_nc.ps1
)
list(TRANSFORM LUW_POWERSHELL_WRAPPERS PREPEND "${PROJECT_SOURCE_DIR}/bin/")
install(FILES ${LUW_POWERSHELL_WRAPPERS} DESTINATION bin)
