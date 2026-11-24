include_guard(GLOBAL)

include(GNUInstallDirs)

function(_luw_append_runtime_target_directory output_variable target_name)
    if(NOT TARGET "${target_name}")
        set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
        return()
    endif()

    get_target_property(_luw_target_type "${target_name}" TYPE)
    if(_luw_target_type STREQUAL "SHARED_LIBRARY"
       OR _luw_target_type STREQUAL "MODULE_LIBRARY"
       OR _luw_target_type STREQUAL "UNKNOWN_LIBRARY"
       OR _luw_target_type STREQUAL "EXECUTABLE")
        set(_luw_directories "${${output_variable}}")
        list(APPEND _luw_directories "$<TARGET_FILE_DIR:${target_name}>")
        set(${output_variable} "${_luw_directories}" PARENT_SCOPE)
    else()
        set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
    endif()
endfunction()

function(_luw_append_target_prefix output_variable target_name)
    if(NOT TARGET "${target_name}")
        set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
        return()
    endif()

    set(_luw_location "")
    foreach(_luw_config RELEASE RELWITHDEBINFO MINSIZEREL DEBUG)
        get_target_property(_luw_candidate "${target_name}"
            "IMPORTED_LOCATION_${_luw_config}")
        if(_luw_candidate AND NOT _luw_candidate MATCHES "-NOTFOUND$")
            set(_luw_location "${_luw_candidate}")
            break()
        endif()
    endforeach()
    if(NOT _luw_location)
        get_target_property(_luw_candidate "${target_name}" IMPORTED_LOCATION)
        if(_luw_candidate AND NOT _luw_candidate MATCHES "-NOTFOUND$")
            set(_luw_location "${_luw_candidate}")
        endif()
    endif()

    set(_luw_prefixes "${${output_variable}}")
    if(_luw_location)
        get_filename_component(_luw_binary_directory "${_luw_location}" DIRECTORY)
        get_filename_component(_luw_directory_name "${_luw_binary_directory}" NAME)
        string(TOLOWER "${_luw_directory_name}" _luw_directory_name)
        if(_luw_directory_name STREQUAL "bin"
           OR _luw_directory_name STREQUAL "lib")
            get_filename_component(_luw_prefix "${_luw_binary_directory}" DIRECTORY)
            list(APPEND _luw_prefixes "${_luw_prefix}")
        elseif(_luw_directory_name STREQUAL "debug")
            get_filename_component(_luw_prefix "${_luw_binary_directory}/.." ABSOLUTE)
            list(APPEND _luw_prefixes "${_luw_prefix}")
        endif()
    endif()
    set(${output_variable} "${_luw_prefixes}" PARENT_SCOPE)
endfunction()

function(_luw_locate_runtime_data)
    set(_luw_prefixes "")
    _luw_append_target_prefix(_luw_prefixes GDAL::GDAL)
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND _luw_prefixes
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
    endif()
    list(APPEND _luw_prefixes ${CMAKE_PREFIX_PATH})
    list(REMOVE_DUPLICATES _luw_prefixes)

    set(_luw_vtk_prefixes "")
    _luw_append_target_prefix(_luw_vtk_prefixes VTK::libproj)

    if(NOT LUW_GDAL_DATA_DIRECTORY)
        set(_luw_gdal_candidates "")
        if(DEFINED GDAL_DIR)
            list(APPEND _luw_gdal_candidates "${GDAL_DIR}")
        endif()
        foreach(_luw_prefix IN LISTS _luw_prefixes)
            list(APPEND _luw_gdal_candidates
                "${_luw_prefix}/share/gdal"
                "${_luw_prefix}/share/gdal/data")
        endforeach()
        foreach(_luw_candidate IN LISTS _luw_gdal_candidates)
            if(EXISTS "${_luw_candidate}/gdalvrt.xsd")
                set(LUW_GDAL_DATA_DIRECTORY "${_luw_candidate}" CACHE PATH
                    "Directory containing GDAL runtime data" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if(NOT LUW_PROJ_DATA_DIRECTORY)
        set(_luw_proj_candidates "")
        foreach(_luw_prefix IN LISTS _luw_prefixes)
            list(APPEND _luw_proj_candidates
                "${_luw_prefix}/share/proj"
                "${_luw_prefix}/share/proj-data")
        endforeach()
        foreach(_luw_candidate IN LISTS _luw_proj_candidates)
            if(EXISTS "${_luw_candidate}/proj.db")
                set(LUW_PROJ_DATA_DIRECTORY "${_luw_candidate}" CACHE PATH
                    "Directory containing PROJ runtime data" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if(NOT LUW_VTK_PROJ_DATA_DIRECTORY)
        foreach(_luw_prefix IN LISTS _luw_vtk_prefixes)
            file(GLOB _luw_vtk_proj_databases
                "${_luw_prefix}/share/vtk-*/proj/proj.db")
            list(SORT _luw_vtk_proj_databases COMPARE NATURAL ORDER DESCENDING)
            if(_luw_vtk_proj_databases)
                list(GET _luw_vtk_proj_databases 0 _luw_vtk_proj_database)
                get_filename_component(_luw_vtk_proj_directory
                    "${_luw_vtk_proj_database}" DIRECTORY)
                set(LUW_VTK_PROJ_DATA_DIRECTORY
                    "${_luw_vtk_proj_directory}" CACHE PATH
                    "Directory containing VTK's bundled PROJ runtime database" FORCE)
                break()
            endif()
        endforeach()
    endif()

    set(LUW_GDAL_DATA_DIRECTORY "${LUW_GDAL_DATA_DIRECTORY}" PARENT_SCOPE)
    set(LUW_PROJ_DATA_DIRECTORY "${LUW_PROJ_DATA_DIRECTORY}" PARENT_SCOPE)
    set(LUW_VTK_PROJ_DATA_DIRECTORY
        "${LUW_VTK_PROJ_DATA_DIRECTORY}" PARENT_SCOPE)
endfunction()

function(luw_configure_install)
    if(LUW_INSTALL_RUNTIME_DEPENDENCIES)
        set(_luw_runtime_directories ${LUW_RUNTIME_SEARCH_DIRECTORIES})
        foreach(_luw_target IN ITEMS
                Qt6::Core Qt6::Concurrent Qt6::Gui Qt6::Widgets
                Qt6::OpenGL Qt6::OpenGLWidgets
                VTK::CommonCore VTK::CommonDataModel VTK::IOXML
                VTK::libproj VTK::netcdf
                GDAL::GDAL GEOS::geos_c JPEG::JPEG
                manifold::manifold OpenCL::OpenCL)
            _luw_append_runtime_target_directory(
                _luw_runtime_directories "${_luw_target}")
        endforeach()
        list(REMOVE_DUPLICATES _luw_runtime_directories)

        if(WIN32)
            install(RUNTIME_DEPENDENCY_SET luw_runtime_dependencies
                PRE_EXCLUDE_REGEXES
                    "^api-ms-win-.*"
                    "^ext-ms-win-.*"
                POST_EXCLUDE_REGEXES
                    ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\\\].*"
                    ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Ww][Oo][Ww]64[/\\\\].*"
                DIRECTORIES ${_luw_runtime_directories}
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            )
        elseif(UNIX AND NOT APPLE)
            install(RUNTIME_DEPENDENCY_SET luw_runtime_dependencies
                POST_EXCLUDE_REGEXES
                    "^/lib/.*"
                    "^/lib64/.*"
                    "^/usr/lib/.*"
                    "^/usr/lib64/.*"
                DIRECTORIES ${_luw_runtime_directories}
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            )
        endif()
    endif()

    if(LUW_BUILD_GUI AND TARGET luw_studio
       AND COMMAND qt_generate_deploy_app_script)
        if(WIN32)
            qt_generate_deploy_app_script(
                TARGET luw_studio
                OUTPUT_SCRIPT _luw_qt_deploy_script
                NO_TRANSLATIONS
                NO_COMPILER_RUNTIME
                DEPLOY_TOOL_OPTIONS
                    --no-libraries
                    --include-plugins qoffscreen
                    --skip-plugin-types generic,networkinformation,tls
                    --no-system-d3d-compiler
            )
        else()
            qt_generate_deploy_app_script(
                TARGET luw_studio
                OUTPUT_SCRIPT _luw_qt_deploy_script
                NO_TRANSLATIONS
                NO_COMPILER_RUNTIME
                POST_EXCLUDE_REGEXES
                    "^/lib/.*"
                    "^/lib64/.*"
                    "^/usr/lib/.*"
                    "^/usr/lib64/.*"
            )
        endif()
        install(SCRIPT "${_luw_qt_deploy_script}")
    endif()

    install(FILES "${CMAKE_SOURCE_DIR}/core/deck_schema.json"
        DESTINATION core)
    install(FILES "${CMAKE_SOURCE_DIR}/LICENSE.md" DESTINATION .)
    # Install only reproducible solver inputs. Solver trees commonly sit next
    # to ignored multi-gigabyte build and validation artifacts, so unrestricted
    # directory installs would silently package local state.
    if(LUW_ENABLE_FLUIDX3D)
        install(FILES
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/CITATION.cff"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/DOCUMENTATION.md"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/FluidX3D.sln"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/FluidX3D.vcxproj"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/LICENSE.md"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/README.md"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/make.sh"
            "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/makefile"
            DESTINATION core/cfd_core/FluidX3D)
        install(DIRECTORY "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/src/"
            DESTINATION core/cfd_core/FluidX3D/src)
    endif()

    install(FILES
        "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/CMakeLists.txt"
        "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/LICENSE.md"
        "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/README.md"
        "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/NOTICE.md"
        "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/build.ps1"
        "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/build.sh"
        DESTINATION core/cfd_core/CULBM)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/src/"
        DESTINATION core/cfd_core/CULBM/src)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/cmake/"
        DESTINATION core/cfd_core/CULBM/cmake)
    if(WIN32)
        set(_luw_culbm_executable
            "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/bin/CULBM.exe")
    else()
        set(_luw_culbm_executable
            "${CMAKE_SOURCE_DIR}/core/cfd_core/CULBM/bin/CULBM")
    endif()
    install(PROGRAMS "${_luw_culbm_executable}"
        OPTIONAL DESTINATION core/cfd_core/CULBM/bin)
    if(LUW_ENABLE_FLUIDX3D)
        if(WIN32)
            set(_luw_fluidx3d_executable
                "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/bin/FluidX3D.exe")
        else()
            set(_luw_fluidx3d_executable
                "${CMAKE_SOURCE_DIR}/core/cfd_core/FluidX3D/bin/FluidX3D")
        endif()
        install(PROGRAMS "${_luw_fluidx3d_executable}"
            OPTIONAL DESTINATION core/cfd_core/FluidX3D/bin)
    endif()
    # Runtime datasets are user inputs and can be very large.  Ship the tracked
    # placeholder only; never capture a developer's local DEM/building cache.
    install(FILES
        "${CMAKE_SOURCE_DIR}/database/leave dem[point]tif here.txt"
        DESTINATION database)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/project_template/"
        DESTINATION project_template)
    if(LUW_BUILD_GUI)
        install(FILES
            "${CMAKE_SOURCE_DIR}/gui/src/assets/img/startup_default.jpg"
            "${CMAKE_SOURCE_DIR}/gui/src/assets/img/background_default.jpg"
            "${CMAKE_SOURCE_DIR}/gui/src/assets/img/logo.png"
            DESTINATION gui/src/assets/img)
    endif()

    _luw_locate_runtime_data()
    if(LUW_GDAL_DATA_DIRECTORY)
        install(DIRECTORY "${LUW_GDAL_DATA_DIRECTORY}/"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/gdal")
    elseif(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND
           AND LUW_REQUIRE_FULL_DEPENDENCIES)
        message(FATAL_ERROR
            "GDAL was found, but its runtime data directory was not. Set "
            "LUW_GDAL_DATA_DIRECTORY to the directory containing gdalvrt.xsd.")
    endif()

    if(LUW_PROJ_DATA_DIRECTORY)
        install(DIRECTORY "${LUW_PROJ_DATA_DIRECTORY}/"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/proj")
    elseif(LUW_ENABLE_GEOSPATIAL AND GDAL_FOUND
           AND LUW_REQUIRE_FULL_DEPENDENCIES)
        message(FATAL_ERROR
            "GDAL was found, but PROJ's runtime database was not. Set "
            "LUW_PROJ_DATA_DIRECTORY to the directory containing proj.db.")
    endif()

    if(LUW_VTK_PROJ_DATA_DIRECTORY)
        get_filename_component(_luw_vtk_proj_parent
            "${LUW_VTK_PROJ_DATA_DIRECTORY}" DIRECTORY)
        get_filename_component(_luw_vtk_share_name
            "${_luw_vtk_proj_parent}" NAME)
        install(DIRECTORY "${LUW_VTK_PROJ_DATA_DIRECTORY}/"
            DESTINATION
                "${CMAKE_INSTALL_DATADIR}/${_luw_vtk_share_name}/proj")
    endif()
endfunction()
