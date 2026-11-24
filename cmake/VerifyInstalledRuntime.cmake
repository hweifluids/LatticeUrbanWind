if(NOT DEFINED LUW_INSTALL_PREFIX OR LUW_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "LUW_INSTALL_PREFIX is required")
endif()

get_filename_component(_luw_prefix "${LUW_INSTALL_PREFIX}" ABSOLUTE)
if(WIN32)
    set(_luw_launcher "${_luw_prefix}/bin/luw.exe")
    set(_luw_studio "${_luw_prefix}/bin/luw_studio.exe")
    set(_luw_path "${_luw_prefix}/bin;$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
else()
    set(_luw_launcher "${_luw_prefix}/bin/luw")
    set(_luw_studio "${_luw_prefix}/bin/luw_studio")
    set(_luw_path "${_luw_prefix}/bin:/usr/bin:/bin")
endif()

if(NOT EXISTS "${_luw_launcher}")
    message(FATAL_ERROR "Installed launcher not found: ${_luw_launcher}")
endif()

set(ENV{LUW_HOME} "${_luw_prefix}")
set(ENV{PATH} "${_luw_path}")
foreach(_luw_variable IN ITEMS
        QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH
        LD_LIBRARY_PATH DYLD_LIBRARY_PATH
        GDAL_DATA PROJ_DATA PROJ_LIB)
    unset(ENV{${_luw_variable}})
endforeach()

execute_process(
    COMMAND "${_luw_launcher}" --version
    RESULT_VARIABLE _luw_version_result
    OUTPUT_VARIABLE _luw_version_output
    ERROR_VARIABLE _luw_version_error)
if(NOT _luw_version_result EQUAL 0
   OR NOT _luw_version_output MATCHES "LatticeUrbanWind Native")
    message(FATAL_ERROR
        "Installed luw --version failed (${_luw_version_result}).\n"
        "stdout: ${_luw_version_output}\nstderr: ${_luw_version_error}")
endif()

execute_process(
    COMMAND "${_luw_launcher}" detect_opencl
    RESULT_VARIABLE _luw_opencl_result
    OUTPUT_VARIABLE _luw_opencl_output
    ERROR_VARIABLE _luw_opencl_error)
if(NOT _luw_opencl_result EQUAL 0 AND NOT _luw_opencl_result EQUAL 3)
    message(FATAL_ERROR
        "Installed luw detect_opencl failed (${_luw_opencl_result}).\n"
        "stdout: ${_luw_opencl_output}\nstderr: ${_luw_opencl_error}")
endif()

if(WIN32)
    file(GLOB _luw_runtime_libraries "${_luw_prefix}/bin/*.dll")
else()
    file(GLOB _luw_runtime_libraries "${_luw_prefix}/lib/*.so*")
endif()
if(NOT _luw_runtime_libraries)
    message(FATAL_ERROR
        "No bundled non-system runtime libraries were found under ${_luw_prefix}")
endif()

if(EXISTS "${_luw_studio}")
    if(NOT EXISTS "${_luw_prefix}/bin/qt.conf")
        message(FATAL_ERROR "Installed GUI is missing bin/qt.conf")
    endif()
    if(WIN32)
        set(_luw_offscreen_plugin
            "${_luw_prefix}/plugins/platforms/qoffscreen.dll")
    else()
        set(_luw_offscreen_plugin
            "${_luw_prefix}/plugins/platforms/libqoffscreen.so")
    endif()
    if(NOT EXISTS "${_luw_offscreen_plugin}")
        message(FATAL_ERROR
            "Installed GUI is missing the offscreen Qt platform plugin: "
            "${_luw_offscreen_plugin}")
    endif()

    if(WIN32)
        file(GLOB_RECURSE _luw_qt_plugins
            "${_luw_prefix}/plugins/*.dll")
        set(_luw_plugin_post_excludes
            ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\\\].*"
            ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Ww][Oo][Ww]64[/\\\\].*")
    else()
        file(GLOB_RECURSE _luw_qt_plugins
            "${_luw_prefix}/plugins/*.so*")
        set(_luw_plugin_post_excludes
            "^/lib/.*" "^/lib64/.*" "^/usr/lib/.*" "^/usr/lib64/.*")
    endif()
    if(NOT _luw_qt_plugins)
        message(FATAL_ERROR "Installed GUI has no Qt plugins")
    endif()
    file(GET_RUNTIME_DEPENDENCIES
        MODULES ${_luw_qt_plugins}
        DIRECTORIES
            "${_luw_prefix}/bin"
            "${_luw_prefix}/lib"
        RESOLVED_DEPENDENCIES_VAR _luw_plugin_dependencies
        UNRESOLVED_DEPENDENCIES_VAR _luw_plugin_unresolved
        PRE_EXCLUDE_REGEXES "^api-ms-win-.*" "^ext-ms-win-.*"
        POST_EXCLUDE_REGEXES ${_luw_plugin_post_excludes})
    if(_luw_plugin_unresolved)
        list(JOIN _luw_plugin_unresolved ", " _luw_plugin_unresolved_text)
        message(FATAL_ERROR
            "Installed Qt plugin dependency closure is incomplete: "
            "${_luw_plugin_unresolved_text}")
    endif()

    execute_process(
        COMMAND "${_luw_studio}"
            -platform offscreen --runtime-self-check
        RESULT_VARIABLE _luw_studio_result
        OUTPUT_VARIABLE _luw_studio_output
        ERROR_VARIABLE _luw_studio_error
        TIMEOUT 30)
    if(NOT _luw_studio_result EQUAL 0
       OR NOT _luw_studio_output MATCHES "runtime self-check: OK")
        message(FATAL_ERROR
            "Installed luw_studio self-check failed (${_luw_studio_result}).\n"
            "stdout: ${_luw_studio_output}\nstderr: ${_luw_studio_error}")
    endif()
endif()

message(STATUS "Installed runtime verification passed: ${_luw_prefix}")
message(STATUS "luw --version: ${_luw_version_output}")
message(STATUS "detect_opencl exit code: ${_luw_opencl_result}")
