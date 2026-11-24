if(NOT DEFINED LUW_SOURCE_ROOT)
    message(FATAL_ERROR "LUW_SOURCE_ROOT is required")
endif()

cmake_path(ABSOLUTE_PATH LUW_SOURCE_ROOT NORMALIZE OUTPUT_VARIABLE _luw_root)

# Only inspect repository-owned production and test sources.  Dependency clones,
# build trees, generated fixtures and local logs are deliberately outside this
# list and must not affect the source-distribution gate.
set(_luw_owned_roots
    bin
    cmake
    core
    gui
    installer
    src
    tests
)

set(_luw_python_files)
set(_luw_enumerated_with_git FALSE)
find_program(_luw_git_executable NAMES git)
if(_luw_git_executable AND EXISTS "${_luw_root}/.git")
    execute_process(
        COMMAND "${_luw_git_executable}" -C "${_luw_root}"
                ls-files --cached --others --exclude-standard --
                "*.py" "*.pyc" "*.pyo"
        RESULT_VARIABLE _luw_git_result
        OUTPUT_VARIABLE _luw_git_python_output
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_luw_git_result EQUAL 0)
        set(_luw_enumerated_with_git TRUE)
        string(REPLACE "\r\n" "\n" _luw_git_python_output "${_luw_git_python_output}")
        string(REPLACE "\n" ";" _luw_git_python_files "${_luw_git_python_output}")
        foreach(_relative_python IN LISTS _luw_git_python_files)
            if(_relative_python STREQUAL "")
                continue()
            endif()
            if(_relative_python MATCHES "^(bin|cmake|core|gui|installer|src|tests)/"
                    OR NOT _relative_python MATCHES "/")
                set(_python_candidate "${_luw_root}/${_relative_python}")
                if(EXISTS "${_python_candidate}")
                    list(APPEND _luw_python_files "${_python_candidate}")
                endif()
            endif()
        endforeach()
    endif()
endif()

# Source archives may not contain Git metadata.  In that case, fall back to a
# filesystem scan; generated/ignored local artifacts are absent from an archive.
if(NOT _luw_enumerated_with_git)
    foreach(_relative_root IN LISTS _luw_owned_roots)
        set(_owned_root "${_luw_root}/${_relative_root}")
        if(NOT EXISTS "${_owned_root}")
            continue()
        endif()
        file(GLOB_RECURSE _root_python_files
            LIST_DIRECTORIES FALSE
            "${_owned_root}/*.py"
            "${_owned_root}/*.pyc"
            "${_owned_root}/*.pyo"
        )
        list(APPEND _luw_python_files ${_root_python_files})
    endforeach()

    file(GLOB _luw_top_level_python
        LIST_DIRECTORIES FALSE
        "${_luw_root}/*.py"
        "${_luw_root}/*.pyc"
        "${_luw_root}/*.pyo"
    )
    list(APPEND _luw_python_files ${_luw_top_level_python})
endif()
list(REMOVE_DUPLICATES _luw_python_files)

# CULBM keeps four Python programs for offline validation/reporting only.  They
# are not installed, launched, or imported by LUW production code.  Keep this
# allowlist exact so that any new Python source still fails the gate.
set(_luw_allowed_offline_python
    "${_luw_root}/core/cfd_core/CULBM/tests/compare_backends.py"
    "${_luw_root}/core/cfd_core/CULBM/tests/compare_physical_vtk.py"
    "${_luw_root}/core/cfd_core/CULBM/validation/gea/analyze_results.py"
    "${_luw_root}/core/cfd_core/CULBM/validation/gea/report/build_report.py"
)
foreach(_allowed_python IN LISTS _luw_allowed_offline_python)
    list(REMOVE_ITEM _luw_python_files "${_allowed_python}")
endforeach()

if(_luw_python_files)
    list(SORT _luw_python_files)
    string(JOIN "\n  " _python_file_list ${_luw_python_files})
    message(FATAL_ERROR
        "Python source or bytecode exists in the repository-owned tree:\n"
        "  ${_python_file_list}")
endif()

# Detect high-confidence embedded/runtime launch APIs.  Compatibility text such
# as the legacy display-only `Usage: python ...` messages is intentionally not
# rejected; it does not execute an interpreter and is part of the public console
# contract.
set(_luw_runtime_sources)
foreach(_relative_root IN ITEMS cmake core/cfd_core/CULBM gui installer src)
    set(_owned_root "${_luw_root}/${_relative_root}")
    if(NOT EXISTS "${_owned_root}")
        continue()
    endif()
    file(GLOB_RECURSE _root_runtime_sources
        LIST_DIRECTORIES FALSE
        "${_owned_root}/*.c"
        "${_owned_root}/*.cc"
        "${_owned_root}/*.cpp"
        "${_owned_root}/*.cxx"
        "${_owned_root}/*.h"
        "${_owned_root}/*.hpp"
        "${_owned_root}/*.cmake"
        "${_owned_root}/CMakeLists.txt"
        "${_owned_root}/*.cmd"
        "${_owned_root}/*.ps1"
        "${_owned_root}/*.sh"
    )
    list(APPEND _luw_runtime_sources ${_root_runtime_sources})
endforeach()
list(APPEND _luw_runtime_sources
    "${_luw_root}/CMakeLists.txt"
    "${_luw_root}/install_linux.sh"
    "${_luw_root}/install_win.cmd"
)

set(_luw_forbidden_runtime_patterns
    "[#][ \t]*include[ \t]*[<\"]Python[.]h[>\"]"
    "Py_Initialize[ \t]*[(]"
    "PyRun_[A-Za-z0-9_]*[ \t]*[(]"
    "PyImport_[A-Za-z0-9_]*[ \t]*[(]"
    "(std::system|_popen|popen)[ \t]*[(][^\n]*(python|python3|py[.]exe)"
    "QProcess[^\n]*(python|python3|py[.]exe)"
)

set(_luw_runtime_violations)
foreach(_source IN LISTS _luw_runtime_sources)
    if(NOT EXISTS "${_source}" OR IS_DIRECTORY "${_source}")
        continue()
    endif()
    if(_source STREQUAL "${_luw_root}/cmake/VerifyPythonFree.cmake")
        continue()
    endif()
    file(READ "${_source}" _source_text)
    foreach(_pattern IN LISTS _luw_forbidden_runtime_patterns)
        if(_source_text MATCHES "${_pattern}")
            cmake_path(RELATIVE_PATH _source BASE_DIRECTORY "${_luw_root}"
                       OUTPUT_VARIABLE _relative_source)
            list(APPEND _luw_runtime_violations "${_relative_source}")
            break()
        endif()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES _luw_runtime_violations)

if(_luw_runtime_violations)
    list(SORT _luw_runtime_violations)
    string(JOIN "\n  " _runtime_file_list ${_luw_runtime_violations})
    message(FATAL_ERROR
        "A Python embedding or explicit interpreter-launch API was detected:\n"
        "  ${_runtime_file_list}")
endif()

message(STATUS "Python-free production source gate passed (4 offline CULBM analysis scripts allowlisted)")
