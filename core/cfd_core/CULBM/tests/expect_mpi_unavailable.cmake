if(NOT DEFINED TEST_EXECUTABLE OR TEST_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()

if(TEST_MODE STREQUAL "forced")
    set(_environment "CULBM_MPI=1")
elseif(TEST_MODE STREQUAL "launcher")
    set(_environment "CULBM_MPI=0" "SLURM_PROCID=0")
else()
    message(FATAL_ERROR "Unknown TEST_MODE='${TEST_MODE}'")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        ${_environment}
        "${TEST_EXECUTABLE}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)

if(_result EQUAL 0)
    message(FATAL_ERROR
        "Non-MPI runtime unexpectedly accepted TEST_MODE='${TEST_MODE}'")
endif()

set(_combined_output "${_stdout}\n${_stderr}")
if(NOT _combined_output MATCHES "built without MPI support")
    message(FATAL_ERROR
        "Non-MPI runtime failed without the expected diagnostic.\n"
        "stdout:\n${_stdout}\n"
        "stderr:\n${_stderr}")
endif()

message(STATUS
    "Non-MPI runtime rejected TEST_MODE='${TEST_MODE}' with the expected diagnostic")
