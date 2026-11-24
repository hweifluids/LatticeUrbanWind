if(NOT DEFINED LUW_INSTALLER_SCRIPT OR LUW_INSTALLER_SCRIPT STREQUAL "")
    message(FATAL_ERROR "LUW_INSTALLER_SCRIPT is required")
endif()

if(WIN32)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "LUW_INSTALL_ORCHESTRATED=1"
            "$ENV{ComSpec}" /d /c "${LUW_INSTALLER_SCRIPT}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
else()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "LUW_INSTALL_ORCHESTRATED=1"
            /bin/sh "${LUW_INSTALLER_SCRIPT}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
endif()

set(combined "${output}\n${error}")
if(NOT result EQUAL 3)
    message(FATAL_ERROR
        "installer OpenCL wrapper did not preserve fake status 3; got ${result}\n${combined}")
endif()
if(NOT combined MATCHES "Probing OpenCL")
    message(FATAL_ERROR "installer OpenCL wrapper omitted probe prompt\n${combined}")
endif()
if(NOT combined MATCHES "FAKE_NATIVE_OPENCL_REPORT")
    message(FATAL_ERROR "installer OpenCL wrapper did not invoke native launcher\n${combined}")
endif()
if(NOT combined MATCHES "LatticeUrbanWind Installer: OpenCL device check")
    message(FATAL_ERROR "installer OpenCL wrapper omitted compatibility banner\n${combined}")
endif()
