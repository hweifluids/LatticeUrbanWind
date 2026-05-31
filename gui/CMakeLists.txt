cmake_minimum_required(VERSION 3.24)

project(LUWStudio VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Git QUIET)

set(LUW_STUDIO_VERSION "${PROJECT_VERSION}")
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
        OUTPUT_VARIABLE LUW_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(LUW_GIT_DESCRIBE)
        set(LUW_STUDIO_VERSION "${LUW_GIT_DESCRIBE}")
    endif()
endif()

string(TIMESTAMP LUW_BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S %Z")
set(LUW_CFD_CORE_VERSION "FluidX3D 3.3")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/luwgui/BuildInfo.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/luwgui/BuildInfo.h
    @ONLY
)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Widgets OpenGLWidgets Concurrent)
find_package(VTK REQUIRED COMPONENTS
    CommonColor
    CommonCore
    CommonDataModel
    FiltersCore
    FiltersGeneral
    FiltersGeometry
    FiltersModeling
    FiltersPoints
    FiltersSources
    GUISupportQt
    IOGeometry
    IOImage
    IOLegacy
    IOXML
    ImagingFourier
    InteractionStyle
    InteractionWidgets
    RenderingAnnotation
    RenderingCore
    RenderingFreeType
    RenderingOpenGL2
    RenderingVolume
    RenderingVolumeOpenGL2
)

qt_standard_project_setup()

qt_add_executable(luw_studio WIN32
    include/luwgui/BuildInfo.h.in
    include/luwgui/Theme.h
    include/luwgui/Preferences.h
    include/luwgui/RuntimePaths.h
    include/luwgui/ConfigSchema.h
    include/luwgui/ConfigDocument.h
    include/luwgui/CommandRunner.h
    include/luwgui/ConsolePanel.h
    include/luwgui/ProgressPanel.h
    include/luwgui/BatchBoundaryPanel.h
    include/luwgui/BoundaryCsvPanel.h
    include/luwgui/PlotWidgets.h
    include/luwgui/VtkViewWidget.h
    include/luwgui/WavenumberPanel.h
    include/luwgui/BuildingScalePanel.h
    include/luwgui/MainWindow.h
    include/luwgui/StartupDiagnostics.h
    include/luwgui/StartupSplash.h
    src/streamcenter/ColorMapCatalog.h
    src/streamcenter/NvidiaFrucRuntime.h
    src/streamcenter/ViewerWidget.h
    src/streamcenter/index/IndexOpenGLCanvas.h
    src/streamcenter/index/IndexRuntime.h
    src/streamcenter/index/IndexScene.h
    src/streamcenter/index/IndexVolumeBackend.h
    src/streamcenter/index/IndexVolumeData.h
    src/main.cpp
    src/Theme.cpp
    src/Preferences.cpp
    src/RuntimePaths.cpp
    src/ConfigSchema.cpp
    src/ConfigDocument.cpp
    src/CommandRunner.cpp
    src/ConsolePanel.cpp
    src/ProgressPanel.cpp
    src/BatchBoundaryPanel.cpp
    src/BoundaryCsvPanel.cpp
    src/PlotWidgets.cpp
    src/VtkViewWidget.cpp
    src/WavenumberPanel.cpp
    src/BuildingScalePanel.cpp
    src/MainWindow.cpp
    src/StartupDiagnostics.cpp
    src/StartupSplash.cpp
    src/streamcenter/ColorMapCatalog.cpp
    src/streamcenter/NvidiaFrucRuntime.cpp
    src/streamcenter/ViewerWidget.cpp
    src/streamcenter/index/IndexOpenGLCanvas.cpp
    src/streamcenter/index/IndexRuntime.cpp
    src/streamcenter/index/IndexScene.cpp
    src/streamcenter/index/IndexVolumeBackend.cpp
    src/streamcenter/index/IndexVolumeData.cpp
)

target_include_directories(luw_studio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streamcenter
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_compile_definitions(luw_studio PRIVATE
    STREAMCENTERPLUS_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    STREAMCENTERPLUS_ENABLE_NVIDIA_INDEX=0
    STREAMCENTERPLUS_ENABLE_NVOF_FRUC=0
    STREAMCENTERPLUS_ENABLE_VTK_RAYTRACING=0
)

target_link_libraries(luw_studio PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGLWidgets
    Qt6::Concurrent
    ${VTK_LIBRARIES}
)

if(MSVC)
    target_compile_options(luw_studio PRIVATE /W4 /permissive-)
else()
    target_compile_options(luw_studio PRIVATE -Wall -Wextra -Wpedantic)
endif()

vtk_module_autoinit(
    TARGETS luw_studio
    MODULES ${VTK_LIBRARIES}
)
