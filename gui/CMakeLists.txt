cmake_minimum_required(VERSION 3.24)

project(LUWStudio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Widgets OpenGLWidgets Concurrent)
find_package(VTK REQUIRED COMPONENTS
    CommonColor
    CommonCore
    CommonDataModel
    FiltersCore
    FiltersGeneral
    FiltersGeometry
    FiltersModeling
    GUISupportQt
    IOGeometry
    IOImage
    IOLegacy
    ImagingFourier
    InteractionStyle
    InteractionWidgets
    RenderingAnnotation
    RenderingCore
    RenderingOpenGL2
)

qt_standard_project_setup()

qt_add_executable(luw_studio
    include/luwgui/Theme.h
    include/luwgui/Preferences.h
    include/luwgui/ConfigSchema.h
    include/luwgui/ConfigDocument.h
    include/luwgui/CommandRunner.h
    include/luwgui/ConsolePanel.h
    include/luwgui/BatchBoundaryPanel.h
    include/luwgui/BoundaryCsvPanel.h
    include/luwgui/PlotWidgets.h
    include/luwgui/VtkViewWidget.h
    include/luwgui/WavenumberPanel.h
    include/luwgui/BuildingScalePanel.h
    include/luwgui/MainWindow.h
    src/main.cpp
    src/Theme.cpp
    src/Preferences.cpp
    src/ConfigSchema.cpp
    src/ConfigDocument.cpp
    src/CommandRunner.cpp
    src/ConsolePanel.cpp
    src/BatchBoundaryPanel.cpp
    src/BoundaryCsvPanel.cpp
    src/PlotWidgets.cpp
    src/VtkViewWidget.cpp
    src/WavenumberPanel.cpp
    src/BuildingScalePanel.cpp
    src/MainWindow.cpp
)

target_include_directories(luw_studio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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
