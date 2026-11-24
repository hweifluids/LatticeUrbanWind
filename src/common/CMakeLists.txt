add_library(luw_common STATIC
    deck_schema.cpp
    deck_document.cpp
    preprocess_compute_config.cpp
    progress.cpp
    runtime_paths.cpp
    terrain_voxel_config.cpp
)

target_include_directories(luw_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(luw_common
    PUBLIC
        LUW_SOURCE_ROOT="${PROJECT_SOURCE_DIR}"
        LUW_VERSION="${PROJECT_VERSION}"
)

target_link_libraries(luw_common
    PUBLIC
        Qt6::Core
        Qt6::Concurrent
        Threads::Threads
)

if(MSVC)
    target_compile_options(luw_common PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(luw_common PRIVATE -Wall -Wextra -Wpedantic)
endif()
