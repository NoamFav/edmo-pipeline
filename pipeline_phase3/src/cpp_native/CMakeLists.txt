cmake_minimum_required(VERSION 3.15)
project(edmo_native VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_WHISPER "Build Whisper integration" ON)

# Find packages
find_package(Threads REQUIRED)

# Whisper runner
if(BUILD_WHISPER)
    add_library(whisper_runner
        whisper_runner.cpp
        whisper_runner.hpp
    )
    target_include_directories(whisper_runner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(whisper_runner PRIVATE Threads::Threads)
endif()

# Video tools
add_library(video_tools
    video_tools.cpp
    video_tools.hpp
)
target_include_directories(video_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install targets
install(TARGETS video_tools
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(BUILD_WHISPER)
    install(TARGETS whisper_runner
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
