# cmake needs this line
cmake_minimum_required(VERSION 3.21)  # required for CMakePresets.json v3

project(obj_tracker_cpp)  # automatically sets var $PROJECT_NAME

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for manually analyzing any of the files using clang-tidy

option(ENABLE_CLANG_TIDY "" OFF)

function(enable_clang_tidy_for target)  # helper fuction for clang tidy
    if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_PATH clang-tidy REQUIRED)
    set_target_properties(${target}
        PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY_PATH}")
    endif()
endfunction()

# Library of reusable pipeline components (detector, transforms, camera, viz) along with the tracker.
# ByteTrack Kept as its own static lib so ByteTrack's Eigen dependency stays isolated
# and clang-tidy runs on our code only, not on third-party sources.
add_library(bytetrack STATIC
    src/ByteTrack/BYTETracker.cpp
    src/ByteTrack/KalmanFilter.cpp
    src/ByteTrack/Object.cpp
    src/ByteTrack/Rect.cpp
    src/ByteTrack/STrack.cpp
    src/ByteTrack/lapjv.cpp)

target_include_directories(bytetrack PUBLIC
    ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(bytetrack PUBLIC Eigen3::Eigen)

# Library of reusable pipeline components (detector, transforms, camera, viz, tracker).
# main.cpp and any test target link against this so we don't rebuild sources.
add_library(obj_tracker_lib STATIC
    src/camera.cpp
    src/detector.cpp
    src/engine.cpp
    src/postprocessor.cpp
    src/preprocessor.cpp
    src/reporting.cpp
    src/supervisor.cpp
    src/transforms.cpp
    src/tracker.cpp
    src/visualization.cpp)

target_include_directories(obj_tracker_lib PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(obj_tracker_lib
    PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime
    PRIVATE bytetrack)

enable_clang_tidy_for(obj_tracker_lib)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE obj_tracker_lib)
enable_clang_tidy_for(${PROJECT_NAME})

# Tests
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    enable_testing()

    # One entry per tests/test_<name>.cpp. Every test wires up identically:
    # a main() returning 0/1 (see tests/test_harness.hpp), so ctest just runs
    # the executable and checks its exit code.
    set(TEST_NAMES
        camera
        detector
        postprocessor
        preprocessor
        reporting
        supervisor
        visualizer)

    foreach(test_name IN LISTS TEST_NAMES)
        add_executable(test_${test_name} tests/test_${test_name}.cpp)
        target_link_libraries(test_${test_name} PRIVATE obj_tracker_lib)
        target_compile_definitions(test_${test_name} PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()
endif()
