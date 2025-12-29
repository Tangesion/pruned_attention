# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.10)

# Project name
project(BF16Simulator)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories
include_directories(include)

# Create a static library for the bf16 simulation
add_library(bf16_sim STATIC
    src/bf16/bf16_basic_ops.cpp
    src/bf16/convert_sim.cpp
    src/bf16/add_sim.cpp
    src/bf16/multiply_sim.cpp
    src/PE/base/backend.cpp
    src/PE/base/componentsRegister.cpp
    src/PE/units/MacUnit.cpp
    src/PE/units/AddUnit.cpp
    src/PE/units/ConvertUnit.cpp
    src/PE/units/MultiplyUnit.cpp
    src/PE/units/MacArrayUnit.cpp
    src/PE/units/AddTreeUnit.cpp
    src/PE/units/ReduceMacUnit.cpp
    src/PE/units/ReduceMacArrayUnit.cpp
    src/PE/scheduler/GemvScheduler.cpp
)

# Add test executables
add_executable(test_bf16_basic_ops test/bf16/test_bf16_basic_ops.cpp)
target_link_libraries(test_bf16_basic_ops bf16_sim)

add_executable(test_convert_sim test/bf16/test_convert_sim.cpp)
target_link_libraries(test_convert_sim bf16_sim)

add_executable(test_add_sim test/bf16/test_add_sim.cpp)
target_link_libraries(test_add_sim bf16_sim)

add_executable(test_add_tree test/PE/test_add_tree.cpp)
target_link_libraries(test_add_tree bf16_sim)

add_executable(test_reduce_mac test/PE/test_reduce_mac.cpp)
target_link_libraries(test_reduce_mac bf16_sim)

add_executable(test_multiply_sim test/bf16/test_multiply_sim.cpp)
target_link_libraries(test_multiply_sim bf16_sim)

add_executable(test_mac_unit test/PE/test_mac_unit.cpp)
target_link_libraries(test_mac_unit bf16_sim)

add_executable(test_gemv_scheduler test/PE/test_gemv_scheduler.cpp)
target_link_libraries(test_gemv_scheduler bf16_sim)

add_executable(test_gemv_arbitrary test/PE/test_gemv_arbitrary.cpp)
target_link_libraries(test_gemv_arbitrary bf16_sim)

add_executable(test_gemv_multibatch test/PE/test_gemv_multibatch.cpp)
target_link_libraries(test_gemv_multibatch bf16_sim)