cmake_minimum_required(VERSION 3.16)

project(cpp_basics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(stop_distance
    stop_distance.cpp
)

add_executable(motor_test
    main.cpp
    motor.cpp
)

target_compile_options(stop_distance PRIVATE -Wall -Wextra)
target_compile_options(motor_test PRIVATE -Wall -Wextra)