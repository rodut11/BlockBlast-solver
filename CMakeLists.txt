cmake_minimum_required(VERSION 4.0)
project(C C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(..)

add_executable(C
        src/main.c
        include/blocks.h
        src/blocks.c
        include/util_blocks.h
        src/utils/util_blocks.c
        src/utils/debug/debug_print_blocks.c
        src/utils/debug/debug.h
        include/ANSI-Color-Codes.h
        src/utils/waydroid/C/waydroid_comm.c
        include/waydroid/waydroid_comm.h
        src/vision/get_grid_origin.cpp
        src/vision/get_grid.cpp
        src/vision/internal_vision.cpp
        src/vision/vision.h
        src/vision/internal_vision.hpp
        src/vision/get_block.cpp
        src/vision/get_block_coord.cpp
        src/vision/get_block_coord.cpp
        src/vision/get_bounding_box.hpp
        assets/sample.cpp
)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(C PRIVATE ${OpenCV_LIBS})

#GTK4
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(GTK4 REQUIRED gtk4)

#include_directories(${GTK4_INCLUDE_DIRS})
#link_directories(${GTK4_LIBRARY_DIRS})
#add_definitions(${GTK4_CFLAGS_OTHER})

#target_link_libraries(C PRIVATE ${GTK4_LIBRARIES} ${OpenCV_LIBS})
