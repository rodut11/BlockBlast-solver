cmake_minimum_required(VERSION 4.0)
project(C C)

set(CMAKE_C_STANDARD 11)

include_directories(..)

add_executable(C
        src/main.c
        src/blocks.h
        src/blocks.c
        utils/util_blocks.h
        utils/util_blocks.c
        utils/debug/debug_print_blocks.c
        utils/debug/debug.h
        include/ANSI-Color-Codes.h
        utils/waydroid/C/waydroid_comm.c
        utils/waydroid/C/waydroid_comm.h
)

#target_link_libraries(C PRIVATE m)