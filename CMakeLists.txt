cmake_minimum_required(VERSION 4.0)
project(C C)

set(CMAKE_C_STANDARD 11)

include_directories(..)

add_executable(C
        main.c
        blocks.h
        blocks.c
        printBlockUtils.h
        printBlockUtils.c)
