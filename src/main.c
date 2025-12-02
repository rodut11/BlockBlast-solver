#include <stdio.h>
#include "../include/blocks.h"
#include <string.h>
#include "../include/util_blocks.h"
#include "../include/waydroid/waydroid_comm.h"

#include <stdlib.h>
#include "utils/debug/debug.h"
#include "vision/vision.h"

int main(int argc, char **argv) {
    int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH] = {0};

    waydroid_connect("192.168.240.112:5555");
    // size_t size;
    // unsigned char* image = get_screencap(&size);

    get_grid(grid);
    get_block_coord();
    grid_print(grid);
    get_block();

    int x, y;
    get_grid_origin(&x, &y);
    printf("Grid origin: %d,%d\n", x, y);

}