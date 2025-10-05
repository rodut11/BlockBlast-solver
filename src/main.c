#include <stdio.h>
#include "blocks.h"
#include <string.h>
#include "../utils/util_blocks.h"
#include "../utils/waydroid/C/waydroid_comm.h"

#include <stdlib.h>
#include "../utils/debug/debug.h"


int main() {
    int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH] = {0};

    waydroid_connect("192.168.240.112:5555");
    size_t size;
    unsigned char* image = get_screencap(&size);
    grid_print(grid);
    get_block();




}