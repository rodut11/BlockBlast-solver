#include <stdio.h>
#include "blocks.h"
#include "../utils/util_blocks.h"
#include "../utils/debug/debug_blocks.h"

int main() {
    //initialize grid
    int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH] = {0};

    place_block(grid, large_square, 1, 1);
    place_block(grid, large_square, 1, 1);
    grid_print(grid);

}
