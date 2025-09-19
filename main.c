#include <stdio.h>
#include "blocks.h"
#include "printBlockUtils.h"

#define MAX_BLOCK_SIZE 10

int main() {
    //initialize grid as fully empty
    int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH] = {
        0
    };
    PrintGrid(grid);
    PrintBlock(Zshape);
}
