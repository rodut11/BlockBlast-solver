#ifndef PRINTBLOCKUTILS_H
#define PRINTBLOCKUTILS_H
#include "../src/blocks.h"
#include <stdbool.h>

void grid_print(int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH]);
void block_print(block block);
void place_block(int grid[8][8], block b, int pivotX, int pivotY);
bool check_collision(int grid[8][8], block b, int pivotX, int pivotY);

#endif //PRINTBLOCKUTILS_H