#ifndef PRINTBLOCKUTILS_H
#define PRINTBLOCKUTILS_H
#include "../src/blocks.h"
#include <stdbool.h>

void grid_print(int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH]);
void block_print(block block);
void place_simulation_block(int grid[8][8], block b, int pivotX, int pivotY);
void place_game_block(int grid[8][8], undefined_block ub, int pivotX, int pivotY);
bool check_simulation_collision(int grid[8][8], block b, int pivotX, int pivotY);
bool check_collision_game_block(int grid[8][8], undefined_block b, int pivotX, int pivotY);
void clear_grid(int grid[8][8]);
void check_full_row(int grid[8][8]);
void get_block();

#endif //PRINTBLOCKUTILS_H