#include "util_blocks.h"
#include <stdio.h>
#include "../src/blocks.h"

void grid_print(int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH]) {

    for(int i = 0; i < MAX_GRID_HEIGHT; i++) {
        for(int j = 0; j < MAX_GRID_WIDTH; j++) {
            printf("%d ", grid[i][j]); // print the value
        }
        printf("\n"); // new line after each row
    }
}

void block_print(block block) {
    for(int i = 0; i < block.height; i++) {
        for(int j = 0; j < block.width; j++) {
            printf("%d ", block.pattern[i][j]);
        }
        printf("\n");
    }
}

void place_block(int grid[8][8], block b, int pivotX, int pivotY) {
    // compute the top-left corner based on the block's pivot
    int topLeftX = pivotX - b.centerX;
    int topLeftY = pivotY - b.centerY;

    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            if (b.pattern[i][j] != 0) { // only place filled parts
                int gx = topLeftX + j;
                int gy = topLeftY + i;

                // make sure were inside the grid
                if (gx >= 0 && gx < 8 && gy >= 0 && gy < 8)
                    grid[gy][gx] = b.pattern[i][j];
            }
        }
    }
}