#include "util_blocks.h"
#include <stdio.h>
#include <stdbool.h>
#include "../src/blocks.h"
#include "../include/ANSI-Color-Codes.h"

void grid_print(int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH]) {
    for (int i = 0; i < MAX_GRID_HEIGHT; i++) {
        printf(" %d |", i);
    }
    printf("\n");
    for (int i = 0; i < MAX_GRID_HEIGHT; i++) {
        for (int j = 0; j < MAX_GRID_WIDTH; j++) {
            if (grid[i][j] == 1) {
                printf(BLU " %d " COLOR_RESET "|", grid[i][j]);
            }else
                printf(" . |");
        }
        printf(" %d \n", i); // new line after each row
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
    pivotX -= 1;
    pivotY -= 1;

    if (!check_collision(grid, b, pivotX, pivotY))
        return; // abort if any collision

    int topLeftX = pivotX - b.centerX;
    int topLeftY = pivotY - b.centerY;

    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            if (b.pattern[i][j] != 0) {
                int gx = topLeftX + j;
                int gy = topLeftY + i;
                grid[gy][gx] = b.pattern[i][j]; // now safe to place
            }
        }
    }
}

bool check_collision(int grid[8][8], block b, int pivotX, int pivotY) {
    int topLeftX = pivotX - b.centerX;
    int topLeftY = pivotY - b.centerY;

    int coordX = pivotX + 1;
    int coordY = pivotY + 1;

    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            if (b.pattern[i][j] != 0) {
                int gx = topLeftX + j;
                int gy = topLeftY + i;

                // collision or out-of-bounds
                if (gx < 0 || gx >= 8 || gy < 0 || gy >= 8 || grid[gy][gx] != 0) {
                    printf(RED "Can't place block at (X=%d;Y=%d)!\n" COLOR_RESET, coordX, coordY);
                    return false;
                }
            }
        }
    }

    return true;
}