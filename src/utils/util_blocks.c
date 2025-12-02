#include "../include/util_blocks.h"
#include <stdio.h>
#include <stdbool.h>
#include "../include/blocks.h"
#include "../include/ANSI-Color-Codes.h"

void grid_print(int grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH]) {
    for (int i = 0; i < MAX_GRID_HEIGHT; i++) {
        printf(" %d |", i);
    }
    printf("\n");
    //loop through the grid
    for (int i = 0; i < MAX_GRID_HEIGHT; i++) {
        for (int j = 0; j < MAX_GRID_WIDTH; j++) {
            // if the cell at position i, j is 1, set it to blue and print it
            if (grid[i][j] == 1) {
                printf(BLU " %d " COLOR_RESET "|", grid[i][j]);
                // else, instead of 0 print .
            }else
                printf(" . |");
        }
        printf(" %d \n", i); // print row index at the end of the row
    }
}


// clear grid
void clear_grid(int grid[8][8]) {

    //loop through the grid
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            //set each cell to 0
            grid[i][j] = 0;
        }
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

void place_simulation_block(int grid[8][8], block b, int pivotX, int pivotY) {

    if (!check_simulation_collision(grid, b, pivotX, pivotY))
        return; // abort if any collision

    // loop through the pattern on x and y-axis
    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {

            if (b.pattern[i][j] != 0) {

                // gx, respectively gy stand for grid x/y
                // these two are equal to the pivot plus j and respectively i
                int gx = pivotX + j;
                int gy = pivotY + i;
                // place the block, this works by getting the cell in b.pattern, and placing it onto the grid at the
                // coordinates calculated earlier
                grid[gy][gx] = b.pattern[i][j];
            }
        }
    }
}

void place_game_block(int grid[8][8], undefined_block ub, int pivotX, int pivotY) {

    if (!check_collision_game_block(grid, ub, pivotX, pivotY))
        return; // abort if any collision

    // loop through the pattern on x and y-axis
    for (int i = 0; i < ub.col; i++) {
        for (int j = 0; j < ub.row; j++) {

            if (ub.pattern[i][j] != 0) {

                // gx, respectively gy stand for grid x/y
                // these two are equal to the pivot plus j and respectively i
                int gx = pivotX + j;
                int gy = pivotY + i;
                // place the block, this works by getting the cell in b.pattern, and placing it onto the grid at the
                // coordinates calculated earlier
                grid[gy][gx] = ub.pattern[i][j];
            }
        }
    }
}

bool check_simulation_collision(int grid[8][8], block b, int pivotX, int pivotY) {

    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            if (b.pattern[i][j] != 0) {
                int gx = pivotX + j;
                int gy = pivotY + i;

                // collision or out-of-bounds
                if (gx < 0 || gx >= 8 || gy < 0 || gy >= 8 || grid[gy][gx] != 0) {
                    // if the coordinate of the block on x-axis on the grid (gx) is less than 0 or >=  8 or gy < 0 or
                    // gy >= 8 OR grid at position gx or gy is not 0, then not able to print and return false
                    printf(RED "Can't place block at (X=%d;Y=%d)!\n" COLOR_RESET, pivotX, pivotY);
                    return false;
                }
            }
        }
    }

    return true;
}

// not the best method but it works...
bool check_collision_game_block(int grid[8][8], undefined_block b, int pivotX, int pivotY) {

    for (int i = 0; i < b.col; i++) {
        for (int j = 0; j < b.row; j++) {
            if (b.pattern[i][j] != 0) {
                int gx = pivotX + j;
                int gy = pivotY + i;

                // collision or out-of-bounds
                if (gx < 0 || gx >= 8 || gy < 0 || gy >= 8 || grid[gy][gx] != 0) {
                    // if the coordinate of the block on x-axis on the grid (gx) is less than 0 or >=  8 or gy < 0 or
                    // gy >= 8 OR grid at position gx or gy is not 0, then not able to print and return false
                    printf(RED "Can't place block at (X=%d;Y=%d)!\n" COLOR_RESET, pivotX, pivotY);
                    return false;
                }
            }
        }
    }

    return true;
}

void check_full_row(int grid[8][8]) {

    // create coord struct
    typedef struct {
        int x;
        int y;
    } Coord;


    // max size is 64 since that's the max size of cells that can be on the grid
    Coord coord[64];
    int clearCount = 0;

    // check rows
    for (int yAxis = 0; yAxis < MAX_GRID_HEIGHT; yAxis++) {
        int xAxisLine = 0;
        // check columns
        for (int xAxis = 0; xAxis < MAX_GRID_WIDTH; xAxis++) {
            if (grid[yAxis][xAxis] == 1) {
                xAxisLine++;
            }
        }

        // mark coordinates in coord array
        if (xAxisLine == MAX_GRID_WIDTH) {
            printf("Row %d is full!\n", yAxis);
            for (int xAxis = 0; xAxis < MAX_GRID_WIDTH; xAxis++) {
                coord[clearCount].x = xAxis;
                coord[clearCount].y = yAxis;
                clearCount++;
            }
        }
    }

    // check columns
    for (int xAxis = 0; xAxis < MAX_GRID_WIDTH; xAxis++) {
        int yAxisLine = 0;
        // check rows
        for (int yAxis = 0; yAxis < MAX_GRID_HEIGHT; yAxis++) {
            if (grid[yAxis][xAxis] == 1) {
                yAxisLine++;
            }
        }

        //mark cell and store in coord array
        if (yAxisLine == MAX_GRID_HEIGHT) {
            printf("Column %d is full!\n", xAxis);
            for (int yAxis = 0; yAxis < MAX_GRID_HEIGHT; yAxis++) {
                coord[clearCount].x = xAxis;
                coord[clearCount].y = yAxis;
                clearCount++;
            }
        }
    }

    // now clear everything
    for (int i = 0; i < clearCount; i++) {
        grid[coord[i].y][coord[i].x] = 0;  // y first, then x
    }
}
