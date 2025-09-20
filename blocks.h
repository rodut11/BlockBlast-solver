#ifndef BLOCKS_H
#define BLOCKS_H

#define MAX_GRID_HEIGHT 9
#define MAX_GRID_WIDTH 9
#define MAX_BLOCK_SIZE 10

typedef struct {
    int width;
    int height;
    int centerX;
    int centerY;
    int pattern[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE]; // max 4x4 for simplicity
} block;

// Declare the blocks as extern so other files can use them
extern block large_LShape;
extern block Line;
extern block Square;
extern block TShape;
extern block normal_Lshape;
extern block Zshape;

#endif // BLOCKS_H
