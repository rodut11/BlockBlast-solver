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
    int pattern[4][4]; // max 4x4 for simplicity
} Block;

// Declare the blocks as extern so other files can use them
extern Block large_LShape;
extern Block Line;
extern Block Square;
extern Block TShape;
extern Block normal_Lshape;
extern Block Zshape;

#endif // BLOCKS_H
