#ifndef BLOCKS_H
#define BLOCKS_H

#define MAX_GRID_HEIGHT 8
#define MAX_GRID_WIDTH 8
#define MAX_BLOCK_SIZE 10

typedef struct {
    int width;
    int height;
    int centerX;
    int centerY;
    int pattern[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE]; // max 4x4 for simplicity
} block;

typedef struct {
    int row;
    int col;
    int pattern[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
} undefined_block;


// declare the blocks as extern so other files can use them
// large lshapes
extern block large_Lshape_r0;
extern block large_Lshape_r90;
extern block large_Lshape_r180;
extern block large_Lshape_r270;

//small lshapes
extern block small_Lshape_r0;
extern block small_Lshape_r90;
extern block small_Lshape_r180;
extern block small_Lshape_r270;


//lines
extern block line_horizontal;
extern block line_vertical;

extern block medium_line_vertical;
extern block medium_line_horizontal;

extern block short_line_vertical;
extern block short_line_horizontal;

extern block mini_line_horizontal;
extern block mini_line_vertical;

//squares
extern block small_square;
extern block large_square;

//rectangles
extern block rectangle_horizontal;
extern block rectangle_vertical;

// T-shapes
extern block Tshape_r0;
extern block Tshape_r90;
extern block Tshape_r180;
extern block Tshape_r270;

// normal L-shape
extern block normal_Lshape_vertical_1;
extern block normal_Lshape_down_1;
extern block normal_Lshape_r180;
extern block normal_Lshape_up_1;

// Z and S shapes
extern block Zshape_r0;
extern block Zshape_r90;
extern block Zshape_r180;
extern block Zshape_r270;

#endif // BLOCKS_H
