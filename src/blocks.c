#include "blocks.h"

//Lshape
block large_Lshape_r0 = {
    .width = 3,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {0, 0, 1},
        {0, 0, 1}
    }
};

block large_Lshape_r90 = {
    .width = 3,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0, 0, 1},
        {0, 0, 1},
        {1, 1, 1}
    }
};

block large_Lshape_r180 = {
    .width = 3,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 0, 0},
        {1, 0, 0},
        {1, 1, 1}
    }
};

block large_Lshape_r270 = {
    .width = 3,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {1, 0, 0},
        {1, 0, 0}
    }
};

block small_Lshape_r0 = {
    .width = 2,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 0},
        {1, 1}
    }
};

block small_Lshape_r90 = {
    .width = 2,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1},
        {0, 1}
    }
};

block small_Lshape_r180 = {
    .width = 2,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0, 1},
        {1, 1}
    }
};

block small_Lshape_r270 = {
    .width = 2,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1},
        {1, 0}
    }
};


//lines
block line_horizontal = {
    .width = 5,
    .height = 1,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1, 1,1}
    }
};

block line_vertical = {
    .width = 1,
    .height = 5,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1},
        {1},
        {1},
        {1},
        {1}
    }
};

block medium_line_vertical = {
    .width = 1,
    .height = 4,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1},
        {1},
        {1},
        {1}
    }
};

block medium_line_horizontal = {
    .width = 4,
    .height = 1,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1, 1}
    }
};

block short_line_vertical = {
    .width = 1,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1},
        {1},
        {1}
    }
};

block short_line_horizontal = {
    .width = 3,
    .height = 1,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1}
    }
};

block mini_line_vertical = {
    .width = 1,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1},
        {1}
    }
};

block mini_line_horizontal = {
    .width = 2,
    .height = 1,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1}
    }
};

//squares
block small_square = {
    .width = 2,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1},
        {1, 1}
    }
};

block large_square = {
    .width = 3,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    }
};

//rectangles
block rectangle_horizontal = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {1, 1, 1},
    }
};

block rectangle_vertical = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1},
        {1, 1},
        {1, 1}
    }
};

block Tshape_r0 = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {0, 1, 0}
    }
};

block Tshape_r90 = {
    .width = 2,
    .height = 1,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0, 1},
        {1, 1},
        {0, 1}
    }
};

block Tshape_r180 = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0, 1, 0},
        {1, 1, 1}
    }
};

block Tshape_r270 = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 0},
        {1, 1},
        {1, 0}
    }
};

block normal_Lshape_vertical_1 = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1,0},
        {1,0},
        {1,1}
    }
};

block normal_Lshape_vertical_2 = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0,1},
        {0,1},
        {1,1}
    }
};

block normal_Lshape_vertical_3 = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1,1},
        {0,1},
        {0,1}
    }
};


block normal_Lshape_vertical_4 = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1,1},
        {1,0},
        {1,0}
    }
};

block normal_Lshape_down_1  = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {1, 0, 0}
    }
};

block normal_Lshape_down_2  = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 1, 1},
        {0, 0, 1}
    }
};


block normal_Lshape_up_1  = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0, 0, 1},
        {1, 1, 1}
    }
};

block normal_Lshape_up_2  = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1, 0, 0},
        {1, 1, 1}
    }
};

//z shapes
block Zshape_r0 = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1,1,0},
        {0,1,1}
    }
};

block Zshape_r90 = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {1,0},
        {1,1},
        {0,1},
    }
};

block Zshape_r180 = {
    .width = 3,
    .height = 2,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0,1,1},
        {1,1,0}
    }
};

block Zshape_r270 = {
    .width = 2,
    .height = 3,
    .centerX = 0,
    .centerY = 0,
    .pattern = {
        {0,1},
        {1,1},
        {1,0},
    }
};

