#include <iostream>
#include <opencv2/imgproc.hpp>
#include "internal_vision.hpp"

extern "C" void get_grid(int grid[8][8]) {
    // virtual grid
    int vgrid[8][8] = {0};

    // get and decode the screen capture
    std::vector<u_char> buffer = grab_screencap();
    cv::Mat img = decode_screencap(buffer);

    if (!img.empty()) {

        // coordinates (top-left and bottom-right)
        int gx1 = 718;
        int gy1 = 209;
        int gx2 = 1202;
        int gy2 = 693;

        int cell_height = 0;
        int cell_width = 0;

        // calculate region
        cv::Point topLeft(gx1, gy1);
        cv::Point bottomRight(gx2, gy2);

        int gwidth = bottomRight.x - topLeft.x;
        int gheight = bottomRight.y - topLeft.y;
        //std::cout << "Width: " << gwidth << " Height: " << gheight << std::endl;
        cv::Rect roi(topLeft.x, topLeft.y, gwidth, gheight);

        cv::Mat region = img(roi);

        // calculate in-game cell dimensions
        cell_height = gheight / 8;
        cell_width = gwidth / 8;

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int y_start = r * cell_height;
                int y_end =   (r + 1) * cell_height;
                int x_start = c * cell_width;
                int x_end  =  (c + 1) * cell_width;

                cv::Rect cell_rect(x_start, y_start, x_end - x_start, y_end - y_start);
                cv::Mat cell = region(cell_rect);

                // convert to grayscale
                // cv::cvtColor(cell, cell, cv::COLOR_BGR2GRAY);
                cv::threshold(cell, cell, 100, 255, cv::THRESH_BINARY_INV);

                // if the fill ration is bigger than .5 set cell in the grid to 1, if not set it to 0,
                // pretty self-explanatory
                double fill_ratio = cv::countNonZero(cell) / (double)cell.total();
                vgrid[r][c] = (fill_ratio > 0.5) ? 0 : 1;

            }
        }

        // copy from vgrid to grid
        memcpy(grid, vgrid, sizeof(vgrid));

    }

}