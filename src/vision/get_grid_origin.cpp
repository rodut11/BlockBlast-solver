#include <opencv2/imgcodecs.hpp>
#include "internal_vision.hpp"

extern "C" void get_grid_origin(int* x, int* y) {

    std::vector<u_char> buffer = grab_screencap();
    cv::Mat img = decode_screencap(buffer);

    if (!img.empty()) {

        // coordinates (top-left and bottom-right)
        int gx1 = 716,
            gy1 = 206,
            gx2 = 1163,
            gy2 = 654;

        int r = 0;
        int c = 0;

        // calculate region
        cv::Point topLeft(gx1, gy1);
        cv::Point bottomRight(gx2, gy2);

        int width = bottomRight.x - topLeft.x;
        int height = bottomRight.y - topLeft.y;
        cv::Rect roi(topLeft.x, topLeft.y, width, height);

        cv::Mat region = img(roi);

        // compute origin cell (0,0)
        *x = gx1 + ((float)c + 0.5 * (width/8.0));
        *y = gy1 + ((float)r + 0.5 * (height/8.0));
    }
}