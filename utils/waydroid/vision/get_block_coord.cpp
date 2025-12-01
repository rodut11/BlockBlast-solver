#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "internal_vision.hpp"
#include "../../../assets/sample.h"
#include "get_bounding_box.hpp"
#include "../../../include/blocks.h"
#include <array>

extern "C" void get_block_coord() {
    std::vector<u_char> buffer = grab_screencap();
    cv::Mat img = decode_screencap(buffer);

    int gx1 = 696;
    int gy1 = 727;

    cv::Rect roi(gx1, gy1, 1224 - gx1, 954 - gy1);
    cv::Mat region = img(roi);

    cv::Mat sample_png_raw(1, (int)sample_png_len, CV_8UC1, (void*)sample_png);
    cv::Mat cell_template = cv::imdecode(sample_png_raw, cv::IMREAD_GRAYSCALE);

    if (cell_template.empty()) {
        std::cerr << "Failed to decode image" << std::endl;
        return;
    }
    std::array<undefined_block, 3> blocks;
    std::array<bounding_box,3> bounding_boxes = get_bounding_box(region, cell_template);

    for (int i = 0; i < 3; i++) {
        int width = bounding_boxes[i].x2 - bounding_boxes[i].x1;
        int height = bounding_boxes[i].y2 - bounding_boxes[i].y1;

        int xc = gx1 + bounding_boxes[i].x1 + width / 2;
        int yc = gy1 + bounding_boxes[i].y1 + height / 2;

        blocks[i].sx = xc;
        blocks[i].sy = yc;
    }
}
