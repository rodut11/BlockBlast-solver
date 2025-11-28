#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "internal_vision.hpp"
#include "../../../include/blocks.h"
#include <array>

//include image
#include <bits/fs_fwd.h>

#include "../../../assets/sample.h"

#define THRESH 0.01

typedef struct {
    int x1, y1, x2, y2;
}bounding_box;


cv::Mat remove_small_components(const cv::Mat& mask, int min_area);
// std::vector<Box> get_block_boxes(const cv::Mat& mask, int min_area);
std::array<bounding_box, 3> get_bounding_box(const cv::Mat& img, const cv::Mat& templ);

extern "C" void get_block() {
    auto buffer = grab_screencap();
    cv::Mat img = decode_screencap(buffer);

    if (!img.empty()) {
        // coordinates (top-left and bottom-right)
        int gx1 = 696;
        int gy1 = 727;
        int gx2 = 1224;
        int gy2 = 954;

        // calculate region
        cv::Point topLeft(gx1, gy1);
        cv::Point bottomRight(gx2, gy2);

        int rwidth = bottomRight.x - topLeft.x;
        int rheight = bottomRight.y - topLeft.y;

        cv::Rect roi(topLeft.x, topLeft.y, rwidth, rheight);
        cv::Mat region = img(roi);
        cv::cvtColor(region, region, cv::COLOR_BGR2GRAY);
        //cv::threshold(region, region, 130, 255, cv::THRESH_BINARY_INV);

        // load sample image
        cv::Mat sample_png_raw(1, (int)sample_png_len, CV_8UC1, (void*)sample_png);
        cv::Mat cell_template = cv::imdecode(sample_png_raw, cv::IMREAD_GRAYSCALE);
        //cv::threshold(cell_template, cell_template, 128, 255, cv::THRESH_BINARY_INV);

        if (cell_template.empty()) {
            std::cerr << "Failed to decode image" << std::endl;
            return;
        }

        std::array<bounding_box,3> bounding_boxes = get_bounding_box(region, cell_template);

        for (int i = 0; i < 3; i++) {
            std::cout << bounding_boxes[i].x1 << std::endl;
            std::cout << bounding_boxes[i].y1 << std::endl;
            std::cout << bounding_boxes[i].x2 << std::endl;
            std::cout << bounding_boxes[i].y2 << std::endl;
        }


    }
}

std::array<bounding_box, 3> get_bounding_box(const cv::Mat& img, const cv::Mat& templ) {

    std::vector<cv::Rect> cells;
    cv::Mat result;
    cv::matchTemplate(img, templ, result, cv::TM_SQDIFF_NORMED);

    cv::Mat display;
    cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);

    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    std::array<bounding_box, 3> bb{};
    int block_count = 0;

    while (true) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        if (minVal > THRESH) break;

        cv::Rect rect(minLoc.x, minLoc.y, templ.cols, templ.rows);

        cv::rectangle(mask, rect, cv::Scalar(255), cv::FILLED);

        cv::floodFill(result, minLoc, cv::Scalar(1));
        cells.push_back(rect);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40));
    cv::dilate(mask, mask, kernel);

    //get block bounding boxes
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    for (int i = 1; i < n && block_count < 3; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        bb[block_count++] = {x, y, x+w, y+h};

        cv::rectangle(display, cv::Rect(x, y, w, h), cv::Scalar(0,255,0), 2);
    }

    cv::imshow("Detected Cells", display);
    cv::waitKey(0);

    return bb;
}

