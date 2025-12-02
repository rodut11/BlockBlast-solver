#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "internal_vision.hpp"
#include <array>
#include "../../include/blocks.h"

//include sample image as bytes (converted using xxd -i)
#include "../../assets/sample.h"

#define THRESH 0.8

typedef struct {
    int x1, y1, x2, y2;
}bounding_box;

cv::Mat remove_small_components(const cv::Mat& mask, int min_area);
std::array<bounding_box, 3> get_bounding_box(const cv::Mat& img, const cv::Mat& templ);

extern "C" void get_block() {
    std::vector<u_char> buffer = grab_screencap();
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

        // rwidth = region-width
        // rheight = region-height
        // I don't know why I needed to note this, I just thought I should, just in case :)
        int rwidth = bottomRight.x - topLeft.x;
        int rheight = bottomRight.y - topLeft.y;

        // get region and convert to grayscale
        cv::Rect roi(topLeft.x, topLeft.y, rwidth, rheight);
        cv::Mat region = img(roi);
        //cv::cvtColor(region, region, cv::COLOR_BGR2GRAY);
        //cv::threshold(region, region, 130, 255, cv::THRESH_BINARY_INV);

        // load sample image
        cv::Mat sample_png_raw(1, (int)sample_png_len, CV_8UC1, (void*)sample_png);
        cv::Mat cell_template = cv::imdecode(sample_png_raw, cv::IMREAD_GRAYSCALE);
        //cv::threshold(cell_template, cell_template, 128, 255, cv::THRESH_BINARY_INV);


        if (cell_template.empty()) {
            std::cerr << "Failed to decode image" << std::endl;
            return;
        }

        // get bounding boxes
        std::array<bounding_box,3> bounding_boxes = get_bounding_box(region, cell_template);

        for (int i = 0; i < 3; i++) {
            // read coordinate, temporary
            // std::cout << bounding_boxes[i].x1 << std::endl;
            // std::cout << bounding_boxes[i].y1 << std::endl;
            // std::cout << bounding_boxes[i].x2 << std::endl;
            // std::cout << bounding_boxes[i].y2 << std::endl;
            // std::cout << "------------------" << std::endl;

            // get region on interest
            cv::Rect box_roi(
                bounding_boxes[i].x1 + gx1, // offset by region top-left x
                bounding_boxes[i].y1 + gy1, // offset by region top-left y
                bounding_boxes[i].x2 - bounding_boxes[i].x1,
                bounding_boxes[i].y2 - bounding_boxes[i].y1
            );

            // crop to region
            cv::Mat box_region = img(box_roi);

            // match template
            cv::Mat result;
            cv::matchTemplate(box_region, cell_template, result, cv::TM_CCOEFF_NORMED);

            // convert output image (temporary) to grayscale
            cv::Mat display;
            cv::cvtColor(box_region, display, cv::COLOR_GRAY2BGR);

            // create single-channel mask
            cv::Mat mask = cv::Mat::zeros(box_region.size(), CV_8UC1);

            // detect cells
            while (true) {
                // compute minVal, maxVal, minLoc, maxLoc
                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                if (maxVal < THRESH) break;

                cv::Rect rect(maxLoc.x, maxLoc.y, cell_template.cols, cell_template.rows);

                // fill already detected pixel so it's not detected again and not fall in a loop
                cv::rectangle(box_region, rect, cv::Scalar(255), cv::FILLED);
                cv::floodFill(result, maxLoc, cv::Scalar(-1));

                // write on display
                cv::rectangle(display, rect, cv::Scalar(255), cv::FILLED);
                cv::rectangle(mask, rect, cv::Scalar(255), cv::FILLED);

            }

             cv::imshow("block", mask);
             cv::waitKey(0);
        }

    }
}

std::array<bounding_box, 3> get_bounding_box(const cv::Mat& img, const cv::Mat& templ) {

    // match template
    cv::Mat result;

    cv::matchTemplate(img, templ, result, cv::TM_CCOEFF_NORMED);

    //create display image (matrix)
    //created so that stuff could be drawn on it
    cv::Mat display;
    cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);

    // create mask
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    std::array<bounding_box, 3> bb{};
    int block_count = 0;

    while (true) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // TM_CCOEFF_NORMED works better when maxVal < THRESH, in this case 0.8
        if (maxVal < THRESH) break;

        cv::Rect rect(maxLoc.x, maxLoc.y, templ.cols, templ.rows);

        // draw on mask and result so cells don't get redetected and fall in a loop
        cv::rectangle(mask, rect, cv::Scalar(255), cv::FILLED);
        cv::floodFill(result, maxLoc, cv::Scalar(-1));
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40));
    cv::dilate(mask, mask, kernel);

    //get block bounding boxes
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    // get top-left and bottom-right coords in
    for (int i = 1; i < n && block_count < 3; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // store in bb array top-left and bottom-right coords
        // bottom right is calculated by adding the width to the top-left x and height to the top-left y
        // I don't know if this makes any sense
        bb[block_count++] = {x, y, x+w, y+h};
    }

    return bb;
}