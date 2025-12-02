#pragma once
#include <array>
#include <opencv2/core.hpp>

typedef struct {
    int x1, y1, x2, y2;
} bounding_box;

std::array<bounding_box, 3> get_bounding_box(const cv::Mat& img, const cv::Mat& templ);
