//
// Created by rodut11 on 11/6/25.
//

#ifndef C_INTERNAL_VISION_H
#define C_INTERNAL_VISION_H

#include <opencv2/imgcodecs.hpp>

cv::Mat decode_screencap(const std::vector<unsigned char>& buffer);
std::vector<unsigned char> grab_screencap();

#endif //C_INTERNAL_VISION_H