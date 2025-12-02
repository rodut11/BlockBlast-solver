#include <iostream>
#include <opencv2/imgcodecs.hpp>

cv::Mat decode_screencap(const std::vector<unsigned char>& buffer) {
    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);

    // check if image is empty
    if (img.empty()) {
        std::cerr << "Failed to decode image\n";
    }

    return img;
}

std::vector<unsigned char> grab_screencap() {
    FILE *fp = popen("adb exec-out screencap -p", "r");
    if (!fp) {
        std::cerr << "Failed to run popen\n";
        exit(EXIT_FAILURE);
    }

    std::vector<unsigned char> buffer;
    u_char tmp[4096];
    size_t n;

    while ((n = fread(tmp, 1, sizeof(tmp), fp)) > 0) {
        buffer.insert(buffer.end(), tmp, tmp+n);
    }
    pclose(fp);
    return buffer;
}