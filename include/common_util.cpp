#pragma once

#include "common_util.h"

#ifdef _WIN32
#include <time.h>
#else
#include <time.h>
#endif

void checkFolder(const fs::path& folder) {
    if (fs::is_directory(folder)) {
        fs::remove_all(folder);
    }
    fs::create_directories(folder);
}

std::chrono::steady_clock::time_point recordTime() {
    std::chrono::steady_clock::time_point  now = std::chrono::steady_clock::now();
    return now;
}

void profileTime(std::chrono::steady_clock::time_point& now,std::string vTip,bool v_profile) {
    if (!v_profile)
        return;

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);
    std::cout << vTip <<": "<<  time_span.count() << std::endl;
    now= std::chrono::steady_clock::now();
}

void debug_img(std::vector<cv::Mat>& vImgs) {
    cv::namedWindow("Debug", cv::WINDOW_NORMAL);
    cv::resizeWindow("Debug", 800 * vImgs.size(), 800);
    cv::Mat totalImg;

    cv::hconcat(vImgs, totalImg);

    cv::imshow("Debug", totalImg);
    cv::waitKey(0);
    cv::destroyWindow("Debug");
}

void override_sleep(float seconds)
{
#ifdef _WIN32
    _sleep(seconds*1000);
#else
    sleep(seconds);
#endif
}
