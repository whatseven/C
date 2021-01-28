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

std::vector<cv::Vec3b> get_color_table_bgr()
{
        std::vector<cv::Vec3b> color_table;
        //color_table.emplace_back(197, 255, 255);
        //color_table.emplace_back(226, 226, 255);
        //color_table.emplace_back(255, 226, 197);
        //color_table.emplace_back(197, 255, 226);
        //color_table.emplace_back(2, 0, 160);
        //color_table.emplace_back(0, 12, 79);
        //color_table.emplace_back(105, 72, 129);
        //color_table.emplace_back(153, 0, 102);
        //color_table.emplace_back(153, 150, 102);
	
        //color_table.emplace_back(0, 0, 0);
        //color_table.emplace_back(0, 255, 0);
        //color_table.emplace_back(0, 0, 255);
        //color_table.emplace_back(2, 0, 104);
        //color_table.emplace_back(51, 0, 255);
        //color_table.emplace_back(51, 102, 255);
        //color_table.emplace_back(47, 84, 227);
        //color_table.emplace_back(25, 0, 203);
        //color_table.emplace_back(38, 137, 243);
        //color_table.emplace_back(8, 69, 231);
        //color_table.emplace_back(41, 160, 252);
        //color_table.emplace_back(0, 102, 255);
        //color_table.emplace_back(0, 206, 255);
        //color_table.emplace_back(24, 104, 235);
        
        color_table.emplace_back(153, 255, 204);
        color_table.emplace_back(255, 204, 153);
        color_table.emplace_back(153, 255, 255);
        color_table.emplace_back(253, 196, 225);
        color_table.emplace_back(0, 182, 246);
        
        return color_table;
}


