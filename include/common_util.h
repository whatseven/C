#pragma once
#include<opencv2/opencv.hpp>
#include<chrono>
#include<iostream>
#include<string>
#include<boost/filesystem.hpp>

namespace fs = boost::filesystem;
void checkFolder(const fs::path& folder);
std::chrono::steady_clock::time_point recordTime();
void profileTime(std::chrono::steady_clock::time_point& now, std::string vTip = "", bool v_profile = true);
void debug_img(std::vector<cv::Mat>& vImgs);
void override_sleep(float seconds);
std::vector<cv::Vec3b> get_color_table_bgr();