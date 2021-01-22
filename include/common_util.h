#pragma once
#include<opencv2/opencv.hpp>
#include<chrono>
#include<iostream>
#include<string>
#include<boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>

namespace fs = boost::filesystem;
void checkFolder(const fs::path& folder);
std::chrono::steady_clock::time_point recordTime();
void profileTime(std::chrono::steady_clock::time_point& now, std::string vTip = "", bool v_profile = true);
void debug_img(std::vector<cv::Mat>& vImgs);
void override_sleep(float seconds);
std::vector<cv::Vec3b> get_color_table_bgr();

namespace std {

    template <typename Scalar, int Rows, int Cols>
    struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
        // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
        size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
            size_t seed = 0;
            for (size_t i = 0; i < matrix.size(); ++i) {
                Scalar elem = *(matrix.data() + i);
                seed ^=
                    std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

}  // namespace std