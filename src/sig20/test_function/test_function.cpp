#include "trajectory.h"


int main(int argc, char** argv) {
	cv::Mat v_map(10, 11, CV_8UC1, cv::Scalar(1));
	v_map.at<cv::uint8_t>(5, 6) = 0;
	v_map.at<cv::uint8_t>(5, 7) = 0;
	print_map(v_map);
	perform_ccpp(v_map, Eigen::Vector2i(9,9), Eigen::Vector2i(2,2));
}
