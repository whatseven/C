//#include "trajectory.h"
//
//
//int main(int argc, char** argv) {
//	cv::Mat v_map(10, 11, CV_8UC1, cv::Scalar(1));
//	v_map.at<cv::uint8_t>(5, 6) = 0;
//	v_map.at<cv::uint8_t>(5, 7) = 0;
//	print_map(v_map);
//	perform_ccpp(v_map, Eigen::Vector2i(9,9), Eigen::Vector2i(2,2));
//}
#include <cpr/cpr.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

std::string process_results(std::string input)
{
	std::string now_string = input;
	std::vector<std::string> labels;
	std::string::size_type position = now_string.find("], ");
	while (position != now_string.npos)
	{
		labels.push_back(now_string.substr(0, position + 3));
		now_string = now_string.substr(position + 3);
		position = now_string.find("], ");
	}
	if (now_string.length() > 10)
		labels.push_back(now_string);
	for (auto label : labels)
	{

	}
	return "success";
}

int main(int argc, char** argv) {
	/*cv::Mat img(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int y = 100; y < 150; y++)
		img.at<cv::Vec3b>(y, 100) = cv::Vec3b(0, 0, 255);*/
	for (int i = 0; i < 6; i++)
	{
		cv::Mat img = cv::imread("D:\\Siggraph\\paper_temp\\" + std::to_string(i) +".png");
		cv::resize(img, img, cv::Size(800, 800));
		cv::imwrite("D:\\Siggraph\\paper_temp\\" + std::to_string(i) + ".jpg", img);
		std::vector<uchar> data(img.ptr(), img.ptr() + img.size().width * img.size().height * img.channels());
		std::string s(data.begin(), data.end());

		/*auto r = cpr::Get(cpr::Url{ "http://172.31.224.4:10000/index" },
			cpr::Parameters{ {"img",s} });*/
		auto r = cpr::Post(cpr::Url{ "http://172.31.224.4:10000/index" },
			cpr::Body{ s },
			cpr::Header{ {"Content-Type", "text/plain"} });
		process_results(r.text);
		std::cout << r.text << std::endl;
	}

	return 0;
}
