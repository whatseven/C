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
#include<iostream>
#include<random>
#include<algorithm>
#include<iterator>

#include "object_detection_tools.h"
#include "airsim_control.h"
#include "model_tools.h"
#include "map_util.h"
#include "common_util.h"

#include "tqdm.h"

#include <argparse/argparse.hpp>
#include<opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <boost/format.hpp>
#include <CGAL/Point_set_3/IO.h>

using namespace std;
int fieldRange = 600;

MapConverter mapConverter;

const string color_map_path = "F:\\Unreal\\sndd\\Env\\Plugins\\AirSim\\Content\\HUDAssets\\seg_color_palette.png";
const string model_path = "D:\\Siggraph\\data\\RAW\\Suzhou\\total_split.obj";
const string point_set_path = "D:\\Siggraph\\data\\RAW\\Suzhou\\sample.ply";
const string output_root_path = "F:\\Sig\\YingRenShi\\";
const string host = "127.0.0.1";
const int sample_num = 27441;
float safe_bounds = 0;
const float max_pitch = 60;
const bool time_profile = true;
const int model_num = 760;
const int img_width = 800;
const int img_height = 800;
const int start_id = 0;

std::vector<cv::Rect2f> process_results(std::string input)
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
	std::vector<cv::Rect2f> result;
	for (auto label : labels)
	{
		now_string = label;
		cv::Rect2f box;
		std::string::size_type position1 = now_string.find_first_of("[");
		std::string::size_type position2 = now_string.find_first_of(",");
		float xmin = atof((now_string.substr(position1 + 1, position2 - position1 - 1)).c_str());
		now_string = now_string.substr(position2 + 2);
		position2 = now_string.find_first_of(",");
		float ymin = atof((now_string.substr(0, position2 - 1)).c_str());
		now_string = now_string.substr(position2 + 2);
		position2 = now_string.find_first_of(",");
		float xmax = atof((now_string.substr(0, position2 - 1)).c_str());
		now_string = now_string.substr(position2 + 2);
		position2 = now_string.find_first_of("]");
		float ymax = atof((now_string.substr(0, position2 - 1)).c_str());
		box.x = xmin;
		box.y = ymin;
		box.width = xmax - xmin;
		box.height = ymax - ymin;
		result.push_back(box);
	}
	return result;
}

int main(int argc, char** argv) {
	fstream txt_file(output_root_path + "VOC2007\\my_test.txt", fstream::out);
	for (int id = 0; id < 711; id++)
	{
		txt_file << std::to_string(id) + "_rgb" << std::endl;
		cv::Mat input_image = cv::imread(output_root_path + std::to_string(id) + "_seg.png");
		std::vector<cv::Vec3b> color_map;
		std::vector<std::vector<cv::Point>> bboxes_points;
		std::vector<CGAL::Bbox_2> building_bboxes;
		int null_area = 0;
		for (int y = 0; y < input_image.rows; ++y) {
			for (int x = 0; x < input_image.cols; ++x) {
				cv::Vec3b point_color = input_image.at<cv::Vec3b>(y, x);
				if (point_color == cv::Vec3b(57, 181, 55))
				{
					null_area++;
					continue;
				}
				auto find_result = std::find(color_map.begin(), color_map.end(), point_color);
				if (find_result == color_map.end()) {
					color_map.push_back(point_color);
					std::vector<cv::Point> new_building(1, cv::Point(x, y));
					bboxes_points.push_back(new_building);
				}
				else
				{
					int id = find_result - color_map.begin();
					bboxes_points[id].push_back(cv::Point(x, y));
				}
			}
		}

		for (const auto& pixel_points : bboxes_points) {
			cv::Rect now_rect = cv::boundingRect(pixel_points);
			if (now_rect.area() < img_width * img_height / 400)
				continue;
			cv::rectangle(input_image, now_rect, cv::Scalar(0, 0, 255));
			building_bboxes.push_back(CGAL::Bbox_2(now_rect.x, now_rect.y, now_rect.x + now_rect.width, now_rect.y + now_rect.height));
		}

		//
		// Write
		//
		writeItem_1216(output_root_path + std::to_string(id) + "_rgb.xml", building_bboxes,
			input_image.size().width);
	}
	
	return 0;
}
