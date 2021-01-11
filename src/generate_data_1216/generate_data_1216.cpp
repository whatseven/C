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
const string output_root_path = "D:\\Siggraph\\data\\Object_detection_data\\";
const string host = "127.0.0.1";
const int sample_num = 27441;
float safe_bounds = 0;
const float max_pitch = 60;
const bool time_profile = true;
const int model_num = 760;
const int img_width = 800;
const int img_height = 800;
const int start_id = 0;

int main(int argc, char* argv[]) {
	mapConverter.initDroneStart(Eigen::Vector3f(-68000.0, 8000.0, 1000.0));
	srand((unsigned int)(time(NULL)));

	Airsim_tools airsim_client(Eigen::Vector3f(-68000.0, 8000.0, 1000.0));

	// Init segmentation color
	//{
	//	std::map<std::string, int> color_map;
	//	std::map<std::string, int> background_color_map;
	//	for (const auto& name : airsim_client.m_agent->simListSceneObjects()) {
	//		if (name[0] == 's')
	//			color_map.insert(std::make_pair(name, rand() % 255 + 1));
	//		else
	//			background_color_map.insert(std::make_pair(name, 0));
	//		background_color_map.insert(std::make_pair(name, 0));

	//	}
	//	//airsim_client.m_agent->simSetSegmentationObjectIDMultiple(background_color_map);
	//	//airsim_client.m_agent->simSetSegmentationObjectIDMultiple(color_map);
	//	airsim_client.m_agent->simSetSegmentationObjectID("Plane", 0);
	//	airsim_client.m_agent->simSetSegmentationObjectID("BP_Sky_Sphere", 0);
	//	/*for (int i = 0; i < model_num; i++)
	//	{
	//		airsim_client.m_agent->simSetSegmentationObjectID(std::to_string(i), rand()%255 + 1);
	//	}*/
	//}

	cv::Mat colorFile = cv::imread(color_map_path);

	//Get Bounds
	std::vector<float> bounds = get_bounds(model_path, safe_bounds);

	// Get HeightMap
	std::ifstream inPointFile(point_set_path);
	Point_set sample_points;
	CGAL::read_ply_point_set(inPointFile, sample_points);
	Height_map heightMap(sample_points, 0.5);

	//Chicago modify
	{
		/*float temp = bounds[2];
		bounds[2] = -bounds[3];
		bounds[3] = -temp;*/
		for (int i = 0; i < bounds.size(); i++)
		{
			bounds[i] *= 100;
		}
		safe_bounds *= 100;
		bounds[0] = -110000;
		bounds[1] = -20000;
	}

	std::ofstream fTrain(output_root_path + "my_train.txt");
	std::ofstream fPose((output_root_path + "pose.txt"));

	//Start iter
	tqdm bar;
	int img_index = start_id;
	auto generator = std::mt19937();
	std::uniform_real_distribution<> dis(0, 1);
	while (img_index < sample_num + start_id) {
		boost::format fmt("%012d");
		std::string imgIndex = (fmt % img_index).str();

		// Get variable
		Eigen::Vector3f posUnreal(
			dis(generator) * (bounds[1] - bounds[0]) + bounds[0],
			dis(generator) * (bounds[3] - bounds[2]) + bounds[2],
			dis(generator) * (bounds[4] * 0.7 - safe_bounds) + safe_bounds
		);


		//posUnreal = mapConverter.convertMeshToUnreal(posUnreal);
		float pitch = (dis(generator) * max_pitch) * M_PI / 180.f;
		float yaw = (dis(generator) * 4) * M_PI / 2;

		while (posUnreal.z() <= heightMap.get_height(posUnreal.x(), posUnreal.y()))
		{
			posUnreal[2] += 500;
		}

		Pos_Pack pos_pack = mapConverter.get_pos_pack_from_unreal(posUnreal, yaw, pitch);

		

		auto timeRecorder = recordTime();

		//Adjust Pose
		airsim_client.adjust_pose(pos_pack);

		std::map<std::string, cv::Mat> images = airsim_client.get_images();
		while (images.size() == 0)
			images = airsim_client.get_images();
		profileTime(timeRecorder, "images", time_profile);

		//Cluster
		std::vector<cv::Vec3b> color_map;
		std::vector<std::vector<cv::Point>> bboxes_points;
		std::vector<CGAL::Bbox_2> building_bboxes;
		int null_area = 0;
		for (int y = 0; y < images["segmentation"].rows; ++y) {
			for (int x = 0; x < images["segmentation"].cols; ++x) {
				cv::Vec3b point_color = images["segmentation"].at<cv::Vec3b>(y, x);
				if (point_color == cv::Vec3b(130, 219, 128))
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
		if (null_area > img_width * img_height * 0.7)
			continue;
		bool invalid_flag = false;
		//if (bboxes_points.size() > 40)
		//	continue;
		//int temp_id = 0;

		for (const auto& pixel_points : bboxes_points) {
			cv::Rect now_rect = cv::boundingRect(pixel_points);
			if (now_rect.area() < img_width * img_height * 0.3)
			{
				if (now_rect.area() < img_width * img_height / 400)
					continue;
				cv::rectangle(images["segmentation"], now_rect, cv::Scalar(0, 0, 255));
				building_bboxes.push_back(CGAL::Bbox_2(now_rect.x, now_rect.y, now_rect.x + now_rect.width, now_rect.y + now_rect.height));
			}
			else
			{
				cv::Rect temp_rect = cv::boundingRect(pixel_points);
				std::vector<std::vector<bool>> is_same_color(temp_rect.height, std::vector<bool>(temp_rect.width, false));
				for (auto pixel : pixel_points)
					is_same_color[pixel.y - temp_rect.y][pixel.x - temp_rect.x] = true;

				std::vector<cv::Rect> rects = getBoundingBoxes(pixel_points, is_same_color);
				//if (rects.size() >= 3)
				//{
				//	invalid_flag = true;
				//	break;
				//}
				for (auto rect : rects)
				{
					cv::rectangle(images["segmentation"], rect, cv::Scalar(0, 0, 255));
					building_bboxes.push_back(CGAL::Bbox_2(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height));
					//temp_id++;
				}
			}
		}
		cv::imwrite("./temp_debug.png", images["segmentation"]);

		if (building_bboxes.size() <= 2 || invalid_flag)
			continue;

		//
		// Write
		//
		writeItem_1216(output_root_path + "Annotations\\" + imgIndex + ".xml", building_bboxes,
			images["rgb"].size().width);

		cv::imwrite(output_root_path + "JPEGImages\\" + imgIndex + ".jpg", images["rgb"]);
		fTrain << imgIndex << std::endl;

		fPose << (boost::format("%f,%f,%f,%f,%f") % pos_pack.pos_mesh[0] % pos_pack.pos_mesh[1] % pos_pack.pos_mesh[2] % yaw % pitch).str() << std::endl;

		bar.progress(img_index - start_id, sample_num);

		img_index += 1;
	}

	fTrain.close();
	fPose.close();

	return 0;
}

//// Test graph algorithm
//int main() 
//{
//	std::vector<cv::Point> pixel_points;
//	pixel_points.push_back(cv::Point(0, 0));
//	pixel_points.push_back(cv::Point(0, 1));
//	pixel_points.push_back(cv::Point(1, 0));
//	pixel_points.push_back(cv::Point(1, 1));
//	pixel_points.push_back(cv::Point(3, 2));
//	pixel_points.push_back(cv::Point(3, 3));
//	pixel_points.push_back(cv::Point(2, 3));
//	pixel_points.push_back(cv::Point(4, 0));
//	pixel_points.push_back(cv::Point(0, 4));
//	pixel_points.push_back(cv::Point(1, 4));
//	std::vector<cv::Rect> result = getBoundingBoxes(pixel_points, 5, 5);
//	return 0;
//}