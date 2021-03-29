#include <iostream>
#include <regex>
#include <glog/logging.h>
#include "airsim_control.h"

int main(int argc,char* argv[])
{
	google::InitGoogleLogging(argv[0]);

	Airsim_tools airsim_tools(Eigen::Vector3f(0.f,0.f,0.f));
	airsim_tools.reset_color("building");

	//airsim_tools.reset_color([](std::string v_name)
	//{
	//	std::regex rx("[0-9]+");
	//	bool bl = std::regex_match(v_name.begin(), v_name.end(), rx);
	//	return bl;
	//});
	//airsim_tools.m_agent->simSetSegmentationObjectID("0", 0);
	MapConverter map_converter;
	map_converter.initDroneStart(Eigen::Vector3f(0.f, 0.f, 0.f));

	Eigen::Vector3f center(-10000.f, -14000.f, -1000.f);
	float radius = 5000.f;

	fs::path root("D:\\Paper\\ICCV_2021\\Yingrenshi_demo\\video_fov_90");
	checkFolder(root);
	checkFolder(root/"rgb");
	checkFolder(root/"depth");
	checkFolder(root/"segmentation");
	checkFolder(root/"pose");

	int idx = 0;
	//for(float i=-100.f;i<50.f;i+=.5f)
	for(float i=150.f;i<190.f;i+=.5f)
	{
		float x = radius * std::sin(i / 180.f * M_PI);
		float y = radius * std::cos(i / 180.f * M_PI);
		x += center.x();
		y += center.y();
		float z = 5000.f;
		
		Eigen::Vector3f direction = center - Eigen::Vector3f(x, y, z);
		direction.normalize();
		float yaw = std::atan2(direction[1], direction[0]);
		float pitch = std::atan2(direction[2], std::sqrt(direction[1] * direction[1] + direction[0] * direction[0]));

		Pos_Pack pos_pack = map_converter.get_pos_pack_from_unreal(Eigen::Vector3f(x, y, z),
			yaw,
			-pitch);

		airsim_tools.adjust_pose(pos_pack);
		auto imgs=airsim_tools.get_images();
		cv::imwrite((root / "rgb" / (std::to_string(idx) + ".png")).string(), imgs.at("rgb"));
		cv::imwrite((root / "depth" / (std::to_string(idx) + ".tiff")).string(), imgs.at("depth_planar"));
		cv::imwrite((root / "segmentation" / (std::to_string(idx) + ".png")).string(), imgs.at("segmentation"));
		std::ofstream f_pose((root / "pose" / (std::to_string(idx) + ".txt")).string());
		f_pose << pos_pack.camera_matrix.matrix();
		f_pose.close();
		idx += 1;
	}
	
	return 0;
}