#include <iostream>
#include <regex>
#include <argparse/argparse.hpp>
#include <glog/logging.h>
#include "airsim_control.h"
#include <json/reader.h>

#include <trajectory.h>

int main(int argc,char* argv[])
{
	Json::Value args;
	{
		FLAGS_logtostderr = 1;
		google::InitGoogleLogging(argv[0]);
		argparse::ArgumentParser program("Capture image from a given trajectory");
		program.add_argument("--config_file").required();
		try {
			program.parse_args(argc, argv);
			const std::string config_file = program.get<std::string>("--config_file");

			std::ifstream in(config_file);
			if (!in.is_open()) {
				LOG(ERROR) << "Error opening file" << config_file << std::endl;
				return 0;
			}
			Json::Reader json_reader;
			if (!json_reader.parse(in, args)) {
				LOG(ERROR) << "Error parse config file" << config_file << std::endl;
				return 0;
			}
			in.close();
		}
		catch (const std::runtime_error& err) {
			std::cout << err.what() << std::endl;
			std::cout << program;
			exit(0);
		}
	}
	
	Airsim_tools airsim_tools(Eigen::Vector3f(0.f,0.f,0.f));

	MapConverter map_converter;
	map_converter.initDroneStart(Eigen::Vector3f(0.f, 0.f, 0.f));

	std::string path_file = args["path"].asString();
	std::string path_type = args["path_type"].asString();
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector2f >> trajectory;

	if (path_type=="wgs")
		trajectory = read_wgs84_trajectory(path_file);

	if (!args["geo_origin_x"].empty())
	{
		float geo_origin_x = args["geo_origin_x"].asFloat();
		float geo_origin_y = args["geo_origin_y"].asFloat();
		for(auto& item: trajectory)
		{
			Eigen::Vector2f mercator = lonLat2Mercator(Eigen::Vector2f(item.first.x(), item.first.y()));

			mercator.x() -= geo_origin_x;
			mercator.y() -= geo_origin_y;
			item.first.x() = mercator.x();
			item.first.y() = mercator.y();

			item.second[0] *= -1;
			continue;
		}
	}
	
	std::pair<Eigen::Vector3f, Eigen::Vector2f> cur_view = trajectory[0];
	int id_view = 1;
	while (id_view!= trajectory.size())
	{
		while ((cur_view.first - trajectory[id_view].first).norm() > 3)
		{
			Pos_Pack pos_pack = map_converter.get_pos_pack_from_mesh(cur_view.first, 
				-(cur_view.second[1] - 90) / 180.*M_PI, 
				cur_view.second[0] / 180. * M_PI);
			airsim_tools.adjust_pose(pos_pack);
			//airsim_tools.get_images();
			{
				using namespace msr::airlib;
				typedef ImageCaptureBase::ImageRequest ImageRequest;
				typedef ImageCaptureBase::ImageResponse ImageResponse;
				typedef ImageCaptureBase::ImageType ImageType;
				vector<ImageRequest> request = {
					ImageRequest("0", ImageType::Scene, false, false),
				};
				const vector<ImageResponse>& response = airsim_tools.m_agent->simGetImages(request);
				cv::Mat rgb = cv::Mat(response[0].height, response[0].width, CV_8UC3,
					(unsigned*)response[0].image_data_uint8.data()).clone();
				cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
				
			}
			std::cout << boost::format("%f, %f, %f, %f, %f") % pos_pack.pos_mesh.x() % pos_pack.pos_mesh.y() % pos_pack.pos_mesh.z() % pos_pack.pitch % pos_pack.yaw << std::endl;

			Eigen::Vector3f direction = trajectory[id_view].first - cur_view.first;
			cur_view.first += direction.normalized() * 5;

		}
		cur_view = trajectory[id_view];
		id_view += 1;
	}
	
		
	
	return 0;
}