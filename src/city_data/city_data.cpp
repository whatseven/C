#include<iostream>
#include<random>
#include<algorithm>
#include<iterator>
#include <regex>
#include <argparse/argparse.hpp>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <json/reader.h>

#include "airsim_control.h"
#include "model_tools.h"
#include "map_util.h"
#include "common_util.h"

#include "tqdm.h"

#include<opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <CGAL/Point_set_3/IO.h>

using namespace std;

MapConverter mapConverter;

int main(int argc, char* argv[])
{
	srand(0);

	// Read arguments
	Json::Value args;
	{
		FLAGS_logtostderr = 1;
		google::InitGoogleLogging(argv[0]);
		argparse::ArgumentParser program("Record data in unreal");
		program.add_argument("--config_file").required();
		try
		{
			program.parse_args(argc, argv);
			boost::filesystem::path config_file(program.get<std::string>("--config_file"));

			std::ifstream in(config_file.string());
			if (!in.is_open())
			{
				LOG(ERROR) << "Error opening file " << config_file << std::endl;
				return 0;
			}
			Json::Reader json_reader;
			if (!json_reader.parse(in, args))
			{
				LOG(ERROR) << "Error parse config file " << config_file << std::endl;
				return 0;
			}
			in.close();
		}
		catch (const std::runtime_error& err)
		{
			std::cout << err.what() << std::endl;
			std::cout << program;
			exit(0);
		}
	}

	// For unreal
	const Eigen::Vector3f DRONE_START(args["DRONE_START_X"].asFloat(), args["DRONE_START_Y"].asFloat(),
	                                  args["DRONE_START_Z"].asFloat());
	const Eigen::Vector3f map_start_unreal(args["MAP_START_UNREAL_X"].asFloat(), args["MAP_START_UNREAL_Y"].asFloat(),
	                                       args["MAP_START_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_end_unreal(args["MAP_END_UNREAL_X"].asFloat(), args["MAP_END_UNREAL_Y"].asFloat(),
	                                     args["MAP_END_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_start_mesh(map_start_unreal.x() / 100.f, -map_end_unreal.y() / 100.f,
	                                     map_start_unreal.z() / 100.f);
	const Eigen::Vector3f map_end_mesh(map_end_unreal.x() / 100.f, -map_start_unreal.y() / 100.f,
	                                   map_end_unreal.z() / 100.f);

	// For data
	const boost::filesystem::path mesh_root(args["mesh_root"].asString());
	LOG(INFO) << "Read mesh from " << mesh_root;
	Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(
		load_obj((mesh_root / "total_split.obj").string()));
	Point_cloud point_cloud(true);
	for (auto& item_point : mesh.points())
		point_cloud.insert(item_point);
	Height_map height_map(point_cloud, args["heightmap_resolution"].asFloat(), args["heightmap_dilate"].asFloat());
	
	/*
	 * TODO Iterate the directory and read the individual points
	 */


	const boost::filesystem::path output_root_path(args["output_root"].asString());
	//if (boost::filesystem::exists(output_root_path))
	//	boost::filesystem::remove_all(output_root_path);
	boost::filesystem::create_directories(output_root_path);
	boost::filesystem::create_directories(output_root_path / "3d_box");

	// Prepare environment
	// Reset segmentation color, initialize map converter
	Airsim_tools* airsim_client;
	MapConverter map_converter;
	std::map<cv::Vec3b, std::string> color_to_mesh_name_map;
	{
		map_converter.initDroneStart(DRONE_START);
		airsim_client = new Airsim_tools(DRONE_START);
		const boost::filesystem::path color_map_path(args["color_map"].asString());
		airsim_client->m_color_map = cv::imread(color_map_path.string());
		if (airsim_client->m_color_map.size == 0)
		{
			LOG(ERROR) << "Cannot open color map " << color_map_path << std::endl;
			return 0;
		}
		cv::cvtColor(airsim_client->m_color_map, airsim_client->m_color_map, cv::COLOR_BGR2RGB);

		std::string building_key_world = args["building_keyword"].asString();
		if(building_key_world.size()==0)
			color_to_mesh_name_map = airsim_client->reset_color([](std::string v_name)
			{
				std::regex rx("^[0-9]+$");
				bool bl = std::regex_match(v_name.begin(), v_name.end(), rx);
				return bl;
			});
		else
			color_to_mesh_name_map = airsim_client->reset_color(building_key_world);
		LOG(INFO) << "Initialization done";
	}

	/*
	 * TODO Sample the environment uniformly and store
	 */
	std::vector<Pos_Pack> place_to_be_travel;
	{
		int delta_x = (map_end_unreal - map_start_unreal).x()/ args["COLLECTION_STEP_X"].asFloat();
		int delta_y = (map_end_unreal - map_start_unreal).y()/ args["COLLECTION_STEP_Y"].asFloat();
		int delta_z = (map_end_unreal - map_start_unreal).z()/ args["COLLECTION_STEP_Z"].asFloat();
		
		for (int id_x=0;id_x<delta_x;id_x+=1)
			for (int id_y=0;id_y<delta_y;id_y+=1)
				for (int id_z=0;id_z<delta_z;id_z+=1)
				{
					Eigen::Vector3f cur_pos_unreal = map_start_unreal + Eigen::Vector3f(
						id_x * args["COLLECTION_STEP_X"].asFloat(),
						id_y * args["COLLECTION_STEP_Y"].asFloat(),
						id_z * args["COLLECTION_STEP_Z"].asFloat()
					);

					Pos_Pack pos_pack = mapConverter.get_pos_pack_from_unreal(cur_pos_unreal, 0, 0);
					if(!height_map.in_bound(pos_pack.pos_mesh.x(), pos_pack.pos_mesh.y()))
						continue;
					if(height_map.in_bound(pos_pack.pos_mesh.x(), pos_pack.pos_mesh.y()) + 30> pos_pack.pos_mesh.z())
						continue;

					for (float pitch_degree = 0;pitch_degree < 60;pitch_degree += args["COLLECTION_STEP_PITCH_DEGREE"].asFloat())
						for (float yaw_degree = 0;yaw_degree < 360;yaw_degree += args["COLLECTION_STEP_YAW_DEGREE"].asFloat())
							place_to_be_travel.push_back(mapConverter.get_pos_pack_from_unreal(
								cur_pos_unreal, 
								yaw_degree / 180 * M_PI,
								pitch_degree/180*M_PI));
				}
	}


	for (const Pos_Pack& item_pos_pack:place_to_be_travel)
	{
		adjust_pose(*airsim_client->m_agent, item_pos_pack);
		/*
		 * TODO Calculate the initial 3d bbox and store
		 */

		/*
		 * TODO Collect 3d bounding box
		 */

		/*
		* TODO Collect depth
		*/

		/*
		* TODO Collect voxel, point cloud in camera space(visible and invisible) and sdf value
		*/
	}


	return 0;
}
