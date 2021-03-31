#include<iostream>
#include<random>
#include<algorithm>
#include<iterator>
#include <argparse/argparse.hpp>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <json/reader.h>
#include <array>

#include "airsim_control.h"
#include "model_tools.h"
#include "map_util.h"
#include "common_util.h"
#include "cgal_tools.h"

#include "tqdm.h"

#include<opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <CGAL/Point_set_3/IO.h>
#include <boost/filesystem.hpp>

using namespace std;
int fieldRange = 600;

MapConverter mapConverter;

void BboxFit(std::string in_path, std::string out_path, std::map<std::string, Point_cloud>& model_point_clouds, std::map<std::string, std::vector<Point_3>>& model_bbox_corner_vertices)
{
	boost::filesystem::path myPath(in_path);
	boost::filesystem::recursive_directory_iterator endIter;
	for (boost::filesystem::recursive_directory_iterator iter(myPath); iter != endIter; iter++) {
		if (iter->path().filename().extension().string() == ".obj")
		{
			std::vector<Point_3> cornerPoints;
			std::array<Point_3, 8> obb_points;
			std::vector<Point_3> vertices;
			Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(
				load_obj(iter->path().string()));
			Point_cloud point_cloud(true);
			for (auto& item_point : mesh.points())
				vertices.push_back(item_point);
				//point_cloud.insert(item_point);

			CGAL::oriented_bounding_box(vertices, obb_points);
			model_point_clouds.insert(std::make_pair(iter->path().stem().string(), point_cloud));
			model_bbox_corner_vertices.insert(std::make_pair(iter->path().stem().string(), cornerPoints));

			//test
			Surface_mesh obb_sm;
			CGAL::make_hexahedron(cornerPoints[0], cornerPoints[1], cornerPoints[2], cornerPoints[3],
				cornerPoints[4], cornerPoints[5], cornerPoints[6], cornerPoints[7], obb_sm);
			ofstream((out_path / iter->path().stem()).string() + "_obb.off") << obb_sm;
		}
	}
}

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
	const boost::filesystem::path color_map_path(args["color_map"].asString());
	cv::Mat color_map = cv::imread(color_map_path.string());
	if (color_map.size == 0)
	{
		LOG(ERROR) << "Cannot open color map " << color_map_path << std::endl;
		return 0;
	}

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
	Height_map height_map(map_start_mesh, map_end_mesh,
		args["heightmap_resolution"].asFloat(),
		args["heightmap_dilate"].asFloat()
	);

	// For data
	const boost::filesystem::path mesh_root(args["mesh_root"].asString());
	LOG(INFO) << "Read mesh from " << mesh_root;
	Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(
		load_obj((mesh_root / "total_split.obj").string()));
	Point_cloud point_cloud(true);
	for (auto& item_point : mesh.points())
		point_cloud.insert(item_point);
	/*
	 * TODO Iterate the directory and read the individual points
	 */
	std::map<string, Point_cloud> model_point_clouds;
	std::map<string, std::vector<Point_3>> model_bbox_corner_vertices;

	BboxFit(args["mesh_root"].asString(), args["output_root"].asString(), model_point_clouds, model_bbox_corner_vertices);

	const boost::filesystem::path output_root_path(args["output_root"].asString());
	//if (boost::filesystem::exists(output_root_path))
	//	boost::filesystem::remove_all(output_root_path);
	boost::filesystem::create_directories(output_root_path);
	boost::filesystem::create_directories(output_root_path / "3d_box");

	// Prepare environment
	// Reset segmentation color, initialize map converter
	Airsim_tools* airsim_client;
	MapConverter map_converter;
	{
		map_converter.initDroneStart(DRONE_START);
		airsim_client = new Airsim_tools(DRONE_START);

		/*
		 * TODO Reset the color correctly
		 */
		airsim_client->reset_color("building");
		//	airsim_client.m_agent->simSetSegmentationObjectID("BP_Sky_Sphere", 0);
		LOG(INFO) << "Initialization done";
	}

	/*
	 * TODO Sample the environment uniformly and store
	 */

	for (;;)
	{
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
