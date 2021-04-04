#include<iostream>
#include <vector>
#include <array>
#include <map>

#include<random>
#include<algorithm>
#include<iterator>
#include <argparse/argparse.hpp>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <json/reader.h>

#include "airsim_control.h"
#include "model_tools.h"
#include "map_util.h"
#include "common_util.h"
#include "cgal_tools.h"

#include "tqdm.h"

#include<opencv2/opencv.hpp>
#include <CGAL/Point_set_3/IO.h>
#include <boost/filesystem.hpp>

void writeBbox(const std::string out_path, const std::vector<Point_3> cornerPoints)
{
	std::fstream outFile(out_path, std::ios::out);
	for (int i = 0; i < 8; i++)
	{
		outFile << cornerPoints[i].x() << " " << cornerPoints[i].y() << " " << cornerPoints[i].z() << "\n";
	}
}

void BboxFit(std::string in_path, std::string out_path, std::map<string, Point_cloud>& model_point_clouds, std::map<string, std::vector<Point_3>>& model_bbox_corner_vertices)
{
	boost::filesystem::path myPath(in_path);
	boost::filesystem::recursive_directory_iterator endIter;
	for (boost::filesystem::recursive_directory_iterator iter(myPath); iter != endIter; iter++) {
		if (iter->path().filename().extension().string() == ".obj" && (std::atoi(iter->path().stem().string().c_str()) || iter->path().stem().string() == "0"))
		{
			std::vector<Point_3> cornerPoints;
			std::array<Point_3, 8> obb_points;
			std::vector<float> verticesZ;
			std::vector<cv::Point2f> vertices2D;
			cv::Point2f cornerPoints2D[4];
			Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(
				load_obj(iter->path().string()));
			Point_cloud point_cloud(true);
			for (auto& item_point : mesh.points())
			{
				verticesZ.push_back(item_point.z());
				vertices2D.push_back(cv::Point2f(item_point.x(), item_point.y()));
				point_cloud.insert(item_point);
			}
			cv::RotatedRect box = cv::minAreaRect(vertices2D);
			float maxZ = *std::max_element(verticesZ.begin(), verticesZ.end());
			float minZ = *std::min_element(verticesZ.begin(), verticesZ.end());
			box.points(cornerPoints2D);
			for (int i = 0; i < 4; i++)
			{
				float z = minZ;
				for (int j = 0; j < 2; j++)
				{
					float x = cornerPoints2D[i].x;
					float y = cornerPoints2D[i].y;
					cornerPoints.push_back(Point_3(x, y, z));
					z = maxZ;
				}
			}
			model_point_clouds.insert(std::make_pair(iter->path().stem().string(), point_cloud));
			model_bbox_corner_vertices.insert(std::make_pair(iter->path().stem().string(), cornerPoints));
			writeBbox((out_path / iter->path().stem()).string() + ".xyz", cornerPoints);
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
	//Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(
	//	load_obj((mesh_root / "total_split.obj").string()));
	//Point_cloud point_cloud(true);
	//for (auto& item_point : mesh.points())
	//	point_cloud.insert(item_point);
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
