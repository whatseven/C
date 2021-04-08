#include<iostream>
#include <vector>
#include <array>
#include <map>

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
#include "cgal_tools.h"

#include "tqdm.h"

#include<opencv2/opencv.hpp>
#include <CGAL/Point_set_3/IO.h>
#include <boost/filesystem.hpp>

float calAngelWithX(Vector_3 in)
{
	float product = Vector_3(1, 0, 0) * in;
	return std::acos(product / std::sqrt(in.squared_length()));
}

void rotatePitch(std::vector<Point_3>& cornerPoints, float pitch, Point_3 center)
{
	Vector_3 vertex_after;
	for (auto& vertex : cornerPoints)
	{
		vertex_after = vertex - center;
		Eigen::Vector4f vertex_temp = Eigen::Vector4f(vertex_after.x(), vertex_after.y(), vertex_after.z(), 1);
		vertex_temp = Eigen::AngleAxis(-pitch, Eigen::Vector3f::UnitX()) * vertex_temp;
		vertex = Point_3(vertex_temp.x() + center.x(), vertex_temp.y() + center.y(), vertex_temp.z() + center.z());
	}
}

void rotateYaw(std::vector<Point_3> cornerPoints, float yaw, Point_3 center, bool is_clockwise)
{
	Vector_3 vertex_after;
	for (auto& vertex : cornerPoints)
	{
		vertex_after = vertex - center;
		Eigen::Vector4f vertex_temp = Eigen::Vector4f(vertex_after.x(), vertex_after.y(), vertex_after.z(), 1);
		if (is_clockwise)
			vertex_temp = Eigen::AngleAxis(-yaw, Eigen::Vector3f::UnitX()) * vertex_temp;
		else
			vertex_temp = Eigen::AngleAxis(yaw, Eigen::Vector3f::UnitX()) * vertex_temp;
		vertex = Point_3(vertex_temp.x() + center.x(), vertex_temp.y() + center.y(), vertex_temp.z() + center.z());
	}
}

void writeBbox(const std::string out_path, std::vector<Point_3>& cornerPoints, CGAL::Bbox_2 Bbox2D)
{
	float nx, ny, nz, roll, pitch, yaw;
	Point_3 center = cornerPoints[8];
	pitch = 0;
	yaw = calAngelWithX(cornerPoints[0] - cornerPoints[2]);
	rotatePitch(cornerPoints, pitch, center);
	Vector_3 cross_product = CGAL::cross_product(Vector_3(1, 0, 0), cornerPoints[0] - cornerPoints[2]);
	if (cross_product.z() > 0)
		yaw = -yaw;
	nx = std::sqrt((cornerPoints[0] - cornerPoints[2]).squared_length());
	ny = std::sqrt((cornerPoints[2] - cornerPoints[4]).squared_length());
	nz = std::sqrt((cornerPoints[0] - cornerPoints[1]).squared_length());
	std::fstream outFile(out_path, std::ios::out);
	outFile << "Car 0 0 0 " <<Bbox2D.xmin()<<" " << Bbox2D.ymin() << " " << Bbox2D.xmax() << " " << Bbox2D.ymax() << " " << nx << " " << ny << " " << nz << " " << center.x() << " " << center.y() << " " << center.z() << " " << yaw << "\n";
}

struct ImageCluster
{
	CGAL::Bbox_2 box;
	std::string name;
	cv::Vec3b color;
	std::vector<int> xs;
	std::vector<int> ys;
};

std::vector<ImageCluster> solveCluster(const cv::Mat& vSeg, const std::map<cv::Vec3b, std::string> colorMap) {

	std::map<cv::Vec3b, int> currentColor;
	std::vector<std::pair<std::vector<int>, std::vector<int>>> bbox;

	for (int y = 0; y < vSeg.size().height; y++) {
		for (int x = 0; x < vSeg.size().width; x++) {
			cv::Vec3b pixel = vSeg.at<cv::Vec3b>(y, x);
			if (pixel == cv::Vec3b(55, 181, 57))
				continue;

			if (currentColor.find(pixel) == currentColor.end()) {
				currentColor.insert(std::make_pair(pixel, bbox.size()));
				bbox.push_back(std::make_pair(std::vector<int>(), std::vector<int>()));
			}

			bbox[currentColor.at(pixel)].first.push_back(x);
			bbox[currentColor.at(pixel)].second.push_back(y);
		}
	}
	std::vector<ImageCluster> result;
	for (auto colorIter = currentColor.begin(); colorIter != currentColor.end(); colorIter++) {
		ImageCluster cluster;
		cluster.box = CGAL::Bbox_2(
			*std::min_element(bbox[colorIter->second].first.begin(), bbox[colorIter->second].first.end()),
			*std::min_element(bbox[colorIter->second].second.begin(), bbox[colorIter->second].second.end()),
			*std::max_element(bbox[colorIter->second].first.begin(), bbox[colorIter->second].first.end()),
			*std::max_element(bbox[colorIter->second].second.begin(), bbox[colorIter->second].second.end())
		);
		cluster.color = colorIter->first;
		if (colorMap.find(cluster.color) == colorMap.end())
			continue;
		cluster.name = colorMap.at(cluster.color);
		cluster.xs = bbox[currentColor.at(colorIter->first)].first;
		cluster.ys = bbox[currentColor.at(colorIter->first)].second;
		result.push_back(cluster);
	}
	return result;
}

void BboxFit(std::string in_path, std::string out_path, std::map<string, Point_cloud>& model_point_clouds, std::map<string, std::vector<Point_3>>& model_bbox_corner_vertices)
{
	boost::filesystem::path myPath(in_path);
	boost::filesystem::recursive_directory_iterator endIter;
	for (boost::filesystem::recursive_directory_iterator iter(myPath); iter != endIter; iter++) {
		std::string v_name = iter->path().stem().string();
		std::regex rx("^[0-9]+$");
		bool bl = std::regex_match(v_name.begin(), v_name.end(), rx);
		if (iter->path().filename().extension().string() == ".obj" && bl)
		{
			std::vector<Point_3> cornerPoints;
			std::array<Point_3, 8> obb_points;
			std::vector<float> verticesZ;
			std::vector<cv::Point2f> vertices2D;
			cv::Point2f cornerPoints2D[4];

			std::fstream in_offset((myPath / iter->path().stem()).string() + ".txt", std::ios::in);
			std::string offsets;
			in_offset >> offsets;
			int x_offset = atoi(offsets.substr(0, offsets.find(",")).c_str());
			int y_offset = atoi(offsets.substr(offsets.find(",") + 1).c_str());

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
					cornerPoints.push_back(Point_3(x + x_offset, y + y_offset, z));
					z = maxZ;
				}
			}
			Point_3 center = Point_3(box.center.x, box.center.y, (maxZ + minZ) / 2);
			cornerPoints.push_back(center);
			model_point_clouds.insert(std::make_pair(iter->path().stem().string(), point_cloud));
			model_bbox_corner_vertices.insert(std::make_pair(iter->path().stem().string(), cornerPoints));

			//writeBbox(((out_path + "\\origin_bbox") / iter->path().stem()).string() + ".xyz", cornerPoints, center);
		}
	}
}

void isInModel(Surface_mesh model, std::vector<Point_3> queryPoints, std::vector<bool>& voxel, std::vector<double>& sdf, const std::vector<char>& axis_seq)
{
	//std::vector<char> axis_seq = { 'z','y','x' };
	std::vector<AABB_Point> vertices;
	std::list<Triangle> faces;
	for (face_descriptor face_id : model.faces())
	{
		halfedge_descriptor hi = model.halfedge(face_id);
		for (halfedge_descriptor hf_id : model.halfedges_around_face(hi))
		{
			vertex_descriptor vi = model.target(hf_id);
			vertices.push_back(AABB_Point(model.point(vi).x(), model.point(vi).y(), model.point(vi).z()));
		}
		faces.push_back(Triangle(vertices[0], vertices[1], vertices[2]));
	}

	std::vector<std::list<Triangle>::iterator> deletedIt;
	for (auto it = faces.begin(); it != faces.end(); ++it)
	{
		if ((*it).is_degenerate())
		{
			deletedIt.push_back(it);
		}
	}
	for (auto it : deletedIt)
	{
		faces.erase(it);
	}
	// Build tree
	Tree_tri tree(faces.begin(), faces.end());
	vector<vector<int>> pointData_vectors;
	for (auto axis : axis_seq)
	{
		vector<int> pointData;
		// Judgement
		for (int i = 0; i < queryPoints.size(); i++)
		{
			vector<double> yCoords;
			AABB_Point temp1, temp2;
			if (axis == 'z')
			{
				temp1 = AABB_Point(queryPoints[i][0], queryPoints[i][1], -1000);
				temp2 = AABB_Point(queryPoints[i][0], queryPoints[i][1], 1000);
			}
			else if (axis == 'y') {
				temp1 = AABB_Point(queryPoints[i][0], -1000, queryPoints[i][2]);
				temp2 = AABB_Point(queryPoints[i][0], 1000, queryPoints[i][2]);
			}
			else {
				temp1 = AABB_Point(-1000, queryPoints[i][1], queryPoints[i][2]);
				temp2 = AABB_Point(1000, queryPoints[i][1], queryPoints[i][2]);
			}

			Segment segment_query(temp1, temp2);
			std::list<Segment_intersection> intersections;
			tree.all_intersections(segment_query, std::back_inserter(intersections));
			int loopNum = intersections.size();
			for (int j = 0; j < loopNum; j++)
			{
				const AABB_Point* p = boost::get<AABB_Point>(&(intersections.front()->first));
				if (p)
				{
					if (axis == 'z')
						yCoords.push_back(p->z());
					else if (axis == 'y')
						yCoords.push_back(p->y());
					else if (axis == 'x')
						yCoords.push_back(p->x());
				}
				intersections.pop_front();
			}
			if (yCoords.size() > 0)
			{
				double yMax = *max_element(yCoords.begin(), yCoords.end());
				double yMin = *min_element(yCoords.begin(), yCoords.end());
				if (yMax == yMin) {
					yMin = 0;
				}
				double yPoint;
				if (axis == 'z')
					yPoint = queryPoints[i].z();
				else if (axis == 'y')
					yPoint = queryPoints[i].y();
				else
					yPoint = queryPoints[i].x();
				if (yPoint < yMax && yPoint > yMin)
					pointData.push_back(1);
				else
					pointData.push_back(0);
			}
			else
				pointData.push_back(0);
		}
		pointData_vectors.push_back(pointData);
	}

	for (int i = 0; i < queryPoints.size(); i++)
	{
		double distance = CGAL::to_double(tree.squared_distance(AABB_Point(queryPoints[i][0], queryPoints[i][1], queryPoints[i][2])));
		bool pointData = true;
		for (auto item_pointData_axis : pointData_vectors)
		{
			if (item_pointData_axis.at(i) == 0)
				pointData = false;
		}
		voxel.push_back(pointData);
		sdf.push_back(distance);
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
		load_obj((mesh_root / "total.obj").string(),true, mesh_root.parent_path().string()));
	Point_cloud point_cloud(true);
	for (auto& item_point : mesh.points())
		point_cloud.insert(item_point);
	Height_map height_map(point_cloud, args["heightmap_resolution"].asFloat(), args["heightmap_dilate"].asFloat());
	


	const boost::filesystem::path output_root_path(args["output_root"].asString());

	checkFolder(output_root_path);
	//checkFolder(output_root_path / "3d_box");
	checkFolder(output_root_path / "depth");
	checkFolder(output_root_path / "rgb");
	checkFolder(output_root_path / "segmentation");
	checkFolder(output_root_path / "origin_bbox");

	/*
	 * TODO Iterate the directory and read the individual points
	 */

	std::map<string, Point_cloud> model_point_clouds;
	std::map<string, std::vector<Point_3>> model_bbox_corner_vertices;
	std::map<string, Point_3> model_bbox_center;
	BboxFit(args["mesh_root"].asString(), args["output_root"].asString(), model_point_clouds, model_bbox_corner_vertices, model_bbox_center);


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
	 * TODO Calculate the initial 3d bbox and store
	 */
	
	/*
	 * Sample the environment uniformly and store
	 */
	std::vector<Pos_Pack> place_to_be_travel;
	{
		int delta_x = (map_end_unreal - map_start_unreal).x()/ args["COLLECTION_STEP_X"].asFloat();
		int delta_y = (map_end_unreal - map_start_unreal).y()/ args["COLLECTION_STEP_Y"].asFloat();
		int delta_z = (map_end_unreal - map_start_unreal).z()/ args["COLLECTION_STEP_Z"].asFloat();
		
		for (int id_x=0;id_x<=delta_x;id_x+=1)
			for (int id_y=0;id_y<=delta_y;id_y+=1)
				for (int id_z=0;id_z<=delta_z;id_z+=1)
				{
					Eigen::Vector3f cur_pos_unreal = map_start_unreal + Eigen::Vector3f(
						id_x * args["COLLECTION_STEP_X"].asFloat(),
						id_y * args["COLLECTION_STEP_Y"].asFloat(),
						id_z * args["COLLECTION_STEP_Z"].asFloat()
					);

					Pos_Pack pos_pack = map_converter.get_pos_pack_from_unreal(cur_pos_unreal, 0, 0);
					if(!height_map.in_bound(pos_pack.pos_mesh.x(), pos_pack.pos_mesh.y()))
						continue;
					if(height_map.get_height(pos_pack.pos_mesh.x(), pos_pack.pos_mesh.y()) + 30 > pos_pack.pos_mesh.z())
						continue;

					for (float pitch_degree = 0;pitch_degree <= 60;pitch_degree += args["COLLECTION_STEP_PITCH_DEGREE"].asFloat())
						for (float yaw_degree = 0;yaw_degree <= 360;yaw_degree += args["COLLECTION_STEP_YAW_DEGREE"].asFloat())
							place_to_be_travel.push_back(map_converter.get_pos_pack_from_unreal(
								cur_pos_unreal, 
								yaw_degree / 180 * M_PI,
								pitch_degree/180*M_PI));
				}
	}

	LOG(INFO) << "Total generate " << place_to_be_travel.size() << " place to be traveled";
	auto tqdm_bar = tqdm();
	int cur_num = 0;
	std::map<string, std::vector<Point_3>> camera_bbox_corner_vertices;
	for (const Pos_Pack& item_pos_pack:place_to_be_travel)
	{
		airsim_client->adjust_pose(item_pos_pack);
		auto imgs = airsim_client->get_images();
		cv::Mat seg = imgs.at("segmentation");
		
		/*
		 * Segment the image
		 */
		std::vector<ImageCluster> clusters = solveCluster(seg, color_to_mesh_name_map);
		
		/*
		 * TODO Collect 3d bounding box
		 */
		Eigen::Vector4f bbox_vertex_camera_coords;
		std::vector<Point_3> bbox_vertices_camera_coords;
		for (auto building : clusters)
		{
			for (auto bbox_vertex : model_bbox_corner_vertices[building.name])
			{
				bbox_vertex_camera_coords = item_pos_pack.camera_matrix * Eigen::Vector4f(bbox_vertex[0], bbox_vertex[1], bbox_vertex[2], 1);
				bbox_vertices_camera_coords.push_back(Point_3(bbox_vertex_camera_coords.x(), bbox_vertex_camera_coords.y(), bbox_vertex_camera_coords.z()));
			}
			checkFolder(output_root_path / std::to_string(cur_num));
			writeBbox(args["output_root"].asString() + "\\" + std::to_string(cur_num) + "\\" + building.name + ".xyz", bbox_vertices_camera_coords, building.box);
		}
		

		/*
		 * Collect point cloud
		 */
		// Use item_pos_pack.camera_matrix to transform the points into camera coordinates
		for(const ImageCluster& item_cluster:clusters)
		{
			//(item_cluster.name+".obj")
		}
		
		/*
		* TODO Collect voxel, point cloud in camera space(visible and invisible) and sdf value
		*/

		/*
		* Collect imgs
		*/
		cv::cvtColor(imgs.at("rgb"), imgs.at("rgb"), cv::COLOR_RGB2BGR);
		cv::cvtColor(imgs.at("segmentation"), imgs.at("segmentation"), cv::COLOR_RGB2BGR);
		cv::imwrite((output_root_path / "depth" / (std::to_string(cur_num) + ".tiff")).string(), imgs.at("depth_planar"));
		cv::imwrite((output_root_path / "rgb" / (std::to_string(cur_num) + ".png")).string(), imgs.at("rgb"));
		cv::imwrite((output_root_path / "segmentation" / (std::to_string(cur_num) + ".png")).string(), imgs.at("segmentation"));
		
		tqdm_bar.progress(&item_pos_pack -&place_to_be_travel[0], place_to_be_travel.size());
		cur_num += 1;

	}


	return 0;
}
