#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>

#include <boost/format.hpp>
#include <glog/logging.h>

#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/viz.h"
#include "../main/building.h"
#include "../main/trajectory.h"
#include "common_util.h"
#include <json/json.h>

int main(int argc, char** argv){
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);
	argparse::ArgumentParser program("Get trajectory with prior map information");
	program.add_argument("--config_file").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}
	const std::string config_file = program.get<std::string>("--config_file");

	std::ifstream in(config_file);
	if (!in.is_open()) {
		LOG(ERROR) << "Error opening file" << config_file << std::endl;
		return 0;
	}
	Json::Reader json_reader;
	Json::Value json_root;
	if (!json_reader.parse(in, json_root))
	{
		LOG(ERROR) << "Error parse config file" << config_file << std::endl;
		return 0;
	}
	in.close();
	
	const float HEIGHT_CLIP = json_root["HEIGHT_CLIP"].asFloat();
	const float BOUNDS_MIN = json_root["BOUNDS_MIN"].asFloat();
	const float Z_UP_BOUNDS = json_root["Z_UP_BOUNDS"].asFloat();
	const float Z_DOWN_BOUND = json_root["Z_DOWN_BOUND"].asFloat();
	const float MERGED_THRESHOLD = json_root["MERGED_THRESHOLD"].asFloat();
	const float SPLIT_THRESHOLD = json_root["SPLIT_THRESHOLD"].asFloat();
	const float HEIGHT_COMPENSATE = json_root["HEIGHT_COMPENSATE"].asFloat();
	const float xy_angle= json_root["xy_angle"].asFloat();
	const float heightmap_resolution = json_root["heightmap_resolution"].asFloat();
	const float step = json_root["step"].asFloat();
	const bool double_flag = json_root["double_flag"].asBool();
	const float cluster_radius = json_root["cluster_radius"].asFloat();
	const std::string model_path = json_root["model_path"].asString();
	CGAL::Point_set_3<Point_3,Vector_3> original_point_cloud;
	CGAL::read_ply_point_set(std::ifstream(model_path), original_point_cloud);
	CGAL::Point_set_3<Point_3, Vector_3> point_cloud(original_point_cloud);
	Height_map height_map(point_cloud, heightmap_resolution);
	height_map.save_height_map_png("1.png", HEIGHT_CLIP);
	height_map.save_height_map_tiff("1.tiff");

	// Delete ground planes
	{
		for (int idx = point_cloud.size() - 1; idx >= 0; idx--) {
			if (point_cloud.point(idx).z() < HEIGHT_CLIP)
				point_cloud.remove(idx);
		}

		point_cloud.collect_garbage();
	}
	CGAL::write_ply_point_set(std::ofstream("points_without_plane.ply"), point_cloud);
	
	// Cluster building
	std::size_t nb_clusters;
	std::vector<Building> buildings;
	{
		Point_set::Property_map<int> cluster_map = point_cloud.add_property_map<int>("cluster", -1).first;

		std::vector<std::pair<std::size_t, std::size_t> > adjacencies;
		
		nb_clusters = CGAL::cluster_point_set(point_cloud, cluster_map,
				point_cloud.parameters().neighbor_radius(cluster_radius).
				adjacencies(std::back_inserter(adjacencies)));
		buildings.resize(nb_clusters);

		Point_set::Property_map<unsigned char> red = point_cloud.add_property_map<unsigned char>("red", 0).first;
		Point_set::Property_map<unsigned char> green = point_cloud.add_property_map<unsigned char>("green", 0).first;
		Point_set::Property_map<unsigned char> blue = point_cloud.add_property_map<unsigned char>("blue", 0).first;
		for (Point_set::Index idx : point_cloud) {
			// One color per cluster
			int cluster_id = cluster_map[idx];
			CGAL::Random rand(cluster_id);
			red[idx] = rand.get_int(64, 192);
			green[idx] = rand.get_int(64, 192);
			blue[idx] = rand.get_int(64, 192);

			Building& current_building = buildings[cluster_id];
			current_building.points_world_space.insert(point_cloud.point(idx));
		}
	}
	for (int i_building_1 = 0; i_building_1 < buildings.size(); ++i_building_1)
	{
		buildings[i_building_1].bounding_box_3d = get_bounding_box(buildings[i_building_1].points_world_space);
		buildings[i_building_1].boxes.push_back(buildings[i_building_1].bounding_box_3d);
	}
	
	// Merge and splitting building
	if(false){
		std::vector<bool> already_merged_flag(buildings.size(),false);
		for(int i_building_1=0; i_building_1 <buildings.size();++i_building_1)
		{
			for (int i_building_2 = i_building_1+1; i_building_2 < buildings.size(); ++i_building_2) {
				float distance = buildings[i_building_1].bounding_box_3d.exteriorDistance(buildings[i_building_2].bounding_box_3d);
				if(!already_merged_flag[i_building_1]&& !already_merged_flag[i_building_2] && distance < MERGED_THRESHOLD)
				{
					buildings[i_building_1].bounding_box_3d=buildings[i_building_1].bounding_box_3d.merged(buildings[i_building_2].bounding_box_3d);
					buildings[i_building_1].boxes.push_back(buildings[i_building_2].bounding_box_3d);
					for(auto item: buildings[i_building_2].points_world_space.points())
						buildings[i_building_1].points_world_space.insert(item);
					already_merged_flag[i_building_2] = true;
				}
			}
		}
		buildings.erase(std::remove_if(buildings.begin(), buildings.end(),
			[&already_merged_flag, idx = 0](auto item)mutable {return already_merged_flag[idx++]; }), buildings.end());
	}
	if (true)
	{
		std::vector<Building> new_buildings;

		for (int i_building = 0; i_building < buildings.size(); ++i_building)
		{
			auto& bbox = buildings[i_building].bounding_box_3d;
			Eigen::Vector3f min = bbox.min();
			Eigen::Vector3f max = bbox.max();

			for(int i_x = 0;i_x < bbox.sizes().x() / SPLIT_THRESHOLD;++i_x)
			{
				for (int i_y = 0; i_y < bbox.sizes().y() / SPLIT_THRESHOLD; ++i_y) {
					Eigen::Vector3f new_min(min);
					Eigen::Vector3f new_max(max);
					new_min.x() = std::min(max.x(), min.x() + SPLIT_THRESHOLD * (i_x));
					new_min.y() = std::min(max.y(), min.y() + SPLIT_THRESHOLD * (i_y));
					new_max.x() = std::min(max.x(), min.x() + SPLIT_THRESHOLD * (i_x + 1));
					new_max.y() = std::min(max.y(), min.y() + SPLIT_THRESHOLD * (i_y + 1));
					
					Building building;
					building.bounding_box_3d = Eigen::AlignedBox3f(new_min, new_max);
					building.boxes.push_back(Eigen::AlignedBox3f(new_min, new_max));
					new_buildings.push_back(building);
				}
			}
		}
		buildings = new_buildings;
	}
	// Generate trajectories
	// No guarantee for the validation of camera position, check it later
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	{
		Trajectory_params params;
		params.view_distance = BOUNDS_MIN;
		params.z_down_bounds = Z_DOWN_BOUND;
		params.z_up_bounds = Z_UP_BOUNDS;
		params.xy_angle = xy_angle;
		
		params.double_flag = double_flag;
		params.step = step;
		
		if(false)
		{
			for (auto& item_building : buildings) {
				std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> item_trajectory;
				bool continue_flag = false;
				std::pair<Eigen::Vector3f, Eigen::Vector3f> cur_pos = std::make_pair(
					Eigen::Vector3f(0.f, 0.f, 0.f),
					Eigen::Vector3f(0.f, 0.f, 0.f)
				);
				do {
					continue_flag = generate_next_view_curvature(params, item_building,
						cur_pos);
					//std::cout << continue_flag << std::endl;
					if (continue_flag)
						item_trajectory.push_back(cur_pos);
				} while (continue_flag);

				float z_step = (item_building.bounding_box_3d.max().z() + params.z_up_bounds - params.z_down_bounds) / item_trajectory.size();
				for (auto& item : item_trajectory) {
					Eigen::Vector3f& position = item.first;
					position.z() = item_building.bounding_box_3d.max().z() + params.z_up_bounds - (&item - &*item_trajectory.begin()) * z_step;
					//item.second.z() = item_building.bounding_box_3d.max().z() - (&item - &*trajectory.begin()) * z_step;
				}

				trajectory.insert(trajectory.end(), item_trajectory.begin(), item_trajectory.end());
				//break;
			}

		}
		else
		{
			for (int id_building = 0; id_building < buildings.size(); ++id_building) {
				float xmin = buildings[id_building].bounding_box_3d.min().x();
				float ymin = buildings[id_building].bounding_box_3d.min().y();
				float zmin = buildings[id_building].bounding_box_3d.min().z();
				float xmax = buildings[id_building].bounding_box_3d.max().x();
				float ymax = buildings[id_building].bounding_box_3d.max().y();
				float zmax = buildings[id_building].bounding_box_3d.max().z();
				for(int i_pass=0;i_pass<2;++i_pass)
				{
					Eigen::Vector3f cur_pos(xmin - params.view_distance, ymin - params.view_distance, zmax + Z_UP_BOUNDS);
					Eigen::Vector3f focus_point;
					if (i_pass == 0)
					{
						focus_point= Eigen::Vector3f(
							(xmin + xmax) / 2,
							(ymin + ymax) / 2,
							(zmin + zmax) / 2
						);
					}
					else if (i_pass == 1)
					{
						cur_pos.z() /= 2;
						focus_point=Eigen::Vector3f(
							(xmin + xmax) / 2,
							(ymin + ymax) / 2,
							(zmin + zmax) / 5
						);
					}
					while (cur_pos.x() <= xmax + params.view_distance) {
						trajectory.push_back(std::make_pair(
							cur_pos, focus_point
						));
						cur_pos[0] += params.step;
					}
					while (cur_pos.y() <= ymax + params.view_distance) {
						trajectory.push_back(std::make_pair(
							cur_pos, focus_point
						));
						cur_pos[1] += params.step;
					}
					while (cur_pos.x() >= xmin - params.view_distance) {
						trajectory.push_back(std::make_pair(
							cur_pos, focus_point
						));
						cur_pos[0] -= params.step;
					}
					while (cur_pos.y() >= ymin - params.view_distance) {
						trajectory.push_back(std::make_pair(
							cur_pos, focus_point
						));
						cur_pos[1] -= params.step;
					}
					if (!double_flag)
						break;
				}
			}
		}
	}
	LOG(INFO) << "New trajectory ( "<< trajectory .size()<<" view) GENERATED!";
	LOG(INFO) << "Length: "<< evaluate_length(trajectory);
	LOG(INFO) << "Total building: "<< buildings.size();

	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> safe_trajectory = ensure_safe_trajectory(trajectory, height_map, HEIGHT_COMPENSATE, Z_UP_BOUNDS);
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> interpolated_trajectory = interpolate_path(trajectory);
	interpolated_trajectory = ensure_safe_trajectory(interpolated_trajectory, height_map, HEIGHT_COMPENSATE, Z_UP_BOUNDS);
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> simplified_trajectory = simplify_path_reduce_waypoints(interpolated_trajectory);

	write_unreal_path(safe_trajectory, "camera_after_transaction.log");
	write_smith_path(safe_trajectory, "camera_smith.log");
	write_normal_path(safe_trajectory, "camera_normal.log");
	write_wgs_path(simplified_trajectory, "camera_wgs.log");

	Visualizer viz;
	// Visualize
	viz.lock();
	//viz.m_buildings = buildings;
	viz.m_pos = trajectory[0].first;
	//viz.m_direction = next_direction;
	viz.m_points= point_cloud;
	//viz.m_points= original_point_cloud;
	//viz.m_trajectories = safe_trajectory;
	viz.m_trajectories_spline = simplified_trajectory;
	std::vector<Eigen::AlignedBox3f> boxes;
	for (auto& item : buildings) {
		boxes.push_back(item.bounding_box_3d);
		//break;
	}
	//viz.m_boxes = boxes;
	viz.unlock();
	override_sleep(1000);
	debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
	// Visualize
	for(int i=0;i< buildings.size();++i)
	{
		viz.lock();
		//viz.m_buildings = buildings;
		viz.m_pos = trajectory[0].first;
		//viz.m_direction = next_direction;
		//viz.m_points= point_cloud;
		//viz.m_points= original_point_cloud;
		viz.m_trajectories = trajectory;
		std::vector<Eigen::AlignedBox3f> boxes;
		for (auto& item : buildings)
		{
			boxes.push_back(item.bounding_box_3d);
			std::cout << item.bounding_box_3d.min().x() << "," << item.bounding_box_3d.min().y() << std::endl;
			std::cout << item.bounding_box_3d.max().x() << "," << item.bounding_box_3d.max().y() << std::endl;
			std::cout <<  std::endl;
			//break;
		}
		//viz.m_boxes = boxes;
		viz.m_boxes = std::vector<Eigen::AlignedBox3f>{ buildings [i].bounding_box_3d};
		viz.unlock();
		//override_sleep(100);
		debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
	}
	override_sleep(1000);
	return 0;
}


class MSLAM
{
public:
	MSLAM();
	bool track(const cv::Mat& v_img, const Eigen::Isometry3f& v_pose); // Return true if new points added.
	std::vector<Eigen::Vector3f> get_points(); // Return the total points in the tracking coordinates
};

class Modeler
{
public:
	Modeler()
	{
		m_slam = new MSLAM;
	}
	std::vector<Eigen::AlignedBox3f> get_bounding_box(const cv::Mat& v_img, const Eigen::Isometry3f& v_pose)
	{
		m_slam->track(v_img,v_pose);
	}
private:
	MSLAM* m_slam;
	
};