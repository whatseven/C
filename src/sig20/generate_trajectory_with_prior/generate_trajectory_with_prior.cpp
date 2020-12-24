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

const float BOUNDS_MIN = 20;
const float Z_UP_BOUNDS = 10;
const float Z_DOWN_BOUND = 10;
//const float MERGED_THRESHOLD = 40;
const float MERGED_THRESHOLD = -1;

const bool BASELINE = false;

int main(int argc, char** argv){
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);
	// Read arguments and point clouds
	argparse::ArgumentParser program("Get trajectory with prior map information");
	program.add_argument("--model_path").required();
	program.add_argument("--xy_angle_degree").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string model_path = program.get<std::string>("--model_path");
	float xy_angle= std::atof(program.get<std::string>("--xy_angle_degree").c_str());
	CGAL::Point_set_3<Point_3,Vector_3> point_cloud;
	CGAL::read_ply_point_set(std::ifstream(model_path), point_cloud);
	Height_map height_map(point_cloud, 1);
	height_map.save_height_map_png("1.png", 2);
	height_map.save_height_map_tiff("1.tiff");

	// Delete ground planes
	{
		for (int idx = point_cloud.size() - 1; idx >= 0; idx--) {
			if (point_cloud.point(idx).z() < 4.)
				point_cloud.remove(idx);
		}

		point_cloud.collect_garbage();
	}

	// Cluster building
	std::size_t nb_clusters;
	std::vector<Building> buildings;
	{
		Point_set::Property_map<int> cluster_map = point_cloud.add_property_map<int>("cluster", -1).first;

		std::vector<std::pair<std::size_t, std::size_t> > adjacencies;
		
		nb_clusters = CGAL::cluster_point_set(point_cloud, cluster_map,
				point_cloud.parameters().neighbor_radius(3.).
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
	
	// Generate trajectories
	// No guarantee for the validation of camera position, check it later
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	{
		Trajectory_params params;
		params.view_distance = BOUNDS_MIN;
		params.z_down_bounds = Z_DOWN_BOUND;
		params.z_up_bounds = Z_UP_BOUNDS;
		params.xy_angle = xy_angle;

		for(auto& item_building:buildings)
		{
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> item_trajectory;
			if (BASELINE) {
				generate_trajectory(params, item_building.bounding_box_3d, item_trajectory, height_map);
				trajectory.insert(trajectory.end(), item_trajectory.begin(), item_trajectory.end());
			}
			else
			{
				std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos = std::make_pair(
					Eigen::Vector3f(0.f, 0.f, 0.f),
					Eigen::Vector3f(0.f, 0.f, 0.f)
				);
				do {
					next_pos = generate_next_view(params, item_building,
						next_pos.first,
						false);
					if (next_pos.first != Eigen::Vector3f(0.f, 0.f, 0.f))
						item_trajectory.push_back(next_pos);
					else
						break;
				} while (next_pos.first != Eigen::Vector3f(0.f, 0.f, 0.f) && item_trajectory.size()<200);

				float z_step = (item_building.bounding_box_3d.max().z() + params.z_up_bounds - params.z_down_bounds) / item_trajectory.size();
				for (auto& item : item_trajectory) {
					Eigen::Vector3f& position = item.first;
					position.z() = item_building.bounding_box_3d.max().z() + params.z_up_bounds - (&item - &*item_trajectory.begin()) * z_step;
					//item.second.z() = item_building.bounding_box_3d.max().z() - (&item - &*trajectory.begin()) * z_step;
				}
				
				if(X2)
				{
					size_t start_trajectory_size = trajectory.size();
					std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos = std::make_pair(
						Eigen::Vector3f(0.f, 0.f, 0.f),
						Eigen::Vector3f(0.f, 0.f, 0.f)
					);
					do {
						next_pos = generate_next_view(params, item_building,
							next_pos.first,
							true);
						if (next_pos.first != Eigen::Vector3f(0.f, 0.f, 0.f)&&trajectory.size()<200)
							trajectory.push_back(next_pos);
						else
							break;
					} while (next_pos.first != Eigen::Vector3f(0.f, 0.f, 0.f));
					z_step = (item_building.bounding_box_3d.max().z() + params.z_up_bounds - params.z_down_bounds) / start_trajectory_size;
					for (auto& item : trajectory) {
						if (&item - &trajectory[0] < start_trajectory_size)
							continue;
						Eigen::Vector3f& position = item.first;
						position.z() = params.z_down_bounds + (&item - &*trajectory.begin()-start_trajectory_size) * z_step;
						//item.second.z() = (&item - &*trajectory.begin()-start_trajectory_size) * z_step;
					}
				}
				trajectory.insert(trajectory.end(), item_trajectory.begin(), item_trajectory.end());
			}
			//break;
		}
		for (auto& item : trajectory)
		{
			while (height_map.get_height(item.first.x(), item.first.y()) + Z_UP_BOUNDS > item.first.z()) {
				item.first[2] += 5;
			}
			item.second = (item.second - item.first);
			if (item.second.z() > 0)
				item.second.z() = 0;
			item.second = item.second.normalized();
		}
	}
	LOG(INFO) << "New trajectory ( "<< trajectory .size()<<" view) GENERATED!";

	Visualizer viz;
	// Visualize
	{
		viz.lock();
		//viz.m_buildings = buildings;
		viz.m_pos = trajectory[0].first;
		//viz.m_direction = next_direction;
		viz.m_points= point_cloud;
		viz.m_trajectories = trajectory;
		viz.unlock();
		//override_sleep(100);
		//debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
	}
	
	for (int i = 0; i < trajectory.size(); ++i)
	{
		point_cloud.insert(Point_3(trajectory[i].first[0], trajectory[i].first[1], trajectory[i].first[2]));
	}
	CGAL::write_ply_point_set(std::ofstream("test_point.ply"), point_cloud);
	write_unreal_path(trajectory, "camera_after_transaction.log");
	write_smith_path(trajectory, "camera_smith.log");
	write_normal_path(trajectory, "camera_normal.log");
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