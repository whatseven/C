#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>

#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"


struct Building
{
	CGAL::Bbox_3 bounding_box;
	Point_set points;
};

const float BOUNDS = 5;
const float Z_UP_BOUNDS = 5;
const float Z_DOWN_BOUND = 5;
const float STEP = 3;
const float CENTER_Z1 = 0.5;
const float CENTER_Z2 = 0.25;
const float MM_PI = 3.14159265358;
const bool DOUBLE_FLAG = true;

void write_unreal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories,const std::string& v_path)
{
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i)
	{
		const Eigen::Vector3f& position = v_trajectories[i].first*100;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) * 180. / MM_PI;
		float yaw = -std::atan2f(direction[1], direction[0]) * 180./ MM_PI + 90.f;
		
		pose << (fmt %i% -position[0] % position[1] % position[2] % -pitch% yaw).str();
	}
	
	pose.close();
}

void write_normal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories, const std::string& v_path) {
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i) {
		const Eigen::Vector3f& position = v_trajectories[i].first;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) * 180. / MM_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / MM_PI;

		pose << (fmt % i % position[0] % position[1] % position[2] % pitch % yaw).str();
	}

	pose.close();
}

int main(int argc, char** argv){
	// Read arguments and point clouds
	argparse::ArgumentParser program("Get depth map through ray tracing");
	program.add_argument("--model_path").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string model_path = program.get<std::string>("--model_path");
	CGAL::Point_set_3<Point_3,Vector_3> point_cloud;
	CGAL::read_ply_point_set(std::ifstream(model_path), point_cloud);
	Height_map height_map(point_cloud, 3);
	height_map.save_height_map_png("1.png", 2);
	height_map.save_height_map_tiff("1.tiff");

	// Delete ground planes
	{
		for (int idx = point_cloud.size() - 1; idx >= 0; idx--) {
			if (point_cloud.point(idx).z() < 2.)
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
			current_building.points.insert(point_cloud.point(idx));
		}
	}
	
	// Generate trajectories
	// No guarantee for the validation of camera position, check it later
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	{
		for (int id_building = 0; id_building < nb_clusters; ++id_building) {
			buildings[id_building].bounding_box = get_bounding_box(buildings[id_building].points);
			float xmin = buildings[id_building].bounding_box.xmin();
			float ymin = buildings[id_building].bounding_box.ymin();
			float zmin = buildings[id_building].bounding_box.zmin();
			float xmax = buildings[id_building].bounding_box.xmax();
			float ymax = buildings[id_building].bounding_box.ymax();
			float zmax = buildings[id_building].bounding_box.zmax();
			Eigen::Vector3f box_third_points_2(
				(xmin + xmax)/2,
				(ymin + ymax)/2,
				(zmin + zmax)* CENTER_Z1
			);
			Eigen::Vector3f box_third_points(
				(xmin + xmax)/2,
				(ymin + ymax)/2,
				(zmin + zmax)* CENTER_Z2
			);
			

			Eigen::Vector3f cur_pos(xmin - BOUNDS, ymin - BOUNDS, zmax + Z_UP_BOUNDS);
			while (cur_pos.x() <= xmax + BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[0] += STEP;
			}
			while (cur_pos.y() <= ymax + BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[1] += STEP;
			}
			while (cur_pos.x() >= xmin - BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[0] -= STEP;
			}
			while (cur_pos.y() >= ymin - BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[1] -= STEP;
			}
		}
	}
	
	// Check the camera position
	{
		for (int i = 0; i < trajectory.size(); ++i) {
			Eigen::Vector3f position = trajectory[i].first;
			Eigen::Vector3f camera_focus= trajectory[i].first+ trajectory[i].second;
			
			while(height_map.get_height(position.x(), position.y()) + Z_DOWN_BOUND >position.z())
			{
				position[2] += 5;
			}
			Eigen::Vector3f camera_direction = camera_focus - position;
			trajectory[i].second = camera_direction.normalized();
			trajectory[i].first = position;
		}
	}

	
	for (int i = 0; i < trajectory.size(); ++i)
	{
		point_cloud.insert(Point_3(trajectory[i].first[0], trajectory[i].first[1], trajectory[i].first[2]));
	}
	CGAL::write_ply_point_set(std::ofstream("test_point.ply"), point_cloud);
	write_unreal_path(trajectory, "camera_after_transaction.log");
	write_normal_path(trajectory, "camera_normal.log");
	
	return 0;
}
