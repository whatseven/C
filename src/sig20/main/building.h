#pragma once
#include "model_tools.h"
#include <CGAL/point_generators_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/random_selection.h>

struct Building {
	//Eigen::AlignedBox3f bounding_box_3d;
	Rotated_box bounding_box_3d;
	Point_set points_camera_space;
	Point_set points_world_space;
	CGAL::Bbox_2 bounding_box_2d;
	cv::Vec3b segmentation_color;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> passed_trajectory;
	std::vector<Rotated_box> boxes;
	bool is_changed = true;
	bool is_divide = false;
	int parent = -1;
	int one_pass_trajectory_num = 0;
	int closest_trajectory_id = 0;

	//Used for trajectory generation
	int start_box = -1;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;

	int find_nearest_trajectory(const Eigen::Vector3f& v_pos) const 
	{
		return std::min_element(trajectory.begin(), trajectory.end(),
			[&v_pos](const std::pair<Eigen::Vector3f, Eigen::Vector3f>& item1, const std::pair<Eigen::Vector3f, Eigen::Vector3f>& item2) {
			return (item1.first - v_pos).norm() < (item2.first - v_pos).norm();
		}) - trajectory.begin();
	}

	int find_nearest_trajectory_2d(const Eigen::Vector3f& v_pos) const {
		return std::min_element(trajectory.begin(), trajectory.end(),
			[&v_pos](const std::pair<Eigen::Vector3f, Eigen::Vector3f>& item1, const std::pair<Eigen::Vector3f, Eigen::Vector3f>& item2) {
			Eigen::Vector2f drone_pos(v_pos.x(), v_pos.y());
			Eigen::Vector2f trajectory_pos1(item1.first.x(), item1.first.y());
			Eigen::Vector2f trajectory_pos2(item1.first.x(), item1.first.y());
				return (trajectory_pos1 - drone_pos).norm() < (trajectory_pos2 - drone_pos).norm();
		}) - trajectory.begin();
	}
};

struct Next_target {
	int origin_index_in_building_vector = -1;
	int origin_index_in_untraveled_pointset = -1;
	Next_target(const int v_origin_index_in_building_vector, const int v_origin_index_in_untraveled_pointset)
		:origin_index_in_building_vector(v_origin_index_in_building_vector), origin_index_in_untraveled_pointset(v_origin_index_in_untraveled_pointset) {
	}
};
