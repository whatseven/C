#pragma once
#include "model_tools.h"
#include <CGAL/point_generators_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/random_selection.h>

struct Building {
	Eigen::AlignedBox3f bounding_box_3d;
	Point_set points_camera_space;
	Point_set points_world_space;
	CGAL::Bbox_2 bounding_box_2d;
	cv::Vec3b segmentation_color;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> passed_trajectory;
	std::vector<Eigen::AlignedBox3f> boxes;

	//Used for trajectory generation
	int start_box = -1;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
};

class Building_Set {
public:
	std::vector<Building> buildings;
	bool hasShot;
	std::vector<int> id_sequence;
	Building_Set::Building_Set(int number){
		buildings = std::vector<Building>(number);
		hasShot = false;
	}
};

struct Connect_information {
	float DISTANCE_THRESHOLD = 20.f;
	
	std::vector<CGAL::Point_2<K>> sample_points;
	std::vector<bool> is_point_traveled;
	
	Connect_information(const Eigen::Vector3f& map_start, const Eigen::Vector3f& map_end) {
		CGAL::Iso_rectangle_2<K> world_rectangle(CGAL::Point_2<K>(map_start.x(), map_start.y()), CGAL::Point_2<K>(map_end.x(), map_end.y()));
		for (float y = map_start.y(); y < map_end.y(); y += DISTANCE_THRESHOLD)
			for (float x = map_start.x(); x < map_end.x(); x += DISTANCE_THRESHOLD)
				sample_points.push_back(CGAL::Point_2<K>(x, y));
		is_point_traveled.resize(sample_points.size(), false);
	}

	void update_sample_points(const Eigen::Vector3f& v_cur_pos) {
		for (int i_point = 0; i_point < is_point_traveled.size(); i_point++){
			if (is_point_traveled[i_point])
				continue;
			const CGAL::Point_2<K>& p = sample_points[i_point];
			float squared_distance = (p - CGAL::Point_2<K>(v_cur_pos.x(), v_cur_pos.y())).squared_length();
			
			if (squared_distance < DISTANCE_THRESHOLD * DISTANCE_THRESHOLD)
				is_point_traveled[i_point] = true;
		}
	}
};

struct Next_target {
	int origin_index_in_building_vector = -1;
	int origin_index_in_untraveled_pointset = -1;
	Next_target(const int v_origin_index_in_building_vector, const int v_origin_index_in_untraveled_pointset)
		:origin_index_in_building_vector(v_origin_index_in_building_vector), origin_index_in_untraveled_pointset(v_origin_index_in_untraveled_pointset) {
	}
};