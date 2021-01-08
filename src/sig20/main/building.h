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

struct Next_target {
	int origin_index_in_building_vector = -1;
	int origin_index_in_untraveled_pointset = -1;
	Next_target(const int v_origin_index_in_building_vector, const int v_origin_index_in_untraveled_pointset)
		:origin_index_in_building_vector(v_origin_index_in_building_vector), origin_index_in_untraveled_pointset(v_origin_index_in_untraveled_pointset) {
	}
};

enum Region_status { Unobserved, Free, Occupied };