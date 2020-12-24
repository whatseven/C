#pragma once
#include "model_tools.h"

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
