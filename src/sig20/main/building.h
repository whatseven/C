#pragma once
#include "model_tools.h"

struct Building {
	float BOUNDS = 20;
	Eigen::AlignedBox3f bounding_box_3d;
	Point_set points_camera_space;
	Point_set points_world_space;
	CGAL::Bbox_2 bounding_box_2d;
	cv::Vec3b segmentation_color;
	std::pair<Eigen::Vector3f, Eigen::Vector3f> exit_path;
	std::pair<Eigen::Vector3f, Eigen::Vector3f> entrance_path;
	Eigen::Vector3f center;
	Eigen::Vector3f top_left;
	Eigen::Vector3f top_right;
	Eigen::Vector3f bottom_left;
	Eigen::Vector3f bottom_right;
	Eigen::Vector3f top_left_BOUNDS;
	Eigen::Vector3f top_right_BOUNDS;
	Eigen::Vector3f bottom_left_BOUNDS;
	Eigen::Vector3f bottom_right_BOUNDS;
	Eigen::Vector3f left;
	Eigen::Vector3f top;
	Eigen::Vector3f bottom;
	Eigen::Vector3f right;
	void update() {
		center = bounding_box_3d.center();
		top_left = bounding_box_3d.corner(bounding_box_3d.TopLeft);
		top_right = bounding_box_3d.corner(bounding_box_3d.TopRight);
		bottom_left = bounding_box_3d.corner(bounding_box_3d.BottomLeft);
		bottom_right = bounding_box_3d.corner(bounding_box_3d.BottomRight);
		top_left_BOUNDS = top_left + (top_left - bottom_left).normalized() * BOUNDS + (top_left - top_right).normalized() * BOUNDS;
		top_right_BOUNDS = top_right + (top_right - bottom_right).normalized() * BOUNDS + (top_right - top_left).normalized() * BOUNDS;
		bottom_left_BOUNDS = bottom_left + (bottom_left - top_left).normalized() * BOUNDS + (bottom_left - bottom_right).normalized() * BOUNDS;
		bottom_right_BOUNDS = bottom_right + (bottom_right - bottom_left).normalized() * BOUNDS + (bottom_right - top_right).normalized() * BOUNDS;
		left = center + ((top_left_BOUNDS - center) + (bottom_left_BOUNDS - top_left_BOUNDS) / 2) * 2;
		top = center + ((top_left_BOUNDS - center) + (top_right_BOUNDS - top_left_BOUNDS) / 2) * 2;
		bottom = center + ((bottom_right_BOUNDS - center) + (bottom_left_BOUNDS - bottom_right_BOUNDS) / 2) * 2;
		right = center + ((bottom_right_BOUNDS - center) + (top_right_BOUNDS - bottom_right_BOUNDS) / 2) * 2;
	}
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