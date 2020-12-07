#pragma once
#include "model_tools.h"

struct Building {
	CGAL::Bbox_3 bounding_box_3d;
	Point_set points_camera_space;
	Point_set points_world_space;
	CGAL::Bbox_2 bounding_box_2d;
};