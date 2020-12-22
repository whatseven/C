#ifndef METRICS_H
#define METRICS_H

#include <iostream>
#include <fstream>
#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>
#include <algorithm>

#include "model_tools.h"
#include "intersection_tools.h"

std::vector<std::array<float, 5>> reconstructability_hueristic(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory,
    const Point_set& point_set,
    const Surface_mesh& v_mesh, std::vector<std::vector<bool>>& point_view_visibility);

#endif // !METRICS_H
