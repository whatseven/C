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

typedef CGAL::Simple_cartesian<double> SC_K;
typedef SC_K::Ray_3 SC_Ray;
typedef SC_K::Line_3 SC_Line;
typedef SC_K::Point_3 SC_Point;
typedef SC_K::FT SC_FT;
typedef SC_K::Triangle_3 SC_Triangle;
typedef SC_K::Segment_3 SC_Segment;
typedef std::list<SC_Triangle>::iterator SC_Iterator;
typedef CGAL::AABB_triangle_primitive<SC_K, SC_Iterator> SC_Primitive;
typedef CGAL::AABB_traits<SC_K, SC_Primitive> SC_AABB_triangle_traits;
typedef CGAL::AABB_tree<SC_AABB_triangle_traits> SC_Tree;
typedef boost::optional<SC_Tree::Intersection_and_primitive_id<SC_Ray>::Type> SC_Ray_intersection;
typedef boost::optional<SC_Tree::Intersection_and_primitive_id<SC_Segment>::Type > SC_Segment_intersection;

const float PI = 3.14159265358;

void read_point_set(std::string path, std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& point_set, CGAL::Bbox_3 bounding_box_3d);
float reconstructability_hueristic(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory, std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> point_set, std::list<SC_Triangle>& triangles);
void readObj(std::string path, std::list<SC_Triangle>& faces);


#endif // !METRICS_H
