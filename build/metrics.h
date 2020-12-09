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

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef K::Segment_3 Segment;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef boost::optional<Tree::Intersection_and_primitive_id<Ray>::Type> Ray_intersection;
typedef boost::optional<Tree::Intersection_and_primitive_id<Segment>::Type > Segment_intersection;

void read_point_set(std::string path, std::vector<std::pair<Point, Point>> point_set);
float reconstructability_hueristic(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>);


#endif // !METRICS_H
