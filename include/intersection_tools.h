#ifndef INTERSECTION_TOOLS_H
#define INTERSECTION_TOOLS_H
#include "cgal_tools.h"

#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>

typedef CGAL::Ray_3<K> Ray;


typedef CGAL::AABB_face_graph_triangle_primitive<Surface_mesh, CGAL::Default, CGAL::Tag_false> Primitive;
typedef CGAL::AABB_traits<K, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef Tree::Primitive_id Primitive_id;

// @brief: Get the depth map by building a bvh tree and query the distance and instance label of each mesh
// @notice: Distance is perspective distance, not planar distance!
// @param:
// @ret: `{ img_distance,img_distance_planar, img_distance_perspective} (0 represent the background, object id start by 1)`
std::tuple<cv::Mat, cv::Mat, cv::Mat> get_depth_map_through_meshes(const std::vector<Surface_mesh>& v_meshes,
    const int v_width, const int v_height,
    const Eigen::Matrix3f& v_intrinsic);

const Point_cloud remove_points_inside(const Surface_mesh& v_mesh, const std::vector<Point_3>& v_points);

#endif // INTERSECTION_TOOLS_H
