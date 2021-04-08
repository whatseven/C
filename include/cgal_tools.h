#ifndef CGAL_TOOLS_H
#define CGAL_TOOLS_H
#include <tiny_obj_loader.h>
#include <tuple>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Sparse>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef K::Point_3 Point_3;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Triangle_3<K> Triangle_3;
//typedef CGAL::Point_2<K> Point_2;
//typedef CGAL::Point_3<K> Point_3;
typedef CGAL::Direction_3<K> Direction_3;
typedef CGAL::Vector_3<K> Vector_3;
typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Point_set_3<K::Point_3> Point_cloud;
typedef CGAL::Point_set_3<Point_3> Point_set;
typedef Surface_mesh::Face_index face_descriptor;
typedef Surface_mesh::Vertex_index vertex_descriptor;
typedef Surface_mesh::Halfedge_index halfedge_descriptor;

typedef CGAL::Simple_cartesian<double> K2;
typedef K2::FT FT;
typedef K2::Triangle_3 Triangle;
typedef K2::Segment_3 Segment;
typedef K2::Point_3 AABB_Point;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K2, Iterator> Primitive_tri;
typedef CGAL::AABB_traits<K2, Primitive_tri> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree_tri;
typedef boost::optional<Tree_tri::Intersection_and_primitive_id<Segment>::Type > Segment_intersection;


// @brief: 
// @notice: Currently only transfer vertices to the cgal Surface mesh
// @param: `attrib_t, shape_t, material_t`
// @ret: Surface_mesh
Surface_mesh convert_obj_from_tinyobjloader_to_surface_mesh(
	const std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> v_obj_in);

Eigen::AlignedBox3f get_bounding_box(const Point_set& v_point_set);

#endif // CGAL_TOOLS_H
