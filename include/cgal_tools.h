#ifndef CGAL_TOOLS_H
#define CGAL_TOOLS_H
#include <tiny_obj_loader.h>
#include <tuple>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangle_3<K> Triangle_3;
typedef CGAL::Point_3<K> Point_3;
typedef CGAL::Vector_3<K> Vector_3;
typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;

// @brief: 
// @notice: Currently only transfer vertices to the cgal Surface mesh
// @param: `attrib_t, shape_t, material_t`
// @ret: Surface_mesh
Surface_mesh convert_obj_from_tinyobjloader_to_surface_mesh(
	const std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> v_obj_in);

#endif // CGAL_TOOLS_H
