#include "cgal_tools.h"

#include <opencv2/opencv.hpp>

// @brief: 
// @notice: Currently only transfer vertices to the cgal Surface mesh
// @param: `attrib_t, shape_t, material_t`
// @ret: Surface_mesh
Surface_mesh convert_obj_from_tinyobjloader_to_surface_mesh(
	const std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> v_obj_in){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::tie(attrib, shapes, materials)=v_obj_in;
	
	Surface_mesh out;

	for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t face_id = 0; face_id < shapes[s].mesh.num_face_vertices.size(); face_id++) {
            if (shapes[s].mesh.num_face_vertices[face_id] != 3)
                throw;

            Surface_mesh::vertex_index vi[3];
            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                vi[v] = out.add_vertex(Point_3(vx, vy, vz));
            }
            out.add_face(vi[0], vi[1], vi[2]);
            index_offset += 3;
        }
    }

    return out;
}

Eigen::AlignedBox3f get_bounding_box(const Point_set& v_point_set)
{
    float xmin = 1e8, ymin = 1e8, zmin = 1e8;
    float xmax = -1e8, ymax = -1e8, zmax = -1e8;

	for(Point_set::Index idx:v_point_set)
	{
        float vx = v_point_set.point(idx).x();
        float vy = v_point_set.point(idx).y();
        float vz = v_point_set.point(idx).z();
        xmin = xmin < vx ? xmin : vx;
        ymin = ymin < vy ? ymin : vy;
        zmin = zmin < vz ? zmin : vz;

        xmax = xmax > vx ? xmax : vx;
        ymax = ymax > vy ? ymax : vy;
        zmax = zmax > vz ? zmax : vz;
	}


    return Eigen::AlignedBox3f(Eigen::Vector3f(xmin, ymin, zmin), Eigen::Vector3f(xmax, ymax, zmax));
}

Rotated_box get_bounding_box_rotated(const Point_set& v_point_set)
{
    float xmin = 1e8, ymin = 1e8, zmin = 1e8;
    float xmax = -1e8, ymax = -1e8, zmax = -1e8;

    std::vector<cv::Point2f> points;
	
    for (Point_set::Index idx : v_point_set)
    {
        float vx = v_point_set.point(idx).x();
        float vy = v_point_set.point(idx).y();
        float vz = v_point_set.point(idx).z();
        xmin = xmin < vx ? xmin : vx;
        ymin = ymin < vy ? ymin : vy;
        zmin = zmin < vz ? zmin : vz;

        xmax = xmax > vx ? xmax : vx;
        ymax = ymax > vy ? ymax : vy;
        zmax = zmax > vz ? zmax : vz;
        points.emplace_back(vx, vy);
    }

    cv::RotatedRect rotated_rect = cv::minAreaRect(points);
    
    Rotated_box box(Eigen::AlignedBox3f(
        Eigen::Vector3f(rotated_rect.center.x - rotated_rect.size.width / 2, rotated_rect.center.y - rotated_rect.size.height / 2, zmin),
        Eigen::Vector3f(rotated_rect.center.x + rotated_rect.size.width / 2, rotated_rect.center.y + rotated_rect.size.height / 2, zmax)));

    box.angle = rotated_rect.angle/180.f * 3.1415926f;
    box.cv_box = rotated_rect;
    return box;
}
