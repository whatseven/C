#include "intersection_tools.h"


std::tuple<cv::Mat, cv::Mat, cv::Mat> get_depth_map_through_meshes(const std::vector<Surface_mesh>& v_meshes,
	const int v_width,const int v_height,
	const Eigen::Matrix3f& v_intrinsic)
{
	std::vector<Surface_mesh> meshes(v_meshes);
	
	Tree tree;
	for (int i = 0; i < v_meshes.size(); ++i)
	{
		meshes[i].add_property_map<Surface_mesh::face_index, int>("instance_id", i);
		tree.insert(CGAL::faces(meshes[i]).first, CGAL::faces(meshes[i]).second, meshes[i]);
	}
	tree.build();

	cv::Mat img_instance(v_height, v_width, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat img_distance_perspective(v_height, v_width, CV_32FC1, cv::Scalar(0.f));
	cv::Mat img_distance_planar(v_height, v_width, CV_32FC1, cv::Scalar(0.f));
	
#pragma omp parallel for
	for (int y = 0; y < v_height; y++) {
		for (int x = 0; x < v_width; x++) {
			Eigen::Vector3f p(x, y, 1);
			p = v_intrinsic.inverse() * p;
			Ray ray(Point_3(0, 0, 0), Vector_3(p.x(), p.y(), p.z()));
			auto result = tree.first_intersection(ray);
			auto result1 = tree.first_intersected_primitive(ray);
			if (result1) {
				const Surface_mesh::face_index face_index = result1->first;
				const Surface_mesh* mesh = result1->second;
				if (mesh) {
					const Primitive_id& primitive_id = boost::get<Primitive_id>(result->second);
					for (int id_mesh = 0; id_mesh < meshes.size(); id_mesh++) {
						Surface_mesh::Property_map<Surface_mesh::face_index, int> gnus_iter;
						bool found;
						boost::tie(gnus_iter, found) = meshes[id_mesh].property_map<Surface_mesh::face_index, int>("instance_id");

						Surface_mesh::Property_map<Surface_mesh::face_index, int> gnus;
						boost::tie(gnus, found) = mesh->property_map<Surface_mesh::face_index, int>("instance_id");

						if (gnus_iter[*(meshes[id_mesh].faces_begin())] == gnus[face_index])
							img_instance.at<cv::Vec3b>(y, x)[2] = id_mesh;
					}
					img_distance_perspective.at<float>(y, x) = std::sqrt(CGAL::squared_distance(boost::get<Point_3>(result->first), Point_3(0, 0, 0)));

					img_distance_planar.at<float>(y, x) = boost::get<Point_3>(result->first)[2];
				}
			}
		}
	}

	//cv::imshow("1", img_instance);
	//cv::waitKey();

	return { img_instance,img_distance_planar, img_distance_perspective};
}
