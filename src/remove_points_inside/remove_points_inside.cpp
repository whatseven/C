#include <argparse/argparse.hpp>

#include "model_tools.h"
#include <CGAL/point_generators_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <boost/filesystem.hpp>

#include "intersection_tools.h"

int main(int argc, char** argv){
	argparse::ArgumentParser program("Sample points and remove points inside");
	program.add_argument("--model_dir").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string model_name = program.get<std::string>("--model_dir");
	boost::filesystem::recursive_directory_iterator end_iter;
	for (boost::filesystem::recursive_directory_iterator iter(model_name); iter != end_iter; iter++) {
		try {
			if (boost::filesystem::is_directory(*iter))
				continue;
			std::string file_path = iter->path().string();
			std::string file_name = iter->path().filename().string();
			Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(load_obj(file_path));
			CGAL::Random_points_in_triangle_mesh_3<Surface_mesh> g(mesh);

			std::vector<Point_3> points;
			std::copy_n(g, 4096 * 2, std::back_inserter(points));

			Point_cloud point_cloud_removed_inside = remove_points_inside(mesh, points);

			while (point_cloud_removed_inside.size() < 4096)
			{
				points.clear();
				std::copy_n(g, 4096 * 2, std::back_inserter(points));
				Point_cloud points_item = remove_points_inside(mesh, points);
				for (auto && item : points_item) 
					point_cloud_removed_inside.insert(points_item.point(item));

			}
			CGAL::write_ply_point_set(std::ofstream(file_name), point_cloud_removed_inside);
		}
		catch (const std::exception& ex) {
			std::cerr << ex.what() << std::endl;
			continue;
		}
	}
	//CGAL::write_off_point_set(std::ofstream("origin.off"), point_cloud);
}
