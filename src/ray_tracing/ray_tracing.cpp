#include <argparse/argparse.hpp>

#include "model_tools.h"
#include "intersection_tools.h"


int main(int argc, char** argv){
	argparse::ArgumentParser program("Get depth map through ray tracing");
	program.add_argument("--model_path").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string model_path = program.get<std::string>("--model_path");
	Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(load_obj(model_path));

	Eigen::Matrix3f intrinsic;
	intrinsic << 400, 0, 400, 0, 400, 400, 0, 0, 1;

	get_depth_map_through_meshes(
		std::vector<Surface_mesh>{mesh},
		800, 800,
		intrinsic
	);
	
	CGAL::write_off(std::ofstream("1.off"), mesh);
	return 0;
}