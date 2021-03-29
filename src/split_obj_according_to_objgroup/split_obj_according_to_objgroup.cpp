#include <argparse/argparse.hpp>

#include "model_tools.h"
#include <boost/filesystem.hpp>
#include <corecrt_math_defines.h>

int main(int argc, char** argv){
		
	argparse::ArgumentParser program("Split obj by group, names of the mesh will be the same as the names of the groups");
	program.add_argument("--model").required().help("The complete path of the model, \"\\\" is support. e.g. D:\\test\\test.obj");
	program.add_argument("--output").required().help("The complete path of the directory to output the mesh, \"\\\" is support. e.g. D:\\test_split");
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string output_mesh_path = program.get<std::string>("--output");
	std::string model_name = program.get<std::string>("--model");

	boost::filesystem::path fs_model_path(model_name);
	boost::filesystem::path fs_output_mesh_path(output_mesh_path);
	
	tinyobj::attrib_t attribute;
	std::vector<tinyobj::shape_t> shapes;
	std::vector < tinyobj::material_t> materials;
	std::tie(attribute, shapes, materials) = load_obj(fs_model_path.string(), true, fs_model_path.parent_path().string());

	for(auto& item_shape: shapes)
	{
		tinyobj::attrib_t item_attribute(attribute);
		std::vector < tinyobj::material_t> item_materials(materials);
		clean_vertex(item_attribute, item_shape);
		clean_materials(item_shape, item_materials);
		write_obj((boost::filesystem::path(output_mesh_path) / (item_shape.name+".obj")).string(), item_attribute, { item_shape }, item_materials);
	}
}