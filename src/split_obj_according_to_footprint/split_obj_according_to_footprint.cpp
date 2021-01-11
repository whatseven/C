#include <argparse/argparse.hpp>

#include "model_tools.h"
#include <boost/filesystem.hpp>
#include <corecrt_math_defines.h>

int main(int argc, char** argv){
	argparse::ArgumentParser program("Split obj data");
	program.add_argument("--model_directory").required();
	program.add_argument("--model_name").required();
	program.add_argument("--footprint_directory").required();
	program.add_argument("--output_mesh_path").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string footprint_directory = program.get<std::string>("--footprint_directory");
	std::string model_directory = program.get<std::string>("--model_directory");
	std::string output_mesh_path = program.get<std::string>("--output_mesh_path");
	std::string model_name = program.get<std::string>("--model_name");

	tinyobj::attrib_t attribute;
	std::vector<tinyobj::shape_t> shapes;
	std::vector < tinyobj::material_t> materials;
	std::tie(attribute, shapes, materials) = load_obj(model_name, true, model_directory);
	std::vector<Proxy> proxys;

	boost::filesystem::path foot_print_path(footprint_directory);
	boost::filesystem::directory_iterator endIter;
	for (boost::filesystem::directory_iterator iter(foot_print_path); iter != endIter; iter++) {
		if (!boost::filesystem::is_directory(*iter)) {
			boost::filesystem::path foot_print_path(iter->path().string());
			if(foot_print_path.extension().string()==".foot_print")
				proxys.push_back(load_footprint(foot_print_path.string()));
		}
	}
	for(auto& item_proxy:proxys)
	{
		for(auto iter= item_proxy.polygon.vertices_begin();iter!= item_proxy.polygon.vertices_end();++iter)
		{
			
			double x = (*iter).x() * 20037508.34 / 180;
			double y = log(tan((90 + (*iter).y()) * M_PI / 360)) / (M_PI / 180);
			y = y * 20037508.34 / 180;
			(*iter) = CGAL::Point_2<K>(x - 12766000.f, y - 2590000.f);
		}
	}
	
	std::vector<tinyobj::shape_t> out_shapes=split_obj_according_to_footprint(attribute, shapes, proxys,0.9f);
	write_obj(output_mesh_path, attribute, out_shapes, materials);
    
}