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
	/*
	 * Yingrenshi geopoint: (12766000.f, 2590000.f)
	 * Shenzhen Univ geopoint: (12682806.58230256f, 2575925.3710538405f)
	 */

	
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
			if(foot_print_path.extension().string()==".foot")
				proxys.push_back(load_footprint(foot_print_path.string()));
		}
	}
	for(auto& item_proxy:proxys)
	{
		for(auto iter= item_proxy.polygon.vertices_begin();iter!= item_proxy.polygon.vertices_end();++iter)
		{
			//double x = (*iter).x() * 20037508.34 / 180;
			//double y = log(tan((90 + (*iter).y()) * M_PI / 360)) / (M_PI / 180);
			//y = y * 20037508.34 / 180;
			//(*iter) = CGAL::Point_2<K>(x - 12766000.f, y - 2590000.f);
			//(*iter) = CGAL::Point_2<K>(x - 12682806.58230256f, y - 2575925.3710538405f);
			//(*iter) = CGAL::Point_2<K>((*iter).x()*0.92, (*iter).y()*0.93);
		}

		Polygon_2 polygon;
		Point_2 last_point;
		for (auto iter = item_proxy.polygon.vertices_begin();iter != item_proxy.polygon.vertices_end();++iter)
		{
			if (iter != item_proxy.polygon.vertices_begin() && CGAL::squared_distance((*iter), last_point) < 2)
				continue;
			polygon.push_back(*iter);
			last_point = *iter;
		}
		//item_proxy.polygon = polygon;
	}
	
	std::vector<tinyobj::shape_t> out_shapes=split_obj_according_to_footprint(attribute, shapes, proxys,5.0f);
	tinyobj::shape_t ground = out_shapes[out_shapes.size() - 1];
	out_shapes.pop_back();
	write_obj((boost::filesystem::path(output_mesh_path)/"split_without_ground.obj").string(), attribute, out_shapes, materials);
    for(int i_shape=0;i_shape<out_shapes.size();++i_shape)
    {
		std::vector < tinyobj::material_t> item_materials(materials);
		tinyobj::attrib_t item_attribute(attribute);
		clean_vertex(item_attribute, out_shapes[i_shape]);
		clean_materials(out_shapes[i_shape], item_materials);
		write_obj((boost::filesystem::path(output_mesh_path) / (std::to_string(i_shape)+".obj")).string(), item_attribute, std::vector<tinyobj::shape_t>{out_shapes[i_shape]}, item_materials);
    }
	write_obj((boost::filesystem::path(output_mesh_path) / "ground.obj").string(), attribute, { ground }, materials);
}