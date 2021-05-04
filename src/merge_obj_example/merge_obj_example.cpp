#include <argparse/argparse.hpp>
#include <boost/filesystem.hpp>

#include "model_tools.h"


int main(int argc, char** argv){
	argparse::ArgumentParser program("Recursively iterate directory and merge obj into one file. Note: Currently only the diffuse, specular, normal texture are copied");
	program.add_argument("--model_directory").required();
	program.add_argument("--output_directory").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	boost::filesystem::path input_dir(program.get<std::string>("--model_directory"));
	boost::filesystem::path output_dir(program.get<std::string>("--output_directory"));

	boost::filesystem::recursive_directory_iterator iter_end;

	if (!boost::filesystem::exists(output_dir.parent_path()))
		boost::filesystem::create_directories(output_dir.parent_path());

	std::vector<tinyobj::attrib_t> attribs;
	std::vector < std::vector<tinyobj::shape_t>> shapes;
	std::vector < std::vector<tinyobj::material_t>> mtls;

	int i = 0;
	for(boost::filesystem::recursive_directory_iterator iter_cur(input_dir);iter_cur!=iter_end;++iter_cur)
	{
		if(boost::filesystem::is_directory(iter_cur->path()))
			continue;
		if(iter_cur->path().extension()!=".obj")
			continue;

		auto result = load_obj(iter_cur->path().string(), true, iter_cur->path().parent_path().string());
		attribs.push_back(std::get<0>(result));
		shapes.push_back(std::get<1>(result));
		mtls.push_back(std::get<2>(result));
		++i;
	}
	merge_obj(output_dir.string(), attribs, shapes, mtls);
}