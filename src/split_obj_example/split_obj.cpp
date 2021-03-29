#include <argparse/argparse.hpp>

#include "model_tools.h"


int main(int argc, char** argv){
	argparse::ArgumentParser program("Split obj data");
	program.add_argument("--model_directory").required();
	program.add_argument("--model_name").required();
	program.add_argument("--filter_height").required();
	program.add_argument("--obj_max_building_num").required();
	program.add_argument("--resolution").required();
	program.add_argument("--output_directory").required();
	program.add_argument("--split_axis").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string model_directory = program.get<std::string>("--model_directory");
	std::string model_name = program.get<std::string>("--model_name");
	float resolution = std::atof(program.get<std::string>("--resolution").c_str());
	float filter_height = std::atof(program.get<std::string>("--filter_height").c_str());
	int obj_max_building_num = std::atof(program.get<std::string>("--obj_max_building_num").c_str());
	std::string output_directory = program.get<std::string>("--output_directory").c_str();
	int split_axis = std::atoi(program.get<std::string>("--split_axis").c_str());
	// Shenzhen v_filter_height: -8
	split_obj(
		model_directory,
		model_name,
		resolution,
		filter_height,
		obj_max_building_num,
		output_directory,
		split_axis
	);
}