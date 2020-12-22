#include <argparse/argparse.hpp>

#include "model_tools.h"


int main(int argc, char** argv){
	argparse::ArgumentParser program("Split obj data");
	program.add_argument("--model_directory").required();
	program.add_argument("--model_name").required();
	program.add_argument("--resolution").required();
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
	// Shenzhen v_filter_height: -8
	split_obj(
		model_directory,
		model_name,
		resolution,
		15,
		500
	);
}