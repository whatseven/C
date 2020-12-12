#include <argparse/argparse.hpp>

#include "model_tools.h"


int main(int argc, char** argv){
	argparse::ArgumentParser program("Split obj data");
	program.add_argument("--model_directory").required();
	program.add_argument("--model_name").required();
	program.add_argument("--output_directory").required();
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string model_directory = program.get<std::string>("--model_directory");
	std::string output_directory = program.get<std::string>("--output_directory");
	std::string model_name = program.get<std::string>("--model_name");
    rename_material(
		model_directory,
		model_name,
		output_directory
	);
}