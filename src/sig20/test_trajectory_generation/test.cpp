#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>

#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/building.h"

const float BOUNDS = 5;
const float Z_UP_BOUNDS = 5;
const float Z_DOWN_BOUND = 5;
const float STEP = 3;
const float CENTER_Z1 = 0.5;
const float CENTER_Z2 = 0.25;
const float MM_PI = 3.14159265358;
const bool DOUBLE_FLAG = true;



int main(int argc, char** argv){
	// Read arguments and point clouds
	argparse::ArgumentParser program("Test trajectory generation");
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::vector<Building> buildings(5);
	
	buildings[0].bounding_box_3d=Eigen::AlignedBox3f(Eigen::Vector3f(3, 4, 0),Eigen::Vector3f(30,32,48));
	buildings[1].bounding_box_3d=Eigen::AlignedBox3f(Eigen::Vector3f(-27, 7, 0), Eigen::Vector3f(1, 34, 34));
	buildings[2].bounding_box_3d=Eigen::AlignedBox3f(Eigen::Vector3f(-28, -11, 0), Eigen::Vector3f(-3, -2, 15));
	buildings[3].bounding_box_3d=Eigen::AlignedBox3f(Eigen::Vector3f(-30, -32, 0), Eigen::Vector3f(-4, -20, 25));
	buildings[4].bounding_box_3d=Eigen::AlignedBox3f(Eigen::Vector3f(1, -32, 0), Eigen::Vector3f(29, -3, 30));


	return 0;
}
