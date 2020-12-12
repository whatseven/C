#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>

#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/building.h"
#include "../main/trajectory.h"
#include "../main/viz.h"
#include "common_util.h"

const float BOUNDS = 5;
const float Z_UP_BOUNDS = 5;
const float Z_DOWN_BOUND = 5;
const float STEP = 3;
const float CENTER_Z1 = 0.5;
const float CENTER_Z2 = 0.25;
const bool DOUBLE_FLAG = true;

const std::string asia_ny = "D:\\SIG21\\asia_path\\ny-asiaproxy-our-85-min-smoothnormal-tsp-filter.utj";
const std::string ny_proxy = "C:\\repo\\C\\temp\\ny_sample_points.ply";

float evaluate_length(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> v_trajectory)
{
	float total_length=0;
	for(int idx=0;idx< v_trajectory.size()-1;++idx)
	{
		total_length += (v_trajectory[idx+1].first - v_trajectory[idx].first).norm();
		idx++;
	}

	return total_length;
}

int main(int argc, char** argv){
	// Read arguments
	argparse::ArgumentParser program("Evaluate trajectory");
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory = read_unreal_trajectory(asia_ny);
	Point_set points;
	CGAL::read_ply_point_set(std::ifstream(ny_proxy), points);

	std::cout << "Total length: " << evaluate_length(trajectory) << std::endl;

	Visualizer vizer;

	vizer.lock();
	vizer.m_trajectories = trajectory;
	vizer.m_points = points;
	vizer.unlock();

	override_sleep(3000);
	return 0;
}
