#include <tuple>
#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <CGAL/IO/OBJ_reader.h>
#include <CGAL/Surface_mesh/IO.h>

#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/building.h"
#include "../main/trajectory.h"
#include "../main/viz.h"
#include "common_util.h"
#include "metrics.h"

const float BOUNDS = 5;
const float Z_UP_BOUNDS = 5;
const float Z_DOWN_BOUND = 5;
const float STEP = 3;
const float CENTER_Z1 = 0.5;
const float CENTER_Z2 = 0.25;
const bool DOUBLE_FLAG = true;

//
//const std::string trajectory_path = "C:\\repo\\C\\temp\\adjacent\\10_single_upper.log";
//const std::string trajectory_path = "D:\\datasets\\uav_path\\SyntheticProxyAndTrajectory\\NY-1\\Trajectory\\oblique-opti-ny-spline.utj";
const std::string trajectory_path = "D:\\SIG21\\ablation\\2_2\\ny\\436_3.3_20_10_10_-1.log";
const std::string sample_points_path = "D:\\SIG21\\ablation\\2_2\\ny\\ny_points.ply";
//


int main(int argc, char** argv){
	// Read arguments
	argparse::ArgumentParser program("Evaluate trajectory cost");
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory = read_smith_spline_trajectory(trajectory_path);
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory = read_normal_trajectory(trajectory_path);
	Point_set points;
	CGAL::read_ply_point_set(std::ifstream(sample_points_path), points);

	std::cout << "Total length: " << evaluate_length(trajectory) << std::endl;
	std::cout << "Total views: " << trajectory.size() << std::endl;
	
	Visualizer vizer;
	vizer.lock();
	vizer.m_trajectories = trajectory;
	vizer.m_points = points;
	vizer.m_pos = trajectory[0].first;
	vizer.unlock();
	debug_img(std::vector<cv::Mat>{cv::Mat(50, 50, CV_8UC3, cv::Scalar(255, 0, 0))});

	override_sleep(3000);
	return 0;
}
