#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>

#include <boost/format.hpp>
#include <glog/logging.h>

#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/viz.h"
#include "../main/building.h"
#include "../main/trajectory.h"
#include "common_util.h"
#include <json/json.h>

const boost::filesystem::path trajectory_path("D:\\SIG21_Local\\2_2_error_map\\bridge_asia_565\\camera_after_transaction.log");

const Eigen::Vector2f CHIKAN3(12513425.f, 2627499.25f);
const Eigen::Vector2f CHIKAN4(12513225.f, 2627233.25f);

int main(int argc, char** argv) {
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	//= read_normal_trajectory(trajectory_path.string());
	//std::ofstream pose((trajectory_path.parent_path() / "chikancun4_wgs.txt").string());

	//std::ifstream f_in(trajectory_path.string());
	//while (!f_in.eof()) {
	//	std::string line;
	//	std::getline(f_in, line);
	//	std::vector<std::string> tokens;
	//	boost::split(tokens, line, boost::is_any_of(","));
	//	trajectory.emplace_back(
	//		Eigen::Vector3f(std::atof(tokens[0].c_str()), std::atof(tokens[1].c_str()), std::atof(tokens[2].c_str())),
	//		Eigen::Vector3f(std::atof(tokens[3].c_str()), std::atof(tokens[4].c_str()), std::atof(tokens[5].c_str())));
	//}
	//f_in.close();
	trajectory= read_smith_trajectory(trajectory_path.string());
	write_normal_path(trajectory, (trajectory_path.parent_path().parent_path() / "asia18_bridge_565.txt").string());
	
	//for (auto& item : trajectory) {
	//	Eigen::Vector3f direction = item.second;
	//	item.first.x() += CHIKAN4.x();
	//	item.first.y() += CHIKAN4.y();
	//	double x = item.first.x() / 20037508.34 * 180;
	//	double y = item.first.y() / 20037508.34 * 180;
	//	y = 180 / M_PI * (2 * atan(exp(y * M_PI / 180)) - M_PI / 2);
	//	item.first = Eigen::Vector3f(x, y, item.first.z());

	//	float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) *
	//		180. / M_PI;
	//	float yaw = -std::atan2f(direction[1], direction[0]) * 180. / M_PI + 90.f;
	//	boost::format fmt("%f %f %f %f %f\n");
	//	pose << (fmt % item.first[0] % item.first[1] % item.first[2] % yaw % pitch).str();

	//}
	//pose.close();
	return 0;
}
