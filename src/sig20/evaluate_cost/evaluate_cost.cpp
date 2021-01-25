#include <tuple>
#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <CGAL/IO/OBJ_reader.h>
#include <CGAL/Surface_mesh/IO.h>

#include <boost/format.hpp>
#include <glog/logging.h>
#include <json/reader.h>

#include "airsim_control.h"
#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/building.h"
#include "../main/trajectory.h"
#include "../main/viz.h"
#include "common_util.h"
#include "metrics.h"

const float ACCELERATE = 1;
const float MAX_SPEED = 8;
MapConverter map_converter;
void fixed_point_fly(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectory)
{
	msr::airlib::MultirotorRpcLibClient m_agent;
	m_agent.confirmConnection();
	m_agent.enableApiControl(true);
	m_agent.takeoffAsync()->waitOnLastTask();
	MapConverter map_converter;
	//map_converter.initDroneStart(Eigen::Vector3f(0.f, 0.f, 200.f));
	map_converter.initDroneStart(Eigen::Vector3f(-16000, -16000.f, 1000.f));

	Pos_Pack pos_start = map_converter.get_pos_pack_from_mesh(v_trajectory[0].first, 0, 0);
	m_agent.moveToPositionAsync(pos_start.pos_airsim.x(), pos_start.pos_airsim.y(), pos_start.pos_airsim.z(), 6)->waitOnLastTask();

	LOG(INFO) << "Start capture";
	auto recorder = recordTime();
	for(int i=1;i<v_trajectory.size();++i)
	{
		Eigen::Vector3f next_pos = v_trajectory[i].first;
		Eigen::Vector3f direction = next_pos- v_trajectory[i].first;
		float distance = direction.norm();
		float speed = 0;
		float max_speed;
		if (MAX_SPEED / 2 * (MAX_SPEED / ACCELERATE) * 2 < distance)
			max_speed = MAX_SPEED;
		
		Pos_Pack pos= map_converter.get_pos_pack_from_mesh(next_pos, 0, 0);

		Pose pose = m_agent.simGetVehiclePose();
		Eigen::Vector3f pos_cur = pose.position;

		float cur_time;
		while (true) {
			pose = m_agent.simGetVehiclePose();
			pos_cur = pose.position;

			Eigen::Vector3f direction = pos.pos_airsim - pos_cur;
			direction.normalize();
			direction = direction * speed;
			m_agent.moveByVelocityAsync(direction[0], direction[1], direction[2], 20);

			Eigen::Vector3f pos_cur = pose.position;
			if ((pos_cur - pos.pos_airsim).norm() < 1)
				break;

			override_sleep(0.05);
		}
	}
	profileTime(recorder, "Done, total time:");
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> continuous_fly(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectory) {
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	
	msr::airlib::MultirotorRpcLibClient m_agent;
	m_agent.confirmConnection();
	m_agent.enableApiControl(true);
	m_agent.takeoffAsync()->waitOnLastTask();


	Pos_Pack pos_start = map_converter.get_pos_pack_from_mesh(v_trajectory[0].first, 0, 0);
	m_agent.moveToPositionAsync(pos_start.pos_airsim.x(), pos_start.pos_airsim.y(), pos_start.pos_airsim.z(), 6)->waitOnLastTask();

	LOG(INFO) << "Start capture";
	auto recorder = recordTime();
	for (int i = 1; i < v_trajectory.size(); ++i) {
		Eigen::Vector3f next_pos = v_trajectory[i].first;
		Eigen::Vector3f cur_direction = (next_pos - v_trajectory[i-1].first).normalized();
		Eigen::Vector3f next_direction = (v_trajectory[i +1].first - next_pos).normalized();
		float speed = 6;
		if (i < v_trajectory.size() - 1 && cur_direction.dot(next_direction) < 0.9f)
			speed = 2;
			
		Pos_Pack pos = map_converter.get_pos_pack_from_mesh(next_pos, 0, 0);
		auto poses = demo_move_to_next(m_agent, pos.pos_airsim, speed,false);
		for(auto& item: poses)
		{
			item.first = map_converter.convertAirsimToMesh(item.first);
		}
		trajectory.insert(trajectory.end(), poses.begin(), poses.end());
		LOG(INFO) << i << "/" << v_trajectory.size();
	}
	profileTime(recorder, "Done, total time:");
	return trajectory;
}

int main(int argc, char** argv){
	// Read arguments
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);
	Json::Value args;
	argparse::ArgumentParser program("Evaluate trajectory cost");
	try {
		program.add_argument("--config_file").required();
		program.parse_args(argc, argv);
		const std::string config_file = program.get<std::string>("--config_file");
		std::ifstream in(config_file);
		Json::Reader json_reader;
		if (!json_reader.parse(in, args)) {
			LOG(ERROR) << "Error parse config file" << config_file << std::endl;
			return 0;
		}
		in.close();
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	map_converter.initDroneStart(Eigen::Vector3f(
		args["drone_start_x"].asFloat(), 
		args["drone_start_y"].asFloat(), 
		args["drone_start_z"].asFloat() 
		));

	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory = read_smith_spline_trajectory(trajectory_path);
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	if(!args["is_smith"].asBool())
		trajectory = read_normal_trajectory(args["trajectory_path"].asString());
	else
		trajectory = read_smith_trajectory(args["trajectory_path"].asString());

	Point_set points;
	CGAL::read_ply_point_set(std::ifstream(args["sample_points_path"].asString(),std::ios::binary), points);

	std::cout << "Total length: " << evaluate_length(trajectory) << std::endl;
	std::cout << "Total views: " << trajectory.size() << std::endl;
	
	Visualizer vizer;
	vizer.lock();
	vizer.m_trajectories = trajectory;
	vizer.m_points = points;
	vizer.m_pos = trajectory[0].first;
	vizer.unlock();

	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> runtime_trajectory;
	if(args["continuous_fly"].asBool())
		runtime_trajectory=continuous_fly(trajectory);
	else
		fixed_point_fly(trajectory);

	debug_img(std::vector<cv::Mat>{cv::Mat(50, 50, CV_8UC3, cv::Scalar(255, 0, 0))});

	vizer.lock();
	vizer.m_trajectories_spline = runtime_trajectory;
	vizer.unlock();

	override_sleep(3000);
	return 0;
}
