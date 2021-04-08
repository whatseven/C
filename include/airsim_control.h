#pragma once

#include <eigen3/Eigen/Dense>

#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>


#include "map_util.h"
#include "common_util.h"
#include "api/RpcLibClientBase.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "vehicles/multirotor/api/MultirotorCommon.hpp"

class Airsim_tools
{
public:
	Airsim_tools(const Eigen::Vector3f& v_drone_start):m_drone_start(v_drone_start)
	{
		m_agent = new msr::airlib::RpcLibClientBase("127.0.0.1");
		m_agent->confirmConnection();
		/*m_agent = new msr::airlib::MultirotorRpcLibClient("192.168.123.64");
		m_agent->enableApiControl(true);
		m_agent->armDisarm(true);
		m_agent->takeoffAsync(5)->waitOnLastTask();*/
	}

    std::map<std::string, cv::Mat> get_images();
	void adjust_pose(const Pos_Pack& v_pos_pack);
	std::map<cv::Vec3b, std::string> reset_color(std::function<bool(std::string)> v_func);
	std::map<cv::Vec3b, std::string> reset_color(const std::string& v_key_words="");

	Eigen::Vector3f m_drone_start;
    msr::airlib::RpcLibClientBase* m_agent;
    cv::Mat m_color_map;
};

void adjust_pose(msr::airlib::RpcLibClientBase& vAgent, const Pos_Pack& v_pos_pack);

void reset_color(msr::airlib::RpcLibClientBase& vAgent,
    const std::map<std::string, std::string>& v_color_map);

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> demo_move_to_next(msr::airlib::MultirotorRpcLibClient& vAgent,
    const Eigen::Vector3f& v_next_pos_airsim, float angle, const float v_speed, bool is_forward = true);