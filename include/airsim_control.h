#pragma once

#include <eigen3/Eigen/Dense>

#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>


#include "map_util.h"
#include "common_util.h"
#include "api/RpcLibClientBase.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

class Airsim_tools
{
public:
	Airsim_tools(const Eigen::Vector3f& v_drone_start):m_drone_start(v_drone_start)
	{
        m_agent = new msr::airlib::RpcLibClientBase();
		m_agent->confirmConnection();
	}

    std::map<std::string, cv::Mat> get_images();
	void adjust_pose(const Pos_Pack& v_pos_pack);
	void reset_color(const std::string& v_key_words="");

	Eigen::Vector3f m_drone_start;
    msr::airlib::RpcLibClientBase* m_agent;
};

void adjust_pose(msr::airlib::RpcLibClientBase& vAgent, const Pos_Pack& v_pos_pack);

void reset_color(msr::airlib::RpcLibClientBase& vAgent,
    const std::map<std::string, std::string>& v_color_map);

void demo_move_to_next(msr::airlib::MultirotorRpcLibClient& vAgent,
    const Eigen::Vector3f& v_next_pos_airsim,const float v_speed);