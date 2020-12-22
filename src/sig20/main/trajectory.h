#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include<corecrt_math_defines.h>

#include "../main/building.h"

void write_unreal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories, const std::string& v_path) {
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i) {
		const Eigen::Vector3f& position = v_trajectories[i].first * 100;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) * 180. / M_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / M_PI;

		pose << (fmt % i % position[0] % -position[1] % position[2] % -pitch % -yaw).str();
	}

	pose.close();
}

void write_normal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories, const std::string& v_path) {
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i) {
		const Eigen::Vector3f& position = v_trajectories[i].first;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) * 180. / M_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / M_PI;

		pose << (fmt % i % position[0] % position[1] % position[2] % pitch % yaw).str();
	}

	pose.close();
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> read_unreal_trajectory(const std::string& v_path)
{
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> o_trajectories;
	std::ifstream pose(v_path);
	if (!pose.is_open()) throw "File not opened";

	std::string line;
	do
	{
		std::getline(pose, line);
		if(line.size() < 3)
		{
			std::getline(pose, line);
			continue;
		}
		std::vector<std::string> tokens;
		boost::split(tokens, line, boost::is_any_of(","));

		float pitch = -std::atof(tokens[4].c_str());
		float yaw = -std::atof(tokens[6].c_str());

		float dx = 1.f;
		float dy = std::tanf(yaw/180.f*M_PI)*dx;
		float dz = std::sqrtf(dx*dx+dy*dy)* std::tanf(pitch / 180.f * M_PI);
		
		Eigen::Vector3f direction(dx,dy,dz);
		
		o_trajectories.push_back(std::make_pair(
			Eigen::Vector3f(std::atof(tokens[1].c_str()), -std::atof(tokens[2].c_str()), std::atof(tokens[3].c_str()))/100,
			direction.normalized()
		));
		
	} while (!pose.eof());
	
	pose.close();
	return o_trajectories;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> read_normal_trajectory(const std::string& v_path) {
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> o_trajectories;
	std::ifstream pose(v_path);
	if (!pose.is_open()) throw "File not opened";

	std::string line;
	do {
		std::getline(pose, line);
		if (line.size() < 3) {
			std::getline(pose, line);
			continue;
		}
		std::vector<std::string> tokens;
		boost::split(tokens, line, boost::is_any_of(","));

		float pitch = std::atof(tokens[4].c_str());
		float yaw = std::atof(tokens[6].c_str());

		float dz = std::sin(pitch / 180.f * M_PI);
		float dxdy = std::cos(pitch / 180.f * M_PI);
		float dy = std::sin(yaw / 180.f * M_PI) * dxdy;
		float dx = std::cos(yaw / 180.f * M_PI) * dxdy;

		Eigen::Vector3f direction(dx, dy, dz);

		o_trajectories.push_back(std::make_pair(
			Eigen::Vector3f(std::atof(tokens[1].c_str()), std::atof(tokens[2].c_str()), std::atof(tokens[3].c_str())),
			direction.normalized()
		));

	} while (!pose.eof());

	pose.close();
	return o_trajectories;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> read_wgs84_trajectory(const std::string& v_path) {
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> o_trajectories;
	std::ifstream pose(v_path);
	if (!pose.is_open()) throw "File not opened";

	std::string line;
	do {
		std::getline(pose, line);
		if (line.size() < 3) {
			std::getline(pose, line);
			continue;
		}
		std::vector<std::string> tokens;
		boost::split(tokens, line, boost::is_any_of(" "));

		float pitch = std::atof(tokens[4].c_str());
		float yaw = std::atof(tokens[3].c_str());

		float dz = std::sin(pitch / 180.f * M_PI);
		float dxdy = std::cos(pitch / 180.f * M_PI);
		float dy = std::cos(yaw / 180.f * M_PI) * dxdy;
		float dx = std::sin(yaw / 180.f * M_PI) * dxdy;

		float wsg_x = std::atof(tokens[0].c_str());
		float wsg_y = std::atof(tokens[1].c_str());
		float z = std::atof(tokens[2].c_str());

		float x= wsg_x *20037508.34f / 180.f- 12766000;
		float y= log(tan((90 + wsg_y) * M_PI / 360.f)) / (M_PI / 180.f) * 20037508.34f / 180.f- 2590000;

		Eigen::Vector3f direction(dx, dy, dz);

		o_trajectories.push_back(std::make_pair(
			Eigen::Vector3f(x, y, z),
			direction.normalized()
		));

	} while (!pose.eof());

	pose.close();
	return o_trajectories;
}

struct Trajectory_params
{
	float view_distance;
	float z_up_bounds;
	float z_down_bounds;
	float xy_angle;
};

void generate_trajectory(const Trajectory_params& v_params,
	const Eigen::AlignedBox3f& v_box, 
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectory,
	const Height_map& v_height_map)
{
	// Golden angle 23
	float horizontal_step = v_params.view_distance * tan(11.5f / 180.f * 3.1415926f) * 2;

	float vertical_step = v_params.view_distance * tan(11.5f / 180.f * 3.1415926f) * 2; // Ensure 23 degree in vertical
	int horizontal_step_num = 360.f / v_params.xy_angle + 1;
	if (horizontal_step_num * vertical_step > v_box.sizes().z())
		vertical_step = (v_box.sizes().z() + v_params.z_up_bounds - v_params.z_down_bounds) / horizontal_step_num;
	float focus_step = (v_box.sizes().z() - v_params.z_down_bounds) / horizontal_step_num;

	//
	float radius = std::max(v_box.sizes().x(), v_box.sizes().y()) / 2 + v_params.view_distance;
	Eigen::Vector3f next_position(0, 0, v_params.z_down_bounds + 1);
	float previous_trajectory_xy_angle;
	Eigen::Vector3f camera_focus;

	while (next_position.z() > v_params.z_down_bounds) {
		if (v_trajectory.size() == 0) {
			previous_trajectory_xy_angle = 0.f;
			next_position.x() = radius * std::cos(previous_trajectory_xy_angle) + v_box.center().x();
			next_position.y() = radius * std::sin(previous_trajectory_xy_angle) + v_box.center().y();
			next_position.z() = v_box.max().z() + v_params.z_up_bounds;
			camera_focus = v_box.center();
		}
		else {
			previous_trajectory_xy_angle -= v_params.xy_angle / 180.f * M_PI;
			next_position.x() = radius * std::cos(previous_trajectory_xy_angle) + v_box.center().x();
			next_position.y() = radius * std::sin(previous_trajectory_xy_angle) + v_box.center().y();
			next_position.z() -= vertical_step;
			//camera_focus.z() -= focus_step;
			camera_focus = v_box.center();
		}

		v_trajectory.push_back(std::make_pair(next_position, camera_focus));
	}
}


std::pair<Eigen::Vector3f, Eigen::Vector3f> generate_next_view(const Trajectory_params& v_params,
	const Building& v_building,
	const Eigen::Vector3f& v_cur_pos,
	const bool v_is_inverse_order) {

	const Eigen::AlignedBox3f* target_box;
	float cur_angle;
	if(v_cur_pos==Eigen::Vector3f(0.f,0.f,0.f))
	{
		target_box = &v_building.boxes[0];
		cur_angle = (v_is_inverse_order ? 0 : 0);
	}
	else
	{
		target_box = &*std::min_element(v_building.boxes.begin(), v_building.boxes.end(),
			[&v_cur_pos](const Eigen::AlignedBox3f& item1, const Eigen::AlignedBox3f& item2) {return (v_cur_pos - item1.center()).norm() < (v_cur_pos - item2.center()).norm(); });
		Eigen::Vector3f view_to_center = v_cur_pos - target_box->center();
		cur_angle = std::atan2f(view_to_center[1], view_to_center[0]);
	}
	
	float radius = std::max(target_box->sizes().x(), target_box->sizes().y()) / 2 + v_params.view_distance;
	cur_angle-= v_params.xy_angle / 180.f * M_PI;
	if(target_box == &v_building.boxes[0])
		if(!v_is_inverse_order&& std::abs(cur_angle) < v_params.xy_angle / 180.f * M_PI - 1e-6)
			return std::make_pair(Eigen::Vector3f(0.f, 0.f, 0.f), Eigen::Vector3f(0.f, 0.f, 0.f));
		else if(v_is_inverse_order&& std::abs(cur_angle)< v_params.xy_angle / 180.f * M_PI - 1e-6)
			return std::make_pair(Eigen::Vector3f(0.f, 0.f, 0.f), Eigen::Vector3f(0.f, 0.f, 0.f));

	//std::cout << target_box - &v_building.boxes[0] << ", " << cur_angle << std::endl;
	Eigen::Vector3f next_position(v_cur_pos);
	Eigen::Vector3f camera_focus=target_box->center();
	
	next_position.x() = radius * std::cos(cur_angle) + target_box->center().x();
	next_position.y() = radius * std::sin(cur_angle) + target_box->center().y();
	
	return std::make_pair(next_position, camera_focus);
}