#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include<corecrt_math_defines.h>

#include "../main/building.h"
#include "map_util.h"

Eigen::Vector2f lonLat2Mercator(const Eigen::Vector2f& lonLat) {
	Eigen::Vector2f mercator;
	double x = lonLat.x() * 20037508.34 / 180;
	double y = log(tan((90 + lonLat.y()) * M_PI / 360)) / (M_PI / 180);
	y = y * 20037508.34 / 180;
	mercator = Eigen::Vector2f(x, y);
	return mercator;
}

Eigen::Vector2f mercator2lonLat(const Eigen::Vector2f& mercator) {
	Eigen::Vector2f lonLat;
	double x = mercator.x() / 20037508.34 * 180;
	double y = mercator.y() / 20037508.34 * 180;
	y = 180 / M_PI * (2 * atan(exp(y * M_PI / 180)) - M_PI / 2);
	lonLat = Eigen::Vector2f(x, y);
	return lonLat;
}

void write_unreal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories,
                       const std::string& v_path)
{
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i)
	{
		const Eigen::Vector3f& position = v_trajectories[i].first * 100;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) *
			180. / M_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / M_PI;

		pose << (fmt % i % position[0] % -position[1] % position[2] % -pitch % -yaw).str();
	}

	pose.close();
}

void write_smith_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories,
                      const std::string& v_path)
{
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i)
	{
		const Eigen::Vector3f& position = v_trajectories[i].first * 100;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) *
			180. / M_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / M_PI;
		yaw = -yaw + 90;
		pose << (fmt % i % -position[0] % position[1] % position[2] % -pitch % yaw).str();
	}

	pose.close();
}

void write_normal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories,
                       const std::string& v_path)
{
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i)
	{
		const Eigen::Vector3f& position = v_trajectories[i].first;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) *
			180. / M_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / M_PI;

		pose << (fmt % i % position[0] % position[1] % position[2] % pitch % yaw).str();
	}

	pose.close();
}

// Second element of the pair is the focus point
std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> interpolate_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories) {
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> interpolated_trajectory;
	for (int i = 0; i < v_trajectories.size()-1; ++i) {
		const Eigen::Vector3f& position = v_trajectories[i].first;
		const Eigen::Vector3f& next_position = v_trajectories[i+1].first;
		const Eigen::Vector3f& next_focus = v_trajectories[i+1].second;
		Eigen::Vector3f towards = next_position-position;
		int num = towards.norm() / 3.f;
		towards.normalize();
		for(int i_interpolate=0;i_interpolate<num;++i_interpolate)
		{
			interpolated_trajectory.push_back(std::make_pair(
				position + towards * i_interpolate,
				next_focus
			));
		}
	}
	interpolated_trajectory.push_back(v_trajectories.back());
	return interpolated_trajectory;
}

// Second element of the pair is the direction
std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> simplify_path_reduce_waypoints(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories) {
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> simplified_trajectory;
	int i0 = 0,i1 = 1;
	Eigen::Vector3f towards(0.f, 0.f, 0.f);
	simplified_trajectory.push_back(v_trajectories[i0]);
	while(i1 < v_trajectories.size())
	{
		const Eigen::Vector3f& position0 = v_trajectories[i1-1].first;
		const Eigen::Vector3f& position1 = v_trajectories[i1].first;
		Eigen::Vector3f next_towards = (position1 - position0).normalized();
		if(towards== Eigen::Vector3f (0.f, 0.f, 0.f))
		{
			i1 += 1;
			towards = next_towards;
		}
		else if (towards.dot(next_towards)>0.99f)
			i1 += 1;
		else
		{
			simplified_trajectory.push_back(v_trajectories[i1-1]);
			i0 = i1;
			towards = next_towards;
			i1 += 1;
		}
	}
	simplified_trajectory.push_back(v_trajectories.back());
	return simplified_trajectory;
}

void write_wgs_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories,const std::string& v_path) {
	Eigen::Vector2f origin_wgs(113.92332,22.64429);
	Eigen::Vector2f origin_xy=lonLat2Mercator(origin_wgs);
	std::ofstream pose(v_path);

	for (int i_id = 0; i_id < v_trajectories.size(); i_id++) {
		const Eigen::Vector3f& position = v_trajectories[i_id].first;
		Eigen::Vector2f pos_mac = Eigen::Vector2f(position.x(), position.y()) + origin_xy;
		Eigen::Vector2f pos_wgs = mercator2lonLat(pos_mac);
		const Eigen::Vector3f& direction = v_trajectories[i_id].second;

		boost::format fmt("%f %f %f %f %f\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) *
			180. / M_PI;
		float yaw = -std::atan2f(direction[1], direction[0]) * 180. / M_PI + 90.f;

		pose << (fmt % pos_wgs[0] % pos_wgs[1] % position[2] % yaw % pitch).str();
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
		if (line.size() < 3)
		{
			std::getline(pose, line);
			continue;
		}
		std::vector<std::string> tokens;
		boost::split(tokens, line, boost::is_any_of(","));

		float pitch = -std::atof(tokens[4].c_str());
		float yaw = -std::atof(tokens[6].c_str());

		float dx = 1.f;
		float dy = std::tanf(yaw / 180.f * M_PI) * dx;
		float dz = std::sqrtf(dx * dx + dy * dy) * std::tanf(pitch / 180.f * M_PI);

		Eigen::Vector3f direction(dx, dy, dz);

		o_trajectories.push_back(std::make_pair(
			Eigen::Vector3f(std::atof(tokens[1].c_str()), -std::atof(tokens[2].c_str()),
			                std::atof(tokens[3].c_str())) / 100,
			direction.normalized()
		));
	}
	while (!pose.eof());

	pose.close();
	return o_trajectories;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> read_normal_trajectory(const std::string& v_path)
{
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> o_trajectories;
	std::ifstream pose(v_path);
	if (!pose.is_open()) throw "File not opened";

	std::string line;
	do
	{
		std::getline(pose, line);
		if (line.size() < 3)
		{
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
	}
	while (!pose.eof());

	pose.close();
	return o_trajectories;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> read_wgs84_trajectory(const std::string& v_path)
{
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> o_trajectories;
	std::ifstream pose(v_path);
	if (!pose.is_open()) throw "File not opened";

	std::string line;
	do
	{
		std::getline(pose, line);
		if (line.size() < 3)
		{
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

		float x = wsg_x * 20037508.34f / 180.f - 12766000;
		float y = log(tan((90 + wsg_y) * M_PI / 360.f)) / (M_PI / 180.f) * 20037508.34f / 180.f - 2590000;

		Eigen::Vector3f direction(dx, dy, dz);

		o_trajectories.push_back(std::make_pair(
			Eigen::Vector3f(x, y, z),
			direction.normalized()
		));
	}
	while (!pose.eof());

	pose.close();
	return o_trajectories;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> read_smith_spline_trajectory(const std::string& v_path)
{
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> o_trajectories;
	std::ifstream pose(v_path);
	if (!pose.is_open()) throw "File not opened";

	std::string line;
	do
	{
		std::getline(pose, line);
		if (line.size() < 3)
		{
			std::getline(pose, line);
			continue;
		}
		std::vector<std::string> tokens;
		boost::split(tokens, line, boost::is_any_of(","));

		float z = std::atof(tokens[3].c_str()) / 100;
		float x = -std::atof(tokens[1].c_str()) / 100;
		float y = std::atof(tokens[2].c_str()) / 100;

		o_trajectories.push_back(std::make_pair(
			Eigen::Vector3f(x, y, z),
			Eigen::Vector3f(0.f, 0.f, 0.f)
		));
	}
	while (!pose.eof());

	pose.close();
	return o_trajectories;
}

float evaluate_length(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> v_trajectory) {
	float total_length = 0;
	if (v_trajectory.size() < 2)
		return 0;
	for (int idx = 0; idx < v_trajectory.size() - 1; ++idx) {
		total_length += (v_trajectory[idx + 1].first - v_trajectory[idx].first).norm();
	}

	return total_length;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> ensure_safe_trajectory_and_calculate_direction(
	const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectory,
	const Height_map& v_height_map,const float Z_UP_BOUNDS)
{
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> safe_trajectory;
	for (auto item : v_trajectory) {
		while (v_height_map.get_height(item.first.x(), item.first.y()) + Z_UP_BOUNDS > item.first.z()) {
			item.first[2] += 5;
		}
		item.second = (item.second - item.first);
		if (item.second.z() > 0)
			item.second.z() = 0;
		item.second = item.second.normalized();
		safe_trajectory.push_back(item);
	}
	return safe_trajectory;
}

struct Trajectory_params
{
	float view_distance;
	float z_up_bounds;
	float z_down_bounds;
	float xy_angle;
	bool double_flag;
	float step;
};

bool generate_next_view_curvature(const Trajectory_params& v_params,
                        Building& v_building,
                        std::pair<Eigen::Vector3f, Eigen::Vector3f>& v_cur_pos)
{
	const Eigen::AlignedBox3f* target_box = nullptr;
	float cur_angle = 0.f;
	bool start_flag = false;
	int cur_box_id = -1;
	float angle_step = v_params.xy_angle / 180.f * M_PI;
	// Init
	if (v_cur_pos.first == Eigen::Vector3f(0.f, 0.f, 0.f))
	{
		start_flag = true;
		const Eigen::AlignedBox3f* closest_box;
		while (closest_box != target_box || target_box == nullptr)
		{
			cur_box_id += 1;
			target_box = &v_building.boxes[cur_box_id];
			Eigen::Vector3f next_position(v_cur_pos.first);
			float radius = std::max(target_box->sizes().x(), target_box->sizes().y()) / 2 + v_params.view_distance;
			next_position.x() = radius * std::cos(cur_angle) + target_box->center().x();
			next_position.y() = radius * std::sin(cur_angle) + target_box->center().y();
			closest_box = &*std::min_element(v_building.boxes.begin(), v_building.boxes.end(),
			                                 [&next_position](const Eigen::AlignedBox3f& item1,
			                                                  const Eigen::AlignedBox3f& item2)
			                                 {
				                                 return (next_position - item1.center()).norm() < (next_position - item2
					                                 .center()).norm();
			                                 });;
		}
		v_building.start_box = cur_box_id;
		cur_angle = v_params.xy_angle / 180.f * M_PI;
	}
	else
	{
		target_box = &*std::min_element(v_building.boxes.begin(), v_building.boxes.end(),
		                                [&v_cur_pos](const Eigen::AlignedBox3f& item1, const Eigen::AlignedBox3f& item2)
		                                {
			                                return (v_cur_pos.first - item1.center()).norm() < (v_cur_pos.first - item2.
				                                center()).norm();
		                                });
		cur_box_id = target_box - &v_building.boxes[0];
		Eigen::Vector3f view_to_center = v_cur_pos.first - target_box->center();
		cur_angle = std::atan2f(view_to_center[1], view_to_center[0]);
	}

	float radius = std::max(target_box->sizes().x(), target_box->sizes().y()) / 2 + v_params.view_distance;
	cur_angle -= angle_step;
	// TODO: UGLY
	//if(target_box == &v_building.boxes[0])
	if (!start_flag && v_building.start_box == cur_box_id && std::abs(cur_angle) < angle_step - 1e-6)
		return false;

	//std::cout << cur_box_id << ", " << cur_angle << std::endl;
	Eigen::Vector3f next_position(v_cur_pos.first);
	Eigen::Vector3f camera_focus = target_box->center();

	next_position.x() = radius * std::cos(cur_angle) + target_box->center().x();
	next_position.y() = radius * std::sin(cur_angle) + target_box->center().y();
	v_cur_pos.first = next_position;
	v_cur_pos.second = camera_focus;
	return true;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> generate_trajectory(const Trajectory_params& v_params,
	std::vector<Building>& v_buildings, const Height_map& v_height_map,const float v_z_up_bound)
{
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> total_trajectory;

	for (int id_building = 0; id_building < v_buildings.size(); ++id_building) {
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> item_trajectory;

		float xmin = v_buildings[id_building].bounding_box_3d.min().x();
		float ymin = v_buildings[id_building].bounding_box_3d.min().y();
		float zmin = v_buildings[id_building].bounding_box_3d.min().z();
		float xmax = v_buildings[id_building].bounding_box_3d.max().x();
		float ymax = v_buildings[id_building].bounding_box_3d.max().y();
		float zmax = v_buildings[id_building].bounding_box_3d.max().z();
		for (int i_pass = 0; i_pass < 2; ++i_pass) {
			Eigen::Vector3f cur_pos(xmin - v_params.view_distance, ymin - v_params.view_distance, zmax + v_params.z_up_bounds);
			Eigen::Vector3f focus_point;
			if (i_pass == 0) {
				focus_point = Eigen::Vector3f(
					(xmin + xmax) / 2,
					(ymin + ymax) / 2,
					(zmin + zmax) / 2
				);
			}
			else if (i_pass == 1) {
				cur_pos.z() /= 2;
				focus_point = Eigen::Vector3f(
					(xmin + xmax) / 2,
					(ymin + ymax) / 2,
					(zmin + zmax) / 5
				);
			}
			while (cur_pos.x() <= xmax + v_params.view_distance) {
				item_trajectory.push_back(std::make_pair(
					cur_pos, focus_point
				));
				cur_pos[0] += v_params.step;
			}
			while (cur_pos.y() <= ymax + v_params.view_distance) {
				item_trajectory.push_back(std::make_pair(
					cur_pos, focus_point
				));
				cur_pos[1] += v_params.step;
			}
			while (cur_pos.x() >= xmin - v_params.view_distance) {
				item_trajectory.push_back(std::make_pair(
					cur_pos, focus_point
				));
				cur_pos[0] -= v_params.step;
			}
			while (cur_pos.y() >= ymin - v_params.view_distance) {
				item_trajectory.push_back(std::make_pair(
					cur_pos, focus_point
				));
				cur_pos[1] -= v_params.step;
			}
			if (!v_params.double_flag)
				break;
		}
		item_trajectory = ensure_safe_trajectory_and_calculate_direction(item_trajectory, v_height_map, v_z_up_bound);
		v_buildings[id_building].trajectory = item_trajectory;
		total_trajectory.insert(total_trajectory.end(), item_trajectory.begin(), item_trajectory.end());
	}
	return total_trajectory;
}