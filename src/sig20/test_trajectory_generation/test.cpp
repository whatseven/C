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

	Building_Set buildings(5);
	
	buildings.buildings[0].bounding_box_3d = Eigen::AlignedBox3f(Eigen::Vector3f(3, 4, 0), Eigen::Vector3f(30, 32, 48));
	buildings.buildings[1].bounding_box_3d = Eigen::AlignedBox3f(Eigen::Vector3f(-27, 7, 0), Eigen::Vector3f(1, 34, 34));
	buildings.buildings[2].bounding_box_3d = Eigen::AlignedBox3f(Eigen::Vector3f(-28, -11, 0), Eigen::Vector3f(-3, -2, 15));
	buildings.buildings[3].bounding_box_3d = Eigen::AlignedBox3f(Eigen::Vector3f(-30, -32, 0), Eigen::Vector3f(-4, -20, 25));
	buildings.buildings[4].bounding_box_3d = Eigen::AlignedBox3f(Eigen::Vector3f(1, -32, 0), Eigen::Vector3f(29, -3, 30));

	// Calculate how many iteration needed
	std::vector<float> z_heights;
	float zmax = 0;
	for (int i = 0; i < buildings.buildings.size(); i++)
	{
		zmax = std::max(buildings.buildings[i].bounding_box_3d.max()[2], zmax);
		z_heights.push_back(buildings.buildings[i].bounding_box_3d.max()[2]);
	}
	std::vector<int> indices(z_heights.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::stable_sort(indices.begin(), indices.end(), [&z_heights](int i1, int i2) {return z_heights[i1] > z_heights[i2]; });

	int iteration_num = (zmax + BOUNDS) / (2 * BOUNDS);
	if (iteration_num == 0)
		iteration_num = 1;

	float now_z = zmax + BOUNDS;
	for (int i = 0; i < iteration_num; i++)
	{
		// Add new building into trajectory
		for (int j = buildings.id_sequence.size(); j < buildings.buildings.size(); j++)
		{
			if (now_z <= buildings.buildings[indices[j]].bounding_box_3d.max()[2] + BOUNDS)
			{
				buildings.id_sequence.push_back(indices[j]);
			}
		}

		// Generate trajectory
		for (int j = 0; j < buildings.id_sequence.size() * 2; j++)
		{
			if (buildings.id_sequence.size() == 1)
			{
				//Set gaze target
				Eigen::Vector3f gaze_target = buildings.buildings[buildings.id_sequence[j]].center;
				gaze_target[2] = now_z - BOUNDS * 2;

				Eigen::Vector3f seg_start, seg_end, current_pos;

				for (int j = 0; j < vertical_step_num; j++)
				{
					seg_start = top_left + (left - top_left) / vertical_step_num * j;
					seg_end = left + (bottom_left - left) / vertical_step_num * j;
					current_pos = seg_start + (seg_end - seg_start) / vertical_step_num * j;
					current_pos[2] = iteration_zmax - z_step * j;
					current_trajectory.push_back(std::make_pair(current_pos, gaze_target - current_pos));
				}

				for (int j = 0; j < horizontal_step_num; j++)
				{
					seg_start = bottom_left + ((bottom - bottom_left) / horizontal_step_num * j);
					seg_end = bottom + (bottom_right - bottom) / horizontal_step_num * j;
					current_pos = seg_start + (seg_end - seg_start) / horizontal_step_num * j;
					current_pos[2] = iteration_zmax - z_step * j - vertical_step_num * z_step;
					current_trajectory.push_back(std::make_pair(current_pos, gaze_target - current_pos));
				}

				for (int j = 0; j < vertical_step_num; j++)
				{
					seg_start = bottom_right + (right - bottom_right) / vertical_step_num * j;
					seg_end = right + (top_right - right) / vertical_step_num * j;
					current_pos = seg_start + (seg_end - seg_start) / vertical_step_num * j;
					current_pos[2] = iteration_zmax - z_step * j - (vertical_step_num + horizontal_step_num) * z_step;
					current_trajectory.push_back(std::make_pair(current_pos, gaze_target - current_pos));
				}

				for (int j = 0; j < horizontal_step_num; j++)
				{
					seg_start = top_right + (top - top_right) / horizontal_step_num * j;
					seg_end = top + (top_left - top) / horizontal_step_num * j;
					current_pos = seg_start + (seg_end - seg_start) / horizontal_step_num * j;
					current_pos[2] = iteration_zmax - z_step * j - (2 * vertical_step_num + horizontal_step_num) * z_step;
					current_trajectory.push_back(std::make_pair(current_pos, gaze_target - current_pos));
				}
			}

			
			else
			{

			}
		}
		// Check the camera position
		for (int i = 0; i < current_trajectory.size(); ++i) {
			Eigen::Vector3f position = current_trajectory[i].first;
			Eigen::Vector3f camera_focus = current_trajectory[i].first + current_trajectory[i].second;

			while (height_map.get_height(position.x(), position.y()) + Z_DOWN_BOUND > position.z()) {
				position[2] += 5;
			}
			Eigen::Vector3f camera_direction = camera_focus - position;
			current_trajectory[i].second = camera_direction.normalized();
			current_trajectory[i].first = position;
		}
		LOG(INFO) << "New trajectory GENERATED!";

		now_z -= 2 * BOUNDS;
	}


	return 0;
}
