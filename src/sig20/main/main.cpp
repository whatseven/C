#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>

#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"
#include "airsim_control.h"
#include "map_util.h"


struct Building
{
	CGAL::Bbox_3 bounding_box;
	Point_set points;
};

const Eigen::Vector3f UNREAL_START(0.f, 0.f, 0.f);
const Eigen::Vector3f MAIN_START(-5000.f, 0.f, 2000.f);
const Eigen::Vector3f MAP_START(-70.f, -55.f, 0.f);
const Eigen::Vector3f MAP_END(70.f, 55.f, 35.f);
const float THRESHOLD = 5;
const cv::Vec3b BACKGROUND_COLOR(57,181,55);
Eigen::Matrix3f INTRINSIC;
const bool SYNTHETIC_POINT_CLOUD = true;
const bool MAP_2D_BOX_TO_3D = true;
const float BOUNDS = 20;
const float Z_UP_BOUNDS = 20;
const float Z_DOWN_BOUND = 5;
const float STEP = 5;
const float MM_PI = 3.14159265358;
const bool DOUBLE_FLAG = true;

MapConverter map_converter;


void write_unreal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories,const std::string& v_path)
{
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i)
	{
		const Eigen::Vector3f& position = v_trajectories[i].first*100;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) * 180. / MM_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180./ MM_PI;
		
		pose << (fmt %i% position[0] % -position[1] % position[2] % -pitch% -yaw).str();
	}
	
	pose.close();
}

void write_normal_path(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& v_trajectories, const std::string& v_path) {
	std::ofstream pose(v_path);
	for (int i = 0; i < v_trajectories.size(); ++i) {
		const Eigen::Vector3f& position = v_trajectories[i].first;
		const Eigen::Vector3f& direction = v_trajectories[i].second;
		boost::format fmt("%04d.png,%s,%s,%s,%s,0,%s\n");
		float pitch = std::atan2f(direction[2], std::sqrtf(direction[0] * direction[0] + direction[1] * direction[1])) * 180. / MM_PI;
		float yaw = std::atan2f(direction[1], direction[0]) * 180. / MM_PI;

		pose << (fmt % i % position[0] % position[1] % position[2] % pitch % yaw).str();
	}

	pose.close();
}

int main(int argc, char** argv){
	// Read arguments
	argparse::ArgumentParser program("Jointly exploration, navigation and reconstruction");
	{
		try {
			program.parse_args(argc, argv);
		}
		catch (const std::runtime_error& err) {
			std::cout << err.what() << std::endl;
			std::cout << program;
			exit(0);
		}
	}

	// Prepare environment
	// Reset segmentation color, initialize map converter
	Airsim_tools airsim_client(UNREAL_START);
	{
		map_converter.initDroneStart(UNREAL_START);
		airsim_client.reset_color("building");
		INTRINSIC << 400, 0, 400, 0, 400, 400, 0, 0, 1;
	}

	bool end = false;
	Height_map height_map(MAP_START, MAP_END, 2);
	std::vector<Building> current_buildings;
	std::vector<vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>> previous_trajectory;
	int now_building_ID = 0;
	int now_point_ID = 0;
	while(!end)
	{
		// Get current image and pose
		// Input: 
		// Output: Image(cv::Mat), Camera matrix(Pos_pack)
		std::map<std::string, cv::Mat> current_image;
		Pos_Pack current_pos = map_converter.get_pos_pack_from_unreal(MAIN_START, 0.f, 0.f);
		{
			airsim_client.adjust_pose(current_pos);
			current_image = airsim_client.get_images();
		}

		// SLAM
		// Input: Image(cv::Mat), Camera matrix(cv::Iso)
		// Output: Point cloud, Refined Camera matrix(cv::Iso), num of clusters
		std::vector<Point_set> current_points;
		std::vector<cv::Vec3b> color_map;
		int num_cluster;
		{
			// Calculate roi mask
			const cv::Mat& seg = current_image["segmentation"];
			cv::Mat roi_mask(seg.rows, seg.cols, CV_8UC1);
			roi_mask.forEach<cv::uint8_t>(
				[&seg](cv::uint8_t& val, const int* position) {
				if (seg.at<cv::Vec3b>(position[0], position[1]) != BACKGROUND_COLOR)
					val = 255;
				else
					val = 0;
			});

			if (SYNTHETIC_POINT_CLOUD) {
				std::vector<cv::KeyPoint> keypoints;
				auto orb = cv::ORB::create(3000);
				orb->detect(current_image["rgb"], keypoints, roi_mask);
				cv::drawKeypoints(current_image["rgb"], keypoints, current_image["rgb"]);
				for (auto it = keypoints.begin(); it != keypoints.end(); it++) {
					cv::Vec3b point_color = current_image["segmentation"].at<cv::Vec3b>(it->pt.y, it->pt.x);
					if (point_color == BACKGROUND_COLOR)
						continue;;
					Eigen::Vector3f point(it->pt.x, it->pt.y, 1.f);
					point = INTRINSIC.inverse() * point;
					point *= current_image["depth_planar"].at<float>(it->pt.y, it->pt.x) / point[2];

					auto find_result = std::find(color_map.begin(), color_map.end(), point_color);
					if (find_result == color_map.end()) {
						color_map.push_back(point_color);
						current_points.push_back(Point_set());
						current_points[current_points.size() - 1].insert(Point_3(point(0), point(1), point(2)));
					}
					else {
						current_points[&*find_result - &color_map[0]].insert(Point_3(point(0), point(1), point(2)));
					}
				}
				for (const auto& item : current_points)
					CGAL::write_ply_point_set(std::ofstream(std::to_string(&item - &current_points[0]) + ".ply"), item);
				//debug_img(std::vector{ current_image["rgb"] });
			}
			num_cluster = current_points.size();
		}

		// Post process point cloud
		// Input: Point cloud
		// Output: Clustered Point cloud without ground
		{
			// Done in previous step
		}

		// Mapping
		// Input: *
		// Output: 3D bounding box of current frame
		std::vector<CGAL::Bbox_3> bboxes_3_world_space(num_cluster);
		{
			if (MAP_2D_BOX_TO_3D) {
				// Get bounding box
				std::vector<CGAL::Bbox_2> bboxes(num_cluster);
				std::vector<std::vector<cv::Point>> bboxes_points(num_cluster);
				for (int y = 0; y < current_image["segmentation"].rows; ++y)
					for (int x = 0; x < current_image["segmentation"].cols; ++x) {
						for (int seg_id = 0; seg_id < color_map.size(); ++seg_id) {
							if (color_map[seg_id] == current_image["segmentation"].at<cv::Vec3b>(y, x))
								bboxes_points[seg_id].push_back(cv::Point2f(x, y));
						}
					}
				for (const auto& pixel_points : bboxes_points) {
					cv::Rect2f rect = cv::minAreaRect(pixel_points).boundingRect2f();
					cv::rectangle(current_image["segmentation"], rect, cv::Scalar(0, 0, 255));
					bboxes[&pixel_points - &*bboxes_points.begin()] = CGAL::Bbox_2(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
				}
				debug_img(std::vector{ current_image["segmentation"] });

				// Calculate Z distance and get 3D bounding box
				std::vector<float> z_mins(num_cluster, std::numeric_limits<float>::max());
				std::vector<float> z_maxs(num_cluster, std::numeric_limits<float>::min());
				for (const auto& item_cluster_points : current_points) {
					size_t cluster_index = &item_cluster_points - &current_points[0];
					z_mins[cluster_index] = std::min_element(item_cluster_points.range(item_cluster_points.point_map()).begin(), item_cluster_points.range(item_cluster_points.point_map()).end(),
						[](const auto& a, const auto& b) {
						return a.z() < b.z();
					})->z();
					z_maxs[cluster_index] = std::max_element(item_cluster_points.range(item_cluster_points.point_map()).begin(), item_cluster_points.range(item_cluster_points.point_map()).end(),
						[](const auto& a, const auto& b) {
						return a.z() < b.z();
					})->z();
					Point_set world_points;
					for (const auto& item_point : item_cluster_points.points()) {
						Eigen::Vector3f point_eigen(item_point.x(), item_point.y(), item_point.z());
						point_eigen = current_pos.camera_matrix.inverse() * point_eigen;
						world_points.insert(Point_3(point_eigen.x(), point_eigen.y(), point_eigen.z()));
					}
					bboxes_3_world_space[cluster_index] = get_bounding_box(world_points);
					CGAL::write_ply_point_set(std::ofstream(std::to_string(cluster_index) + "_world.ply"), world_points);
				}

				// Calculate height of the building, Get 3D bbox world space
				Eigen::Matrix3f rotate_to_XZ = Eigen::AngleAxisf(current_pos.pitch, Eigen::Vector3f::UnitX()).toRotationMatrix();
				for (const auto& item_cluster_points : current_points) {
					size_t cluster_index = &item_cluster_points - &current_points[0];
					float min_distance = z_mins[cluster_index];
					float max_distance = z_maxs[cluster_index];
					float y_min_2d = bboxes[cluster_index].ymin();

					Eigen::Vector3f point_pos_img(0, y_min_2d, 1);
					Eigen::Vector3f point_pos_camera_XZ = rotate_to_XZ.inverse() * INTRINSIC.inverse() * point_pos_img;

					float distance_candidate = min_distance;
					float scale = distance_candidate / point_pos_camera_XZ[2];

					float final_height = (point_pos_camera_XZ * scale)[1];
					// Short than camera, recalculate using max distance
					if (final_height > 0) {
						distance_candidate = max_distance;
						scale = distance_candidate / point_pos_camera_XZ[2];
						final_height = (point_pos_camera_XZ * scale)[1];
					}

					bboxes_3_world_space[cluster_index] = CGAL::Bbox_3(bboxes_3_world_space[cluster_index].xmin(), bboxes_3_world_space[cluster_index].ymin(), current_pos.pos_mesh[2] + final_height, bboxes_3_world_space[cluster_index].xmax(), bboxes_3_world_space[cluster_index].ymax(), 0);
				}
			}
		}

		// Merging
		// Input: 3D bounding box of current frame and previous frame
		// Output: Building vectors (std::vector<Building>)
		{
			for (const auto& item_bbox_3 : bboxes_3_world_space) {
				size_t index_box = &item_bbox_3 - &bboxes_3_world_space[0];
				bool is_register_new = true;
				for (auto& item_previous_building : current_buildings) {
					if(do_overlap(item_bbox_3, item_previous_building.bounding_box))
					{
						is_register_new = false;
						item_previous_building.bounding_box += item_bbox_3;
						for (const auto& item_point : current_points[index_box].points())
							item_previous_building.points.insert(item_point);
					}
				}
				if(is_register_new)
				{
					Building building;
					building.bounding_box = item_bbox_3;
					for(const auto& item_point: current_points[index_box].points())
						building.points.insert(item_point);
					current_buildings.push_back(building);
				}
			}
			
		}

		// Generating trajectory
		// No guarantee for the validation of camera position, check it later
		// Input: Building vectors (std::vector<Building>)
		// Output: Total 3D bounding box
		
		std::vector<std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>> current_trajectory(current_buildings.size());
		{
			height_map.save_height_map_png("1.png", 2);
			height_map.save_height_map_tiff("1.tiff");
			for (int id_building = 0; id_building < current_buildings.size(); ++id_building) {
				current_buildings[id_building].bounding_box = get_bounding_box(current_buildings[id_building].points);
				float xmin = current_buildings[id_building].bounding_box.xmin();
				float ymin = current_buildings[id_building].bounding_box.ymin();
				float zmin = current_buildings[id_building].bounding_box.zmin();
				float xmax = current_buildings[id_building].bounding_box.xmax();
				float ymax = current_buildings[id_building].bounding_box.ymax();
				float zmax = current_buildings[id_building].bounding_box.zmax();
				Eigen::Vector3f box_third_points_2(
					(xmin + xmax) / 2,
					(ymin + ymax) / 2,
					(zmin + zmax) / 3 * 2
				);
				Eigen::Vector3f box_third_points(
					(xmin + xmax) / 2,
					(ymin + ymax) / 2,
					(zmin + zmax) / 3
				);


				Eigen::Vector3f cur_pos(xmin - BOUNDS, ymin - BOUNDS, zmax + Z_UP_BOUNDS);
				while (cur_pos.x() <= xmax + BOUNDS) {
					trajectory_current[id_building].push_back(std::make_pair(
						cur_pos, box_third_points_2 - cur_pos
					));
					if (DOUBLE_FLAG) {
						Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
						trajectory_current[id_building].push_back(std::make_pair(
							cur_pos_copy, box_third_points - cur_pos_copy
						));
					}
					cur_pos[0] += STEP;
				}
				while (cur_pos.y() <= ymax + BOUNDS) {
					trajectory_current[id_building].push_back(std::make_pair(
						cur_pos, box_third_points_2 - cur_pos
					));
					if (DOUBLE_FLAG) {
						Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
						trajectory_current[id_building].push_back(std::make_pair(
							cur_pos_copy, box_third_points - cur_pos_copy
						));
					}
					cur_pos[1] += STEP;
				}
				while (cur_pos.x() >= xmin - BOUNDS) {
					trajectory_current[id_building].push_back(std::make_pair(
						cur_pos, box_third_points_2 - cur_pos
					));
					if (DOUBLE_FLAG) {
						Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
						trajectory_current[id_building].push_back(std::make_pair(
							cur_pos_copy, box_third_points - cur_pos_copy
						));
					}
					cur_pos[0] -= STEP;
				}
				while (cur_pos.y() >= ymin - BOUNDS) {
					trajectory_current[id_building].push_back(std::make_pair(
						cur_pos, box_third_points_2 - cur_pos
					));
					if (DOUBLE_FLAG) {
						Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
						trajectory_current[id_building].push_back(std::make_pair(
							cur_pos_copy, box_third_points - cur_pos_copy
						));
					}
					cur_pos[1] -= STEP;
				}

				// Check the camera position
				for (int i = 0; i < trajectory_current[id_building].size(); ++i) {
					Eigen::Vector3f position = trajectory_current[id_building][i].first;
					Eigen::Vector3f camera_focus = trajectory_current[id_building][i].first + trajectory_current[id_building][i].second;

					while (height_map.get_height(position.x(), position.y()) + Z_DOWN_BOUND > position.z()) {
						position[2] += 5;
					}
					Eigen::Vector3f camera_direction = camera_focus - position;
					trajectory_current[id_building][i].second = camera_direction.normalized();
					trajectory_current[id_building][i].first = position;
				}
			}

		}

		// Merging trajectory
		int next_point_ID = 0;
		{
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> single_building_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> single_building_trajectory_previous;
			std::vector<bool> has_taken_picture_current(current_trajectory[now_building_ID].size(), false);
			std::vector<bool> has_taken_picture_previous(previous_trajectory[now_building_ID].size(), false);
			{
				if (!now_point_ID) {
					single_building_trajectory = current_trajectory[now_building_ID];
					single_building_trajectory_previous = previous_trajectory[now_building_ID];
					for (int i = 0; i <= now_point_ID; i++) {
						if (has_taken_picture_previous[i]) {
							Eigen::Vector3f previous_point_coord = single_building_trajectory_previous[i].first;
							for (int j = 0; j < has_taken_picture_current.size(); j++) {
								Eigen::Vector3f now_point_coord = single_building_trajectory[j].first;
								float distance = (now_point_coord - previous_point_coord).norm();
								if (distance < THRESHOLD) {
									has_taken_picture_current[j] = true;
									if (j > next_point_ID)
										next_point_ID = j;
									break;
								}
							}
						}
					}
				}
			}
			previous_trajectory = current_trajectory;
		}

		
		// Shot
		{
			Eigen::Vector3f next_pos = current_trajectory[now_building_ID][now_point_ID].first;
			Eigen::Vector3f next_direction = current_trajectory[now_building_ID][now_point_ID].second;
			// Shot

			
		}


		//for (int i = 0; i < trajectory.size(); ++i) {
		//	point_cloud.insert(Point_3(trajectory[i].first[0], trajectory[i].first[1], trajectory[i].first[2]));
		//}
		//CGAL::write_ply_point_set(std::ofstream("test_point.ply"), point_cloud);
		//write_unreal_path(trajectory, "camera_after_transaction.log");
		//write_normal_path(trajectory, "camera_normal.log");
	}
	
	
	return 0;
}
