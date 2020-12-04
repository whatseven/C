#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
//#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <glog/logging.h>
#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"
#include "airsim_control.h"
#include "map_util.h"


struct Building
{
	CGAL::Bbox_3 bounding_box_3d;
	Point_set points_camera_space;
	Point_set points_world_space;
	CGAL::Bbox_2 bounding_box_2d;
};

const Eigen::Vector3f UNREAL_START(0.f, 0.f, 0.f);
const Eigen::Vector3f MAIN_START(-5000.f, 0.f, 2000.f);
const Eigen::Vector3f MAP_START(-70.f, -55.f, 0.f);
const Eigen::Vector3f MAP_END(70.f, 55.f, 35.f);
const float THRESHOLD = 5;
const cv::Vec3b BACKGROUND_COLOR(57,181,55);
const cv::Vec3b SKY_COLOR(161, 120, 205);
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
	google::InitGoogleLogging(argv[0]);
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
		LOG(INFO) << "Initialization done";
	}

	bool end = false;
	Height_map height_map(MAP_START, MAP_END, 2);
	std::vector<Building> total_buildings;
	int current_building_id = 0;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> previous_trajectory;
	Pos_Pack current_pos = map_converter.get_pos_pack_from_unreal(MAIN_START, -MM_PI / 2, 0);
	int cur_frame_id = 0;
	while(!end)
	{
		LOG(INFO) << "<<<<<<<<<<<<< Frame "<<cur_frame_id<<" <<<<<<<<<<<<<";

		// Get current image and pose
		// Input: 
		// Output: Image(cv::Mat), Camera matrix(Pos_pack)
		std::map<std::string, cv::Mat> current_image;
		{
			airsim_client.adjust_pose(current_pos);
			current_image = airsim_client.get_images();
			LOG(INFO) << "Image done";
		}

		// SLAM
		// Input: Image(cv::Mat), Camera matrix(cv::Iso)
		// Output: Vector of building with Point cloud in camera frames (std::vector<Building>)
		//		   Refined Camera matrix(cv::Iso)
		//		   num of clusters (int)
		std::vector<Building> current_buildings;
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
						current_buildings.push_back(Building());
						current_buildings[current_buildings.size() - 1].points_camera_space.insert(Point_3(point(0), point(1), point(2)));
					}
					else {
						current_buildings[&*find_result - &color_map[0]].points_camera_space.insert(Point_3(point(0), point(1), point(2)));
					}
				}
				for (const auto& item : current_buildings)
					CGAL::write_ply_point_set(std::ofstream(std::to_string(&item - &current_buildings[0]) + ".ply"), item.points_camera_space);
				//debug_img(std::vector{ current_image["rgb"] });
			}
			num_cluster = current_buildings.size();
		}
		
		// Object detection
		// Input: Vector of building (std::vector<Building>)
		// Output: Vector of building with 2D bounding box (std::vector<Building>)
		{
			std::vector<std::vector<cv::Point>> bboxes_points(num_cluster);
			for (int y = 0; y < current_image["segmentation"].rows; ++y)
				for (int x = 0; x < current_image["segmentation"].cols; ++x) {
					for (int seg_id = 0; seg_id < color_map.size(); ++seg_id) {
						if (color_map[seg_id] == current_image["segmentation"].at<cv::Vec3b>(y, x))
							bboxes_points[seg_id].push_back(cv::Point2f(x, y));
					}
				}
			for (const auto& pixel_points : bboxes_points) {
				cv::Rect2f rect = cv::boundingRect(pixel_points);
				cv::rectangle(current_image["segmentation"], rect, cv::Scalar(0, 0, 255));
				current_buildings[&pixel_points - &*bboxes_points.begin()].bounding_box_2d = CGAL::Bbox_2(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
			}
			debug_img(std::vector{ current_image["segmentation"] });
		}

		// Post process point cloud
		// Input: Vector of building (std::vector<Building>)
		// Output: Vector of building with point cloud in world space (std::vector<Building>)
		std::vector<Point_set> current_points_world_space(num_cluster);
		{
			for (auto& item_building : current_buildings) {
				size_t cluster_index = &item_building - &current_buildings[0];
				for (const auto& item_point : item_building.points_camera_space.points()) {
					Eigen::Vector3f point_eigen(item_point.x(), item_point.y(), item_point.z());
					point_eigen = current_pos.camera_matrix.inverse() * point_eigen;
					item_building.points_world_space.insert(Point_3(point_eigen.x(), point_eigen.y(), point_eigen.z()));
				}
				CGAL::write_ply_point_set(std::ofstream(std::to_string(cluster_index) + "_world.ply"), item_building.points_world_space);
			}
			
		}

		// Mapping
		// Input: *
		// Output: Vector of building with 3D bounding box (std::vector<Building>)
		{
			if (MAP_2D_BOX_TO_3D) {
				// Calculate Z distance and get 3D bounding box
				std::vector<float> z_mins(num_cluster, std::numeric_limits<float>::max());
				std::vector<float> z_maxs(num_cluster, std::numeric_limits<float>::min());
				for (const auto& item_building : current_buildings) {
					size_t cluster_index = &item_building - &current_buildings[0];
					z_mins[cluster_index] = std::min_element(item_building.points_camera_space.range(item_building.points_camera_space.point_map()).begin(), item_building.points_camera_space.range(item_building.points_camera_space.point_map()).end(),
						[](const auto& a, const auto& b) {
						return a.z() < b.z();
					})->z();
					z_maxs[cluster_index] = std::max_element(item_building.points_camera_space.range(item_building.points_camera_space.point_map()).begin(), item_building.points_camera_space.range(item_building.points_camera_space.point_map()).end(),
						[](const auto& a, const auto& b) {
						return a.z() < b.z();
					})->z();
				}

				// Calculate height of the building, Get 3D bbox world space
				for (auto& item_building : current_buildings) {
					size_t cluster_index = &item_building - &current_buildings[0];
					float min_distance = z_mins[cluster_index];
					float max_distance = z_maxs[cluster_index];
					float y_min_2d = item_building.bounding_box_2d.ymin();

					Eigen::Vector3f point_pos_img(0, y_min_2d, 1);
					Eigen::Vector3f point_pos_camera_XZ = INTRINSIC.inverse() * point_pos_img;

					float distance_candidate = min_distance;
					float scale = distance_candidate / point_pos_camera_XZ[2];
					Eigen::Vector3f point_pos_world = current_pos.camera_matrix.inverse() * (scale*point_pos_camera_XZ) ;

					float final_height = point_pos_world[2];
					// Shorter than camera, recalculate using max distance
					if (final_height < current_pos.pos_mesh[2]) {
						distance_candidate = max_distance;
						scale = distance_candidate / point_pos_camera_XZ[2];
						point_pos_world = current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);
						final_height = point_pos_world[2];
					}

					item_building.bounding_box_3d = get_bounding_box(item_building.points_world_space);
					item_building.bounding_box_3d = CGAL::Bbox_3(item_building.bounding_box_3d.xmin(), item_building.bounding_box_3d.ymin(), 0, item_building.bounding_box_3d.xmax(), item_building.bounding_box_3d.ymax(), final_height);
				}
			}
		}

		// Merging
		// Input: 3D bounding box of current frame and previous frame
		// Output: Total building vectors (std::vector<Building>)
		{
			std::vector<bool> need_register(num_cluster, true);
			for (auto& item_building : total_buildings) {
				for (const auto& item_current_building : current_buildings) {
					size_t index_box = &item_current_building - &current_buildings[0];
					if (do_overlap(item_current_building.bounding_box_3d, item_building.bounding_box_3d)) {
						need_register[index_box] = false;
						item_building.bounding_box_3d += item_current_building.bounding_box_3d;
						for (const auto& item_point : item_current_building.points_world_space.points())
							item_building.points_world_space.insert(item_point);
					}
				}
			}
			for (int i = 0; i < need_register.size();++i) {
				if(need_register[i])
				{
					total_buildings.push_back(current_buildings[i]);
				}
			}

			// Updata height map
			for (auto& item_building : total_buildings) {
				height_map.update(item_building.bounding_box_3d);
			}
		}

		// Generating trajectory
		// No guarantee for the validation of camera position, check it later
		// Input: Building vectors (std::vector<Building>)
		// Output: Total 3D bounding box
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> current_trajectory;
		{
			Building& building= total_buildings[current_building_id];
			height_map.save_height_map_png("1.png", 2);
			height_map.save_height_map_tiff("1.tiff");

			float xmin = building.bounding_box_3d.xmin();
			float ymin = building.bounding_box_3d.ymin();
			float zmin = building.bounding_box_3d.zmin();
			float xmax = building.bounding_box_3d.xmax();
			float ymax = building.bounding_box_3d.ymax();
			float zmax = building.bounding_box_3d.zmax();
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
				current_trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					current_trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[0] += STEP;
			}
			while (cur_pos.y() <= ymax + BOUNDS) {
				current_trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					current_trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[1] += STEP;
			}
			while (cur_pos.x() >= xmin - BOUNDS) {
				current_trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					current_trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[0] -= STEP;
			}
			while (cur_pos.y() >= ymin - BOUNDS) {
				current_trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					current_trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[1] -= STEP;
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
		}

		// Merging trajectory

		// Returen -1 if shot for the current building is done
		Eigen::Vector3f next_pos;
		Eigen::Vector3f next_direction;
		{
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> points_has_shotted;
			{
				for (auto& point_has_shotted : points_has_shotted) {
					Eigen::Vector3f previous_point_coord = point_has_shotted.first;
					for (auto it = current_trajectory.begin(); it != current_trajectory.end();) {
						Eigen::Vector3f now_point_coord = (*it).first;
						float distance = (now_point_coord - previous_point_coord).norm();
						if (distance < THRESHOLD) {
							point_has_shotted = *it;
							it = current_trajectory.erase(it);
						}
						else
							it++;
					}
				}
			}

			//Find next point to go
			float min_distance = 999;
			std::pair<Eigen::Vector3f, Eigen::Vector3f> next_point;
			if (points_has_shotted.size() == 0)
			{
				next_point = current_trajectory[0];
			}
			else {
				Eigen::Vector3f now_point_coord = points_has_shotted[points_has_shotted.size() - 1].first;
				for (auto it = current_trajectory.begin(); it != current_trajectory.end(); it++) {
					Eigen::Vector3f next_point_coord = (*it).first;
					float distance = (now_point_coord - next_point_coord).norm();
					if (distance < min_distance) {
						next_point = *it;
					}
				}
			}
			
			points_has_shotted.push_back(next_point);
			next_pos = next_point.first;
			next_direction = next_point.second;
			if (current_trajectory.size() == 1)
			{
				current_building_id++;
				points_has_shotted.clear();
			}
		}

		
		// Shot
		//
		// Output: current_pos
		{
			float pitch= -std::atan2f(next_direction[2], std::sqrtf(next_direction[0]* next_direction[0]+ next_direction[1]* next_direction[1]));
			float yaw = std::atan2f(next_direction[1], next_direction[0]);
			current_pos = map_converter.get_pos_pack_from_unreal(map_converter.convertMeshToUnreal(next_pos), yaw, pitch);
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
