#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
//#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <CGAL/cluster_point_set.h>
#include <CGAL/random_selection.h>
#include <CGAL/point_generators_2.h>
#include <json/reader.h>


#include "model_tools.h"
#include "intersection_tools.h"
#include "map_util.h"
#include "viz.h"
#include "building.h"
#include "airsim_control.h"
#include "metrics.h"
#include "trajectory.h"

const Eigen::Vector3f UNREAL_START(0.f, 0.f, 0.f);
//Camera
const cv::Vec3b BACKGROUND_COLOR(57,181,55);
const cv::Vec3b SKY_COLOR(161, 120, 205);
Eigen::Matrix3f INTRINSIC;

MapConverter map_converter;

int main(int argc, char** argv){
	// Read arguments
	Json::Value args;
	{
		FLAGS_logtostderr = 1; 
		google::InitGoogleLogging(argv[0]);
		argparse::ArgumentParser program("Jointly exploration, navigation and reconstruction");
		program.add_argument("--config_file").required();
		try {
			program.parse_args(argc, argv);
			const std::string config_file = program.get<std::string>("--config_file");

			std::ifstream in(config_file);
			if (!in.is_open()) {
				LOG(ERROR) << "Error opening file" << config_file << std::endl;
				return 0;
			}
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
	}
	// Prepare environment
	// Reset segmentation color, initialize map converter
	Visualizer viz;
	Airsim_tools* airsim_client;
	{
		map_converter.initDroneStart(UNREAL_START);
		INTRINSIC << 400, 0, 400, 0, 400, 400, 0, 0, 1;

		if(args["IS_FLYING"].asBool())
		{
			airsim_client = new Airsim_tools(UNREAL_START);
			airsim_client->reset_color("building");
		}
		LOG(INFO) << "Initialization done";
	}
	// Get building if in simulator
	std::vector<Building> fake_buildings;
	{
		CGAL::Point_set_3<Point_3, Vector_3> original_point_cloud;
		CGAL::read_ply_point_set(std::ifstream(args["model_path"].asString()), original_point_cloud);
		CGAL::Point_set_3<Point_3, Vector_3> point_cloud(original_point_cloud);
		Height_map height_map(point_cloud, args["heightmap_resolution"].asFloat());
		height_map.save_height_map_png("map.png", args["HEIGHT_CLIP"].asFloat());
		height_map.save_height_map_tiff("map.tiff");

		// Delete ground planes
		{
			for (int idx = point_cloud.size() - 1; idx >= 0; idx--) {
				if (point_cloud.point(idx).z() < args["HEIGHT_CLIP"].asFloat())
					point_cloud.remove(idx);
			}

			point_cloud.collect_garbage();
		}
		CGAL::write_ply_point_set(std::ofstream("points_without_plane.ply"), point_cloud);
		// Cluster building
		std::size_t nb_clusters;
		{
			Point_set::Property_map<int> cluster_map = point_cloud.add_property_map<int>("cluster", -1).first;

			std::vector<std::pair<std::size_t, std::size_t> > adjacencies;

			nb_clusters = CGAL::cluster_point_set(point_cloud, cluster_map,
				point_cloud.parameters().neighbor_radius(args["cluster_radius"].asFloat()).
				adjacencies(std::back_inserter(adjacencies)));
			fake_buildings.resize(nb_clusters);

			Point_set::Property_map<unsigned char> red = point_cloud.add_property_map<unsigned char>("red", 0).first;
			Point_set::Property_map<unsigned char> green = point_cloud.add_property_map<unsigned char>("green", 0).first;
			Point_set::Property_map<unsigned char> blue = point_cloud.add_property_map<unsigned char>("blue", 0).first;
			for (Point_set::Index idx : point_cloud) {
				// One color per cluster
				int cluster_id = cluster_map[idx];
				CGAL::Random rand(cluster_id);
				red[idx] = rand.get_int(64, 192);
				green[idx] = rand.get_int(64, 192);
				blue[idx] = rand.get_int(64, 192);

				Building& current_building = fake_buildings[cluster_id];
				current_building.points_world_space.insert(point_cloud.point(idx));
			}
		}
		for (int i_building_1 = 0; i_building_1 < fake_buildings.size(); ++i_building_1) {
			fake_buildings[i_building_1].bounding_box_3d = get_bounding_box(fake_buildings[i_building_1].points_world_space);
			fake_buildings[i_building_1].boxes.push_back(fake_buildings[i_building_1].bounding_box_3d);
		}

	}
	
	// Some global structure
	bool end = false;
	const Eigen::Vector3f map_start_unreal(args["MAP_START_UNREAL_X"].asFloat(), args["MAP_START_UNREAL_Y"].asFloat(), args["MAP_START_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_end_unreal(args["MAP_END_UNREAL_X"].asFloat(), args["MAP_END_UNREAL_Y"].asFloat(), args["MAP_END_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_start_mesh(map_start_unreal.x() / 100.f, -map_end_unreal.y() / 100.f, map_start_unreal.z() / 100.f) ;
	const Eigen::Vector3f map_end_mesh(map_end_unreal.x() / 100.f, -map_start_unreal.y() / 100.f, map_end_unreal.z() / 100.f);
	Height_map height_map(map_start_mesh,map_end_mesh,args["heightmap_resolution"].asFloat());
	std::vector<Building> total_buildings;
	Pos_Pack current_pos = map_converter.get_pos_pack_from_unreal(Eigen::Vector3f(args["START_X"].asFloat(), args["START_Y"].asFloat(), args["START_Z"].asFloat()), M_PI / 2, 0);
	int cur_frame_id = 0;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> passed_trajectory;

	Connect_information connect_information(map_start_mesh, map_end_mesh);
	Next_target current_building_id = find_next_target(current_pos, total_buildings, connect_information);
	
	while(!end)
	{
		LOG(INFO) << "<<<<<<<<<<<<< Frame "<<cur_frame_id<<" <<<<<<<<<<<<<";
		std::vector<Building> current_buildings;
		int num_building_current_frame;

		if (args["IS_FLYING"].asBool())
		{
			// Get current image and pose
			// Input: 
			// Output: Image(cv::Mat), Camera matrix(Pos_pack)
			std::map<std::string, cv::Mat> current_image;
			{
				airsim_client->adjust_pose(current_pos);
				current_image = airsim_client->get_images();
				LOG(INFO) << "Image done";
			}

			// SLAM
			// Input: Image(cv::Mat), Camera matrix(cv::Iso)
			// Output: Vector of building with Point cloud in camera frames (std::vector<Building>)
			//		   Refined Camera matrix(cv::Iso)
			//		   num of clusters (int)
			std::vector<cv::Vec3b> color_map;
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

				if (args["SYNTHETIC_POINT_CLOUD"].asBool()) {
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
							current_buildings[current_buildings.size() - 1].segmentation_color = point_color;
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
				num_building_current_frame = current_buildings.size();
				LOG(INFO) << "Sparse point cloud generation and building cluster done";
			}

			// Object detection
			// Input: Vector of building (std::vector<Building>)
			// Output: Vector of building with 2D bounding box (std::vector<Building>)
			{
				std::vector<std::vector<cv::Point>> bboxes_points(num_building_current_frame);
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
				//debug_img(std::vector{ current_image["segmentation"] });
				LOG(INFO) << "Object detection done";
			}

			// Post process point cloud
			// Input: Vector of building (std::vector<Building>)
			// Output: Vector of building with point cloud in world space (std::vector<Building>)
			std::vector<Point_set> current_points_world_space(num_building_current_frame);
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
				if (args["MAP_2D_BOX_TO_3D"].asBool()) {
					// Calculate Z distance and get 3D bounding box
					std::vector<float> z_mins(num_building_current_frame, std::numeric_limits<float>::max());
					std::vector<float> z_maxs(num_building_current_frame, std::numeric_limits<float>::min());
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
						Eigen::Vector3f point_pos_world = current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);

						float final_height = point_pos_world[2];
						// Shorter than camera, recalculate using max distance
						if (final_height < current_pos.pos_mesh[2]) {
							distance_candidate = max_distance;
							scale = distance_candidate / point_pos_camera_XZ[2];
							point_pos_world = current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);
							final_height = point_pos_world[2];
						}

						item_building.bounding_box_3d = get_bounding_box(item_building.points_world_space);
						item_building.bounding_box_3d.min()[2] = 0;
						item_building.bounding_box_3d.max()[2] = final_height;
					}
				}
				LOG(INFO) << "2D Bbox to 3D Bbox done";

			}
		}
		else
		{
			const float distance = 30.f;
			Eigen::AlignedBox3f drone_box(current_pos.pos_mesh - Eigen::Vector3f(distance, distance, distance), current_pos.pos_mesh + Eigen::Vector3f(distance, distance, distance));
			for(int i_building=0;i_building<fake_buildings.size();++i_building)
			{
				if(fake_buildings[i_building].bounding_box_3d.intersects(drone_box))
					current_buildings.push_back(fake_buildings[i_building]);
			}
			num_building_current_frame = current_buildings.size();
		}
		// Merging
		// Input: 3D bounding box of current frame and previous frame
		// Output: Total building vectors (std::vector<Building>)
		{
			std::vector<bool> need_register(num_building_current_frame, true);
			for (auto& item_building : total_buildings) {
				for (const auto& item_current_building : current_buildings) {
					size_t index_box = &item_current_building - &current_buildings[0];
					if (item_current_building.bounding_box_3d.intersects(item_building.bounding_box_3d)) {
						float overlap_volume = item_current_building.bounding_box_3d.intersection(item_building.bounding_box_3d).volume();
						if (overlap_volume / item_current_building.bounding_box_3d.volume() > 0.8 || overlap_volume / item_building.bounding_box_3d.volume() > 0.8)
						{
							need_register[index_box] = false;
							item_building.bounding_box_3d = item_building.bounding_box_3d.merged(item_current_building.bounding_box_3d);
							for (const auto& item_point : item_current_building.points_world_space.points())
								item_building.points_world_space.insert(item_point);
						}
					}
				}
			}
			for (int i = 0; i < need_register.size(); ++i) {
				if (need_register[i]) {
					total_buildings.push_back(current_buildings[i]);
				}
			}

			// Update height map
			for (auto& item_building : total_buildings) {
				height_map.update(item_building.bounding_box_3d);
			}
			//debug_img(std::vector{ height_map.m_map });
			LOG(INFO) << "Building BBox update: DONE!";

		}
		//debug_img(std::vector<cv::Mat>{height_map.m_map, height_map.m_map_dilated});
		//cv::imwrite("map.tiff", height_map.m_map);
		//cv::imwrite("map_d.tiff", height_map.m_map_dilated);
		
		// Generating trajectory
		// No guarantee for the validation of camera position, check it later
		// Input: Building vectors (std::vector<Building>), previous trajectory position
		// Output: Trajectory on current buildings
		//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> current_trajectory;
		{
			Trajectory_params params;
			params.view_distance = args["BOUNDS_MIN"].asFloat();
			params.z_down_bounds = args["Z_DOWN_BOUND"].asFloat();
			params.z_up_bounds = args["Z_UP_BOUNDS"].asFloat();
			params.xy_angle = args["xy_angle"].asFloat();

			params.double_flag = args["double_flag"].asBool();
			params.step = args["step"].asFloat();
			generate_trajectory(params,total_buildings);
			LOG(INFO) << "New trajectory ¡Ì!";
		}

		// Merging trajectory
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos_direction;
		bool is_current_building_done = false;
		{
			if (current_building_id.origin_index_in_building_vector != -1) {
				int cur_building_id = current_building_id.origin_index_in_building_vector;
				if (total_buildings[cur_building_id].passed_trajectory.size() == 0) {
					//next_pos_direction = total_buildings[current_building_id].trajectory[0];
					next_pos_direction = *std::min_element(total_buildings[cur_building_id].trajectory.begin(), total_buildings[cur_building_id].trajectory.end(),
						[&current_pos](const auto& t1, const auto& t2) {
						return (t1.first - current_pos.pos_mesh).norm() < (t2.first - current_pos.pos_mesh).norm();
					});
				}
				else {
					const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = total_buildings[cur_building_id].passed_trajectory;
					std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
					std::copy_if(total_buildings[cur_building_id].trajectory.begin(), total_buildings[cur_building_id].trajectory.end(), std::back_inserter(unpassed_trajectory),
						[&passed_trajectory](const auto& item_new_trajectory) {
						bool untraveled = true;
						for (const auto& item_passed_trajectory : passed_trajectory)
							if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < 5.f) {
								untraveled = false;
							}
						return untraveled;
					});
					if (unpassed_trajectory.size() == 0)
						is_current_building_done = true;
					else
					{
						auto it_min_distance = std::min_element(
							unpassed_trajectory.begin(), unpassed_trajectory.end(),
							[&current_pos](const auto& t1, const auto& t2) {
							return (t1.first - current_pos.pos_mesh).norm() < (t2.first - current_pos.pos_mesh).norm();
						});
						next_pos_direction = *it_min_distance;
					}
				}
			}
			else {
				const auto& p = connect_information.sample_points[current_building_id.origin_index_in_untraveled_pointset];
				next_pos_direction.first = Eigen::Vector3f(p.x(), p.y(), 100.f);
				next_pos_direction.second = Eigen::Vector3f(p.x() + 1.f, p.y(), 100.f);
			}
			LOG(INFO) << "Merge Trajectory ¡Ì";
		}

		// Statics
		{
			passed_trajectory.push_back(next_pos_direction);
			if(current_building_id.origin_index_in_building_vector != -1)
			{
				int cur_building_id = current_building_id.origin_index_in_building_vector;
				if(is_current_building_done)
				{
					LOG(INFO) << "Finish "<< cur_building_id<< "th building, total "<<total_buildings.size()<<" buildings";
					current_building_id = find_next_target(current_pos, total_buildings, connect_information);
				}
				else
				{
					total_buildings[cur_building_id].passed_trajectory.push_back(next_pos_direction);
					LOG(INFO) << "Continue on " << cur_building_id << "th building";
				}
			}
			else {
				LOG(INFO) << "Search on " << current_building_id.origin_index_in_building_vector << "th tile";
				current_building_id = find_next_target(current_pos, total_buildings, connect_information);
				//debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
			}
			connect_information.update_sample_points(current_pos.pos_mesh);

			LOG(INFO) << "Finish frame\n";
		}
		
		//
		// Shot
		//
		// Output: current_pos
		{
			Eigen::Vector3f next_direction = (next_pos_direction.second - next_pos_direction.first).normalized();
			Eigen::Vector3f next_pos = next_pos_direction.first;
			float pitch= -std::atan2f(next_direction[2], std::sqrtf(next_direction[0] * next_direction[0]+ next_direction[1]* next_direction[1]));
			float yaw = std::atan2f(next_direction[1], next_direction[0]);
			current_pos = map_converter.get_pos_pack_from_unreal(map_converter.convertMeshToUnreal(next_pos), yaw, pitch);
			cur_frame_id++;
		}

		// End
		if (current_building_id.origin_index_in_building_vector == current_building_id.origin_index_in_untraveled_pointset)
			break;

		// Visualize
		{
			viz.lock();
			viz.m_buildings = total_buildings;
			Eigen::Vector4f seen(1.0f, 0.f, 0.f, 1.f);
			Eigen::Vector4f unseen(0.0f, 0.f, 0.f, 1.f);
			viz.m_points_color.clear();
			viz.m_points.clear();
			for (const auto& item : connect_information.sample_points)
			{
				viz.m_points.insert(Point_3(item.x(), item.y(), 0.f));
				viz.m_points_color.push_back(connect_information.is_point_traveled[&item - &connect_information.sample_points[0]] ? seen : unseen);
			}
			viz.m_pos = current_pos.pos_mesh;
			viz.m_direction = (next_pos_direction.second - next_pos_direction.first).normalized();
			//viz.m_trajectories = current_trajectory;
			viz.unlock();
			//override_sleep(100);
			debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});

		}

		//for (int i = 0; i < trajectory.size(); ++i) {
		//	point_cloud.insert(Point_3(trajectory[i].first[0], trajectory[i].first[1], trajectory[i].first[2]));
		//}
		//CGAL::write_ply_point_set(std::ofstream("test_point.ply"), point_cloud);
	}
	
	write_unreal_path(passed_trajectory, "camera_after_transaction.log");
	write_normal_path(passed_trajectory, "camera_normal.log");
	LOG(INFO) << "Write trajectory done!";
	return 0;
}
