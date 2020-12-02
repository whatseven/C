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
const Eigen::Vector2f IMAGE_START(-70.f, -55.f);
const Eigen::Vector3f MAIN_START(-5000.f, 0.f, 2000.f);
const cv::Vec3b BACKGROUND_COLOR(57,181,55);
Eigen::Matrix3f INTRINSIC;
const bool SYNTHETIC_POINT_CLOUD = true;
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
		map_converter.initImageStart(IMAGE_START[0], IMAGE_START[1]);
		airsim_client.reset_color("building");
		INTRINSIC << 400, 0, 400, 0, 400, 400, 0, 0, 1;
	}

	// Get current image and pose
	// Input: 
	// Output: Image(cv::Mat), Camera matrix(Pos_pack)
	std::map<std::string, cv::Mat> current_image;
	Pos_Pack current_pos= map_converter.get_pos_pack_from_unreal(MAIN_START,0.f,0.f);
	{
		airsim_client.adjust_pose(current_pos);
		current_image = airsim_client.get_images();
	}

	// SLAM
	// Input: Image(cv::Mat), Camera matrix(cv::Iso)
	// Output: Point cloud, Refined Camera matrix(cv::Iso)
	Point_set current_points;
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
		
		if(SYNTHETIC_POINT_CLOUD)
		{
			std::vector<cv::KeyPoint> keypoints;
			auto orb = cv::ORB::create(3000);
			orb->detect(current_image["rgb"], keypoints, roi_mask);
			cv::drawKeypoints(current_image["rgb"], keypoints, current_image["rgb"]);
			std::vector<Eigen::Vector3f> points;
			std::vector<float> points_z;
			for (auto it = keypoints.begin(); it != keypoints.end(); it++) {
				points.emplace_back(it->pt.x, it->pt.y,1.f);
				points_z.push_back(current_image["depth_planar"].at<float>(it->pt.x, it->pt.y));
			}
			Eigen::MatrixXf point_matrix=Eigen::VectorXf::Map(&points[0],points.size());
			Eigen::VectorXf point_z_matrix =Eigen::VectorXf::Map(&points_z[0], points_z.size());
			point_matrix = INTRINSIC.inverse() * point_matrix;
			point_matrix *= point_z_matrix/point_matrix.col(2);
			for(int i=0;i<point_matrix.rows();++i)
			{
				current_points.insert(Point_3(point_matrix(i, 0), point_matrix(i, 1), point_matrix(i, 2)));
			}
			CGAL::write_ply_point_set(std::ofstream("1.ply"), current_points);
			debug_img(std::vector{ current_image["rgb"] });

		}
	}

	// Post process point cloud
	// Input: Point cloud
	// Output: Clustered Point cloud without ground
	{
		
	}

	// Mapping
	// Input: *
	// Output: 3D bounding box of current frame
	{
		
	}

	// Merging
	// Input: 3D bounding box of current frame and previous frame
	// Output: Total 3D bounding box 
	{

	}

	// Generating trajectory

	// Merging trajectory

	// Shot

	std::string model_path = program.get<std::string>("--model_path");
	CGAL::Point_set_3<Point_3,Vector_3> point_cloud;
	CGAL::read_ply_point_set(std::ifstream(model_path), point_cloud);
	Height_map height_map(point_cloud, 3);
	height_map.save_height_map_png("1.png", 2);
	height_map.save_height_map_tiff("1.tiff");

	// Delete ground planes
	{
		for (int idx = point_cloud.size() - 1; idx >= 0; idx--) {
			if (point_cloud.point(idx).z() < .5)
				point_cloud.remove(idx);
		}

		point_cloud.collect_garbage();
	}

	// Cluster building
	std::size_t nb_clusters;
	std::vector<Building> buildings;
	{
		Point_set::Property_map<int> cluster_map = point_cloud.add_property_map<int>("cluster", -1).first;

		std::vector<std::pair<std::size_t, std::size_t> > adjacencies;
		
		nb_clusters = CGAL::cluster_point_set(point_cloud, cluster_map,
				point_cloud.parameters().neighbor_radius(3.).
				adjacencies(std::back_inserter(adjacencies)));
		buildings.resize(nb_clusters);

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

			Building& current_building = buildings[cluster_id];
			current_building.points.insert(point_cloud.point(idx));
		}
	}
	
	// Generate trajectories
	// No guarantee for the validation of camera position, check it later
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory;
	{
		for (int id_building = 0; id_building < nb_clusters; ++id_building) {
			buildings[id_building].bounding_box = get_bounding_box(buildings[id_building].points);
			float xmin = buildings[id_building].bounding_box.xmin();
			float ymin = buildings[id_building].bounding_box.ymin();
			float zmin = buildings[id_building].bounding_box.zmin();
			float xmax = buildings[id_building].bounding_box.xmax();
			float ymax = buildings[id_building].bounding_box.ymax();
			float zmax = buildings[id_building].bounding_box.zmax();
			Eigen::Vector3f box_third_points_2(
				(xmin + xmax)/2,
				(ymin + ymax)/2,
				(zmin + zmax)/3*2
			);
			Eigen::Vector3f box_third_points(
				(xmin + xmax)/2,
				(ymin + ymax)/2,
				(zmin + zmax)/3
			);
			

			Eigen::Vector3f cur_pos(xmin - BOUNDS, ymin - BOUNDS, zmax + Z_UP_BOUNDS);
			while (cur_pos.x() <= xmax + BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[0] += STEP;
			}
			while (cur_pos.y() <= ymax + BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[1] += STEP;
			}
			while (cur_pos.x() >= xmin - BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[0] -= STEP;
			}
			while (cur_pos.y() >= ymin - BOUNDS) {
				trajectory.push_back(std::make_pair(
					cur_pos, box_third_points_2 - cur_pos
				));
				if (DOUBLE_FLAG) {
					Eigen::Vector3f cur_pos_copy(cur_pos[0], cur_pos[1], cur_pos[2] / 3);
					trajectory.push_back(std::make_pair(
						cur_pos_copy, box_third_points - cur_pos_copy
					));
				}
				cur_pos[1] -= STEP;
			}
		}
	}
	
	// Check the camera position
	{
		for (int i = 0; i < trajectory.size(); ++i) {
			Eigen::Vector3f position = trajectory[i].first;
			Eigen::Vector3f camera_focus= trajectory[i].first+ trajectory[i].second;
			
			while(height_map.get_height(position.x(), position.y()) + Z_DOWN_BOUND >position.z())
			{
				position[2] += 5;
			}
			Eigen::Vector3f camera_direction = camera_focus - position;
			trajectory[i].second = camera_direction.normalized();
			trajectory[i].first = position;
		}
	}

	
	for (int i = 0; i < trajectory.size(); ++i)
	{
		point_cloud.insert(Point_3(trajectory[i].first[0], trajectory[i].first[1], trajectory[i].first[2]));
	}
	CGAL::write_ply_point_set(std::ofstream("test_point.ply"), point_cloud);
	write_unreal_path(trajectory, "camera_after_transaction.log");
	write_normal_path(trajectory, "camera_normal.log");
	
	return 0;
}
