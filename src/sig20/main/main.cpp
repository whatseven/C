#include <argparse/argparse.hpp>
#include <cpr/cpr.h>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <CGAL/cluster_point_set.h>
#include <CGAL/random_selection.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/point_generators_2.h>
#include <json/reader.h>
#include <algorithm>
#include <regex>

#include "model_tools.h"
#include "intersection_tools.h"
#include "map_util.h"
#include "viz.h"
#include "building.h"
#include "airsim_control.h"
#include "metrics.h"
#include "trajectory.h"
#include "common_util.h"
#include <opencv2/features2d.hpp>
//#include "SLAM/include/vcc_zxm_mslam.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;
typedef CGAL::Polygon_2<K> Polygon_2;

//Path
boost::filesystem::path log_root("log");
const Eigen::Vector3f UNREAL_START(-4000.f, 30000.f, 200.f);
//Camera
const cv::Vec3b BACKGROUND_COLOR(57,181,55);
const cv::Vec3b SKY_COLOR(161, 120, 205);
const int MAX_FEATURES = 100000;

Eigen::Matrix3f INTRINSIC;

MapConverter map_converter;

class Unreal_object_detector
{
public:

	Unreal_object_detector()
	{
		
	}

	void get_bounding_box(std::map<std::string, cv::Mat>& v_img, std::vector<cv::Vec3b>& v_color_map, std::vector<Building>& v_buildings)
	{
		cv::Mat seg = v_img["segmentation"].clone();
		cv::Mat roi_mask(seg.rows, seg.cols, CV_8UC1);
		roi_mask.forEach<cv::uint8_t>(
			[&seg](cv::uint8_t& val, const int* position) {
			if (seg.at<cv::Vec3b>(position[0], position[1]) != BACKGROUND_COLOR)
				val = 255;
			else
				val = 0;
		});
		v_img.insert(std::make_pair("roi_mask", roi_mask));
		std::vector<std::vector<cv::Point>> bboxes_points;

		for (int y = 0; y < seg.rows; ++y)
			for (int x = 0; x < seg.cols; ++x) {
				if (seg.at<cv::Vec3b>(y, x) == BACKGROUND_COLOR)
					continue;
				bool found = false;
				for (int seg_id = 0; seg_id < v_color_map.size(); ++seg_id) {
					if (v_color_map[seg_id] == seg.at<cv::Vec3b>(y, x))
					{
						bboxes_points[seg_id].push_back(cv::Point2f(x, y));
						found = true;
					}
				}
				if(!found)
				{
					v_color_map.push_back(seg.at<cv::Vec3b>(y, x));
					bboxes_points.push_back(std::vector<cv::Point>());
					bboxes_points[v_color_map.size()-1].push_back(cv::Point2f(x, y));
				}
			}

		v_buildings.resize(v_color_map.size());
		
		for (const auto& pixel_points : bboxes_points) {
			if (pixel_points.size() < 20 * 20)
				continue;
			cv::Rect2f rect = cv::boundingRect(pixel_points);
			cv::rectangle(seg, rect, cv::Scalar(0, 0, 255));
			size_t id = &pixel_points - &*bboxes_points.begin();
			v_buildings[id].bounding_box_2d = CGAL::Bbox_2(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
			v_buildings[id].segmentation_color = v_color_map[id];
		}
		//debug_img(std::vector{ seg });
	}
};

class Real_object_detector {
public:
	Real_object_detector() 
	{

	}

	std::vector<cv::Rect2f> process_results(std::string input, int cols, int rows)
	{
		std::string now_string = input;
		std::vector<std::string> labels;
		std::string::size_type position = now_string.find("], ");
		while (position != now_string.npos)
		{
			labels.push_back(now_string.substr(0, position + 3));
			now_string = now_string.substr(position + 3);
			position = now_string.find("], ");
		}
		if (now_string.length() > 10)
			labels.push_back(now_string);
		std::vector<cv::Rect2f> result;
		for (auto label : labels)
		{
			now_string = label;
			cv::Rect2f box;
			std::string::size_type position1 = now_string.find_first_of("[");
			std::string::size_type position2 = now_string.find_first_of(",");
			float xmin = atof((now_string.substr(position1 + 1, position2 - position1 - 1)).c_str());
			now_string = now_string.substr(position2 + 2);
			position2 = now_string.find_first_of(",");
			float ymin = atof((now_string.substr(0, position2 - 1)).c_str());
			now_string = now_string.substr(position2 + 2);
			position2 = now_string.find_first_of(",");
			float xmax = atof((now_string.substr(0, position2 - 1)).c_str());
			now_string = now_string.substr(position2 + 2);
			position2 = now_string.find_first_of("]");
			float ymax = atof((now_string.substr(0, position2 - 1)).c_str());
			if (xmin < 0 && ymin < 0 && xmax > cols && ymax > rows)
				continue;
			if (xmin < 0)
				xmin = 0;
			if (ymin < 0)
				ymin = 0;
			if (xmax > cols)
				xmax = cols;
			if (ymax > rows)
				ymax = rows;
			box.x = xmin;
			box.y = ymin;
			box.width = xmax - xmin;
			box.height = ymax - ymin;
			result.push_back(box);
		}
		return result;
	}

	cv::Vec3b stringToVec3b(std::string input)
	{
		std::string now_string = input;
		std::vector<std::string> color;
		color.push_back(now_string.substr(0, now_string.find_first_of(" ")));
		now_string = now_string.substr(now_string.find_first_of(" ") + 1);
		color.push_back(now_string.substr(0, now_string.find_first_of(" ")));
		now_string = now_string.substr(now_string.find_first_of(" ") + 1);
		color.push_back(now_string);

		return cv::Vec3b(uchar(atoi(color[0].c_str())), uchar(atoi(color[1].c_str())), uchar(atoi(color[2].c_str())));
	}

	std::string Vec3bToString(cv::Vec3b color)
	{
		return std::to_string(color.val[0]) + " " + std::to_string(color.val[1]) + " " + std::to_string(color.val[2]);
	}

	void get_bounding_box(std::map<std::string, cv::Mat>& v_img, std::vector<cv::Vec3b>& v_color_map, std::vector<Building>& v_buildings)
	{
		std::vector<cv::Rect2f> boxes;
		cv::Mat img = v_img["rgb"];
		std::vector<uchar> data(img.ptr(), img.ptr() + img.size().width * img.size().height * img.channels());
		std::string s(data.begin(), data.end());

		/*auto r = cpr::Get(cpr::Url{ "http://172.31.224.4:10000/index" },
			cpr::Parameters{ {"img",s} });*/
		auto r = cpr::Post(cpr::Url{ "http://172.31.224.4:10000/index" },
			cpr::Body{ s },
			cpr::Header{ {"Content-Type", "text/plain"} });
		std::cout << r.text << std::endl;
		boxes = process_results(r.text, img.cols, img.rows);
		// Updata color map
		for (auto box : boxes)
		{
			// Calculate the main color
			std::map<std::string, int> color_num;
			cv::Rect rect(box.x, box.y, box.width, box.height);
			cv::Mat img_roi = v_img["segmentation"](rect);
			for (int y = 0; y < img_roi.rows; y++)
			{
				for (int x = 0; x < img_roi.cols; x++)
				{
					cv::Vec3b color = img_roi.at<cv::Vec3b>(y, x);
					std::string color_string = Vec3bToString(color);
					auto find_result = color_num.find(color_string);
					if (find_result == color_num.end())
						color_num.insert(std::make_pair(color_string, 1));
					else
						color_num[color_string] += 1;
				}
			}
			cv::Vec3b current_color;
			int max_num = 0;
			for (auto color : color_num)
			{
				if (color.second > max_num)
				{
					max_num = color.second;
					current_color = stringToVec3b(color.first);
				}
			}
			Building current_building;
			current_building.bounding_box_2d = CGAL::Bbox_2(box.x, box.y, box.x + box.width, box.y + box.height);
			current_building.segmentation_color = current_color;
			v_buildings.resize(v_color_map.size());
			auto found = std::find(v_color_map.begin(), v_color_map.end(), current_color);
			if (found == v_color_map.end())
			{
				v_color_map.push_back(current_color);
				v_buildings.push_back(current_building);
			}
				
		}
		return;
	}

};

class Synthetic_SLAM {
public:

	Synthetic_SLAM() {

	}

	void get_points(const std::map<std::string, cv::Mat>& v_img,const std::vector<cv::Vec3b>& v_color_map, std::vector<Building>& v_buildings) {
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat rgb = v_img.at("rgb").clone();
		auto orb = cv::ORB::create(200);
		//orb->detect(rgb, keypoints, v_img.at("roi_mask"));
		orb->detect(rgb, keypoints);
		cv::drawKeypoints(rgb, keypoints, rgb);
		for (auto it = keypoints.begin(); it != keypoints.end(); it++) {
			cv::Vec3b point_color = v_img.at("segmentation").at<cv::Vec3b>(it->pt.y, it->pt.x);
			if (point_color == BACKGROUND_COLOR)
				continue;
			Eigen::Vector3f point(it->pt.x, it->pt.y, 1.f);
			point = INTRINSIC.inverse() * point;
			point *= v_img.at("depth_planar").at<float>(it->pt.y, it->pt.x) / point[2];

			auto find_result = std::find(v_color_map.begin(), v_color_map.end(), point_color);
			if (find_result == v_color_map.end()) {
				LOG(INFO) << "It's not a building.";
				//throw "";
			}
			else {
				v_buildings[&*find_result - &v_color_map[0]].points_camera_space.insert(Point_3(point(0), point(1), point(2)));
			}
		}
		//for (const auto& item : v_buildings)
			//CGAL::write_ply_point_set(std::ofstream(std::to_string(&item - &v_buildings[0]) + "_camera.ply"), item.points_camera_space);
		//debug_img(std::vector{ rgb });
	}
};

enum Motion_status { initialization,exploration,reconstruction,done, final_check,reconstruction_in_exploration};

class Next_best_target {
public:
	Motion_status m_motion_status;
	int m_current_building_id = -1;

	float DISTANCE_THRESHOLD;
	//float DISTANCE_THRESHOLD = 30.f;
	std::vector<CGAL::Point_2<K>> sample_points;
	std::vector<cv::Vec3b> region_status;
	std::vector<cv::Vec3b> region_viz_color;

	Eigen::Vector3f m_map_start;
	Eigen::Vector3f m_map_end;

	Polygon_2 m_boundary;
	
	Next_best_target(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,float v_ccpp_cell_distance):m_map_start(v_map_start_mesh), m_map_end(v_map_end_mesh)
	{
		DISTANCE_THRESHOLD = v_ccpp_cell_distance;
		m_motion_status = Motion_status::initialization;
		for (float y = v_map_start_mesh.y(); y < v_map_end_mesh.y(); y += DISTANCE_THRESHOLD)
			for (float x = v_map_start_mesh.x(); x < v_map_end_mesh.x(); x += DISTANCE_THRESHOLD)
				sample_points.push_back(CGAL::Point_2<K>(x, y));
		region_viz_color = get_color_table_bgr();
		region_status.resize(sample_points.size(), region_viz_color[0]);
	}

	virtual void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) = 0;

	virtual void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) = 0;
	
	virtual std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration,float v_threshold) = 0;
};

class Next_best_target_min_distance_ccpp :public Next_best_target {
public:
	int m_current_exploration_id = -1;
	cv::Vec3b color_unobserved,color_free,color_occupied;

	std::vector<bool> already_explored;

	std::queue<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_exploration_point;
	
	Next_best_target_min_distance_ccpp(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh, float v_ccpp_cell_distance)
		:Next_best_target(v_map_start_mesh, v_map_end_mesh,v_ccpp_cell_distance) {
		color_unobserved = region_viz_color[0];
		color_free = region_viz_color[1];
		color_occupied = region_viz_color[2];
		already_explored.resize(region_status.size(), false);
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		// Find next building in current cell
		std::vector<Next_target> untraveled_buildings;
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].passed_trajectory.size() == 0 
				//&& (v_buildings[i_building].bounding_box_3d.center().block(0,0,2,1)-Eigen::Vector2f(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y())).norm()<DISTANCE_THRESHOLD/2
				)
				untraveled_buildings.emplace_back(i_building, -1);
		}
		if(untraveled_buildings.size()==0&& with_exploration)
		{
			for (int i_point = 0; i_point < sample_points.size(); ++i_point)
				if (!already_explored[i_point])
					//if (region_status[i_point]==color_unobserved)
					untraveled_buildings.emplace_back(-1, i_point);
		}

		if (untraveled_buildings.size() == 0) {
			m_motion_status = Motion_status::done;
			return;
		}
		
		int next_target_id = std::min_element(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
			float distance1, distance2;
			if (b1.origin_index_in_building_vector != -1)
			{
				int id_trajectory = v_buildings[b1.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
				Eigen::Vector3f pos = v_buildings[b1.origin_index_in_building_vector].trajectory[id_trajectory].first;
				Eigen::Vector2f pos_2(pos.x(), pos.y());
				distance1 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			}
			else
			{
				distance1 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b1.origin_index_in_untraveled_pointset].x(), sample_points[b1.origin_index_in_untraveled_pointset].y())).norm();
			}
			if (b2.origin_index_in_building_vector != -1)
			{
				int id_trajectory = v_buildings[b2.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
				Eigen::Vector3f pos = v_buildings[b2.origin_index_in_building_vector].trajectory[id_trajectory].first;
				Eigen::Vector2f pos_2(pos.x(), pos.y());
				distance2 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			}
			else
				distance2 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b2.origin_index_in_untraveled_pointset].x(), sample_points[b2.origin_index_in_untraveled_pointset].y())).norm();

			return  distance1 < distance2;
		})- untraveled_buildings.begin();

		if (!with_exploration) {
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[0].origin_index_in_building_vector;
			return;
		}

		if (untraveled_buildings[next_target_id].origin_index_in_building_vector == -1) {
			m_motion_status = Motion_status::exploration;
			m_current_exploration_id = untraveled_buildings[next_target_id].origin_index_in_untraveled_pointset;

			/*m_exploration_point.emplace(
				Eigen::Vector3f(
					sample_points[m_current_exploration_id].x() - DISTANCE_THRESHOLD * 0.3,
					sample_points[m_current_exploration_id].y(),
					100),
				Eigen::Vector3f(0, 1, -std::tan(64.f / 180 * M_PI)).normalized()
			);
			m_exploration_point.emplace(
				Eigen::Vector3f(
					sample_points[m_current_exploration_id].x() - DISTANCE_THRESHOLD * 0.1,
					sample_points[m_current_exploration_id].y(),
					100),
				Eigen::Vector3f(1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
			);
			m_exploration_point.emplace(
				Eigen::Vector3f(
					sample_points[m_current_exploration_id].x() + DISTANCE_THRESHOLD * 0.1,
					sample_points[m_current_exploration_id].y(),
					100),
				Eigen::Vector3f(0, -1, -std::tan(64.f / 180 * M_PI)).normalized()
			);
			m_exploration_point.emplace(
				Eigen::Vector3f(
					sample_points[m_current_exploration_id].x() + DISTANCE_THRESHOLD * 0.3,
					sample_points[m_current_exploration_id].y(),
					100),
				Eigen::Vector3f(-1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
			);*/
			m_exploration_point.emplace(
				Eigen::Vector3f(
					sample_points[m_current_exploration_id].x(),
					sample_points[m_current_exploration_id].y(),
					100),
				Eigen::Vector3f(-1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
			);
			
		}
		else {
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
		}
		return;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		}) - sample_points.begin();
		region_status[nearest_region_id] = color_free;
		if (m_motion_status == Motion_status::exploration)
			already_explored[nearest_region_id] = true;

		for (int i_point = 0; i_point < region_status.size(); i_point++) {
			if (region_status[i_point] != color_unobserved)
				continue;
			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

			for (const auto& item_building : v_buildings) {
				Eigen::Vector3f point(p.x(), p.y(), 0.f);
				if (item_building.bounding_box_3d.inside_2d(point)) {
					region_status[i_point] = color_occupied;
					break;
				}
			}
			continue;
		}
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status == Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if (m_motion_status == Motion_status::exploration) {
			if(m_exploration_point.size() > 0)
			{
				auto item = m_exploration_point.front();
				m_exploration_point.pop();
				return item;
			}
			else
			{
				already_explored[m_current_exploration_id] = true;
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			
		}
 		if (m_motion_status == Motion_status::reconstruction) {
			const int& cur_building_id = m_current_building_id;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
				std::back_inserter(unpassed_trajectory),
				[&passed_trajectory, v_threshold](const auto& item_new_trajectory) {
				bool untraveled = true;
				for (const auto& item_passed_trajectory : passed_trajectory)
					if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold) {
						untraveled = false;
					}
				return untraveled;
			});

			if (unpassed_trajectory.size() == 0) {
				//debug_img(std::vector<cv::Mat>{cv::Mat(1, 1, CV_8UC1)});
				//m_motion_status = Motion_status::exploration;
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				LOG(INFO) << "Change target !";
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				auto it_min_distance = std::min_element(
					unpassed_trajectory.begin(), unpassed_trajectory.end(),
					[&v_cur_pos](const auto& t1, const auto& t2) {
					return (t1.first - v_cur_pos.pos_mesh).norm() < (t2.first - v_cur_pos.pos_mesh).norm();
				});
				next_pos = *it_min_distance;
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}
		}
	}

};

class Next_best_target_reconstruction_only :public Next_best_target {
public:
	int m_current_exploration_id = -1;
	cv::Vec3b color_unobserved, color_free, color_occupied;

	Next_best_target_reconstruction_only(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,float v_ccpp_cell_distance)
		:Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance) {
		color_unobserved = region_viz_color[0];
		color_free = region_viz_color[1];
		color_occupied = region_viz_color[2];
		m_current_building_id = 0;
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		// Find with distance
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].passed_trajectory.size() == 0)
				untraveled_buildings.emplace_back(i_building, -1);
		}

		if (untraveled_buildings.size() == 0) {
			m_motion_status = Motion_status::done;
			return;
		}

		int next_target_id = std::min_element(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
			float distance1, distance2;
			int id_trajectory = v_buildings[b1.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
			Eigen::Vector3f pos = v_buildings[b1.origin_index_in_building_vector].trajectory[id_trajectory].first;
			Eigen::Vector2f pos_2(pos.x(), pos.y());
			distance1 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			id_trajectory = v_buildings[b2.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
			pos = v_buildings[b2.origin_index_in_building_vector].trajectory[id_trajectory].first;
			pos_2 = Eigen::Vector2f(pos.x(), pos.y());
			distance2 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			return  distance1 < distance2;
		}) - untraveled_buildings.begin();

		m_motion_status = Motion_status::reconstruction;
		m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
		return;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[m_current_building_id].passed_trajectory;

		if (passed_trajectory.size() == v_buildings[m_current_building_id].trajectory.size()) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Change target !";
			if (m_motion_status == Motion_status::done)
				return next_pos;
			return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
		}
		else {
			next_pos = v_buildings[m_current_building_id].trajectory[passed_trajectory.size()];
			passed_trajectory.push_back(next_pos);
			return next_pos;
		}
	}

};

class Next_best_target_exploration_only :public Next_best_target {
public:
	int m_current_exploration_id = 0;
	cv::Vec3b color_unobserved, color_free, color_occupied;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_ccpp_trajectory;

	Next_best_target_exploration_only(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh, float v_ccpp_cell_distance)
		:Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance) {
		color_occupied = region_viz_color[0];
		color_unobserved = cv::Vec3b(0, 0, 0);
		m_current_building_id = -1;
		m_motion_status = Motion_status::exploration;
		int num = region_status.size();
		region_status.clear();
		region_status.resize(num, color_unobserved);
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		return;
	}
	
	bool get_ccpp_trajectory(const Eigen::Vector3f& v_cur_pos) {
		cv::Mat ccpp_map((m_map_end.y() - m_map_start.y()) / DISTANCE_THRESHOLD,
			(m_map_end.x() - m_map_start.x()) / DISTANCE_THRESHOLD,
			CV_8UC1, cv::Scalar(255));
		Eigen::Vector3f t1 = (v_cur_pos - m_map_start) / DISTANCE_THRESHOLD;
		Eigen::Vector3f t2 = (m_map_end - m_map_start) / DISTANCE_THRESHOLD;
		Eigen::Vector2i start_pos_on_map(t1.x(), t1.y());
		Eigen::Vector2i end_pos_on_map(t2.x(), t2.y());
		cv::Mat start_end = ccpp_map.clone();
		start_end.at<cv::uint8_t>(start_pos_on_map.y(), start_pos_on_map.x()) = 255;
		start_end.at<cv::uint8_t>(end_pos_on_map.y(), end_pos_on_map.x()) = 255;

		std::vector<Eigen::Vector2i> map_trajectory = perform_ccpp(start_end,
			start_pos_on_map, end_pos_on_map);
		cv::Mat viz_ccpp = ccpp_map.clone();

		float iter_trajectory = 0;
		for (const Eigen::Vector2i& item : map_trajectory) {
			// todo: invert x,y!!!!!!!!!!!!!
			viz_ccpp.at<cv::uint8_t>(item.x(), item.y()) = iter_trajectory++ * 255. / map_trajectory.size();
			Eigen::Vector3f t3 = m_map_start + DISTANCE_THRESHOLD * Eigen::Vector3f(item.y(), item.x(), 0.f);
			t3.z() = 100;
			m_ccpp_trajectory.emplace_back(
				t3,
				Eigen::Vector3f(0, 0, -1)
			);
		}
		//Eigen::Vector3f next_point(m_map_end.x()-60, m_map_end.y() - 60,100);
		//m_ccpp_trajectory.emplace_back(
		//	next_point,
		//	Eigen::Vector3f(0, 0, -1)
		//);
		return true;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		}) - sample_points.begin();
		if (m_motion_status == Motion_status::exploration)
			region_status[nearest_region_id] = color_occupied;
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status == Motion_status::exploration) {
			if (m_ccpp_trajectory.size() == 0)
				get_ccpp_trajectory(v_cur_pos.pos_mesh);
			next_pos = m_ccpp_trajectory[m_current_exploration_id];
			m_current_exploration_id += 1;
			if (m_current_exploration_id == m_ccpp_trajectory.size())
			{
				std::vector<Next_target> untraveled_buildings;
				for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
					if (v_buildings[i_building].passed_trajectory.size() == 0)
						untraveled_buildings.emplace_back(i_building, -1);
				}

				if (untraveled_buildings.size() == 0) {
					m_motion_status = Motion_status::done;
					return next_pos;
				}

				int next_target_id = std::min_element(untraveled_buildings.begin(),
					untraveled_buildings.end(),
					[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
					float distance1, distance2;
					int id_trajectory = v_buildings[b1.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
					Eigen::Vector3f pos = v_buildings[b1.origin_index_in_building_vector].trajectory[id_trajectory].first;
					Eigen::Vector2f pos_2(pos.x(), pos.y());
					distance1 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
					id_trajectory = v_buildings[b2.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
					pos = v_buildings[b2.origin_index_in_building_vector].trajectory[id_trajectory].first;
					pos_2 = Eigen::Vector2f(pos.x(), pos.y());
					distance2 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
					return  distance1 < distance2;
				}) - untraveled_buildings.begin();

				m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
				m_motion_status = Motion_status::reconstruction;
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
		}
		else {
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[m_current_building_id].passed_trajectory;

			if (passed_trajectory.size() == v_buildings[m_current_building_id].trajectory.size()) {
				std::vector<Next_target> untraveled_buildings;
				for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
					if (v_buildings[i_building].passed_trajectory.size() == 0)
						untraveled_buildings.emplace_back(i_building, -1);
				}

				if (untraveled_buildings.size() == 0) {
					m_motion_status = Motion_status::done;
					return next_pos;
				}

				int next_target_id = std::min_element(untraveled_buildings.begin(),
					untraveled_buildings.end(),
					[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
					float distance1, distance2;
					int id_trajectory = v_buildings[b1.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
					Eigen::Vector3f pos = v_buildings[b1.origin_index_in_building_vector].trajectory[id_trajectory].first;
					Eigen::Vector2f pos_2(pos.x(), pos.y());
					distance1 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
					id_trajectory = v_buildings[b2.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
					pos = v_buildings[b2.origin_index_in_building_vector].trajectory[id_trajectory].first;
					pos_2 = Eigen::Vector2f(pos.x(), pos.y());
					distance2 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
					return  distance1 < distance2;
				}) - untraveled_buildings.begin();

				m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;

				if (m_motion_status == Motion_status::done)
					return next_pos;
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				next_pos = v_buildings[m_current_building_id].trajectory[passed_trajectory.size()];
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}
		}
		return next_pos;
	}

};

class Next_best_target_random_min_distance:public Next_best_target
{
public:
	int m_current_exploration_id = -1;
	cv::Vec3b color_unobserved, color_free, color_occupied;
	std::vector<bool> already_explored;

	Next_best_target_random_min_distance(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,float v_ccpp_cell_distance)
		:Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance) {
		//color_unobserved = region_viz_color[0];
		color_free = region_viz_color[1];
		color_occupied = region_viz_color[2];
		color_unobserved = cv::Vec3b(0, 0, 0);

		int num = region_status.size();
		region_status.clear();
		region_status.resize(num, color_unobserved);
		
		already_explored.resize(region_status.size(), false);
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings,bool with_exploration) override
	{
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		// Find with distance
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].passed_trajectory.size() == 0)
				untraveled_buildings.emplace_back(i_building, -1);
		}
		if (with_exploration)
			for (int i_point = 0; i_point < sample_points.size(); ++i_point)
				//if (region_status[i_point] == color_unobserved)
				if (already_explored[i_point] == false)
					untraveled_buildings.emplace_back(-1, i_point);

		if (untraveled_buildings.size() == 0) {
			m_motion_status = Motion_status::done;
			return;
		}

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(untraveled_buildings.begin(), untraveled_buildings.end(),g);
		int next_target_id = std::min_element(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
			float distance1, distance2;
			if (b1.origin_index_in_building_vector != -1) {
				distance1 = (v_buildings[b1.origin_index_in_building_vector].bounding_box_3d.box.center().block(0,0,2,1) - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			}
			else {
				distance1 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b1.origin_index_in_untraveled_pointset].x(), sample_points[b1.origin_index_in_untraveled_pointset].y())).norm();
			}
			if (b2.origin_index_in_building_vector != -1) {
				distance2 = (v_buildings[b2.origin_index_in_building_vector].bounding_box_3d.box.center().block(0, 0, 2, 1) - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			}
			else
				distance2 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b2.origin_index_in_untraveled_pointset].x(), sample_points[b2.origin_index_in_untraveled_pointset].y())).norm();

			return  distance1 < distance2;
		}) - untraveled_buildings.begin();

		if (!with_exploration) {
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[0].origin_index_in_building_vector;
			return;
		}

		if (untraveled_buildings[next_target_id].origin_index_in_building_vector == -1) {
			m_motion_status = Motion_status::exploration;
			m_current_exploration_id = untraveled_buildings[next_target_id].origin_index_in_untraveled_pointset;
		}
		else {
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
		}
		return;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override
	{
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id=std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		})- sample_points.begin();
		if (m_motion_status == Motion_status::exploration)
		{
			region_status[nearest_region_id] = region_viz_color[1];
			already_explored[nearest_region_id] = true;
		}
		
		for (int i_point = 0; i_point < region_status.size(); i_point++) {
			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

			for (const auto& item_building : v_buildings) {
				Eigen::Vector3f point(p.x(), p.y(), 0.f);
				if (item_building.bounding_box_3d.inside_2d(point)) {
					region_status[i_point] = color_occupied;
					break;
				}
			}
		}
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status == Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if (m_motion_status == Motion_status::exploration) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			Eigen::Vector3f next_pos(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y(), 100);
			return std::make_pair(next_pos, (next_pos- v_cur_pos.pos_mesh).normalized());
		}
		if (m_motion_status == Motion_status::reconstruction) {
			const int& cur_building_id = m_current_building_id;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
				std::back_inserter(unpassed_trajectory),
				[&passed_trajectory, v_threshold](const auto& item_new_trajectory) {
				bool untraveled = true;
				for (const auto& item_passed_trajectory : passed_trajectory)
					if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold) {
						untraveled = false;
					}
				return untraveled;
			});

			if (unpassed_trajectory.size() == 0) {
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				LOG(INFO) << "Change target !";
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				auto it_min_distance = std::min_element(
					unpassed_trajectory.begin(), unpassed_trajectory.end(),
					[&v_cur_pos](const auto& t1, const auto& t2) {
					return (t1.first - v_cur_pos.pos_mesh).norm() < (t2.first - v_cur_pos.pos_mesh).norm();
				});
				next_pos = *it_min_distance;
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}
		}
	}
	
};

class Next_best_target_first_building_next_region :public Next_best_target {
public:
	int m_current_exploration_id = -1;
	cv::Vec3b color_unobserved, color_free, color_occupied;
	std::vector<bool> already_explored;

	Next_best_target_first_building_next_region(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,float v_ccpp_cell_distance)
		:Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance) {
		color_unobserved = region_viz_color[0];
		color_free = region_viz_color[1];
		color_occupied = region_viz_color[2];
		already_explored.resize(region_status.size(), false);
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		// Find with distance
		if(m_motion_status==Motion_status::reconstruction)
		{
			for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
				if (v_buildings[i_building].passed_trajectory.size() == 0)
					untraveled_buildings.emplace_back(i_building, -1);
			}
			if (untraveled_buildings.size() == 0) {
				m_motion_status = Motion_status::exploration;
			}
		}
		if (m_motion_status == Motion_status::exploration)
		{
			for (int i_point = 0; i_point < sample_points.size(); ++i_point)
				//if (region_status[i_point] == color_unobserved)
				if (already_explored[i_point] == false)
					untraveled_buildings.emplace_back(-1, i_point);
			if (untraveled_buildings.size() == 0) {
				m_motion_status = Motion_status::done;
				return;
			}
		}

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(untraveled_buildings.begin(), untraveled_buildings.end(), g);
		int next_target_id = std::min_element(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
			float distance1, distance2;
			if (b1.origin_index_in_building_vector != -1) {
				distance1 = (v_buildings[b1.origin_index_in_building_vector].bounding_box_3d.box.center().block(0, 0, 2, 1) - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			}
			else {
				distance1 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b1.origin_index_in_untraveled_pointset].x(), sample_points[b1.origin_index_in_untraveled_pointset].y())).norm();
			}
			if (b2.origin_index_in_building_vector != -1) {
				distance2 = (v_buildings[b2.origin_index_in_building_vector].bounding_box_3d.box.center().block(0, 0, 2, 1) - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
			}
			else
				distance2 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b2.origin_index_in_untraveled_pointset].x(), sample_points[b2.origin_index_in_untraveled_pointset].y())).norm();

			return  distance1 < distance2;
		}) - untraveled_buildings.begin();

		if (untraveled_buildings[next_target_id].origin_index_in_building_vector == -1) {
			m_current_exploration_id = untraveled_buildings[next_target_id].origin_index_in_untraveled_pointset;
		}
		else {
			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
		}
		return;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		}) - sample_points.begin();
		//region_status[nearest_region_id] = color_free;
		if (m_motion_status == Motion_status::exploration|| m_motion_status == Motion_status::initialization)
		{
			region_status[nearest_region_id] = color_free;
			already_explored[nearest_region_id] = true;
		}

		for (int i_point = 0; i_point < region_status.size(); i_point++) {
			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

			for (const auto& item_building : v_buildings) {
				Eigen::Vector3f point(p.x(), p.y(), 0.f);
				if (item_building.bounding_box_3d.inside_2d(point)) {
					//region_status[i_point] = color_occupied;
					break;
				}
			}
		}
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status == Motion_status::initialization) {
			m_motion_status = Motion_status::reconstruction;
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if (m_motion_status == Motion_status::exploration) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			Eigen::Vector3f next_pos(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y(), 100);
			return std::make_pair(next_pos, (next_pos - v_cur_pos.pos_mesh).normalized());
		}
		if (m_motion_status == Motion_status::reconstruction) {
			const int& cur_building_id = m_current_building_id;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
				std::back_inserter(unpassed_trajectory),
				[&passed_trajectory, v_threshold](const auto& item_new_trajectory) {
				bool untraveled = true;
				for (const auto& item_passed_trajectory : passed_trajectory)
					if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold) {
						untraveled = false;
					}
				return untraveled;
			});

			if (unpassed_trajectory.size() == 0) {
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				LOG(INFO) << "Change target !";
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				auto it_min_distance = std::min_element(
					unpassed_trajectory.begin(), unpassed_trajectory.end(),
					[&v_cur_pos](const auto& t1, const auto& t2) {
					return (t1.first - v_cur_pos.pos_mesh).norm() < (t2.first - v_cur_pos.pos_mesh).norm();
				});
				next_pos = *it_min_distance;
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}
		}
	}

};

class Next_best_target_topology_exploration: public Next_best_target {
public:
	std::vector<Eigen::AlignedBox2f> topology;
	//std::vector<cv::Vec3b> topology_viz_color;
	cv::Vec3b color_occupied;
	cv::Vec3b color_unobserved;
	cv::Vec3b color_reconstruction;
	Json::Value m_arg;

	//const int CCPP_CELL_THRESHOLD = 50;
	//const int CCPP_CELL_THRESHOLD = 70;
	int CCPP_CELL_THRESHOLD;
	int rotation_status = 0;
	//const int CCPP_CELL_THRESHOLD = 10;

	int m_current_ccpp_trajectory_id;
	int m_current_color_id;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_ccpp_trajectory;
	std::queue<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_exploration_point;

	float memory_y = -1.f;
	int dummy1 = 0;
	int dummy2 = 0;
	float dummy3 = 0;

	Next_best_target_topology_exploration(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,
		int v_CCPP_CELL_THRESHOLD,const Polygon_2& m_boundary,float v_ccpp_cell_distance, const Json::Value& v_arg):CCPP_CELL_THRESHOLD(v_CCPP_CELL_THRESHOLD), m_arg(v_arg),
		Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance)
	{
		//color_reconstruction = region_viz_color[2];
		//color_occupied = region_viz_color[1];
		//color_unobserved = region_viz_color[0];
		color_reconstruction = cv::Vec3b(0, 255, 0);
		color_occupied = cv::Vec3b(0, 255, 0);
		color_unobserved = cv::Vec3b(205, 205, 209);
		
		region_status.clear();
		region_status.resize(sample_points.size(), color_unobserved);

		if(m_boundary.size()!=0){
			for (int i_sample_point = 0; i_sample_point < sample_points.size(); ++i_sample_point) {
				if (m_boundary.bounded_side(sample_points[i_sample_point]) != CGAL::ON_BOUNDED_SIDE) {
					//LOG(INFO) << m_boundary.is_simple();
					region_status[i_sample_point] = color_occupied;
				}
			}
		}
		
		m_current_color_id = 0;
		region_status[0] = region_viz_color[m_current_color_id];
		//topology.emplace_back(Eigen::Vector2f(m_map_start.x(), m_map_start.y()), 
		//	Eigen::Vector2f(m_map_start.x()+1, m_map_start.y()+1));
		//topology_viz_color= get_color_table_bgr();
	}

	bool get_ccpp_trajectory(const Eigen::Vector3f& v_cur_pos, const Building& v_building,int v_ccpp_threshold)
	{
		
		const Eigen::AlignedBox3f& cur_box_3 = v_building.bounding_box_3d.box;
		

		// Ensure even
		Eigen::AlignedBox2f last_topology;
		Eigen::Vector2i last_map_pos;
		int x_add = 0, y_add = 0;
		if (topology.size() == 0)
			last_map_pos = Eigen::Vector2i(0, 0);
		else
		{
			last_topology = topology.at(topology.size() - 1);
			last_map_pos = Eigen::Vector2i(int((last_topology.max().x() - m_map_start.x()) / DISTANCE_THRESHOLD + 1), int((last_topology.max().y() - m_map_start.y()) / DISTANCE_THRESHOLD + 1));
		}
		Eigen::Vector2i now_map_pos(int((cur_box_3.max().x() - m_map_start.x()) / DISTANCE_THRESHOLD + 1), int((cur_box_3.max().y() - m_map_start.y()) / DISTANCE_THRESHOLD + 1));
		if ((now_map_pos.x() - last_map_pos.x()) % 2 == 1)
			x_add = 1;
		if ((now_map_pos.y() - last_map_pos.y()) % 2 == 1)
			y_add = 1;

		Eigen::AlignedBox2f cur_box_2(Eigen::Vector2f(m_map_start.x(), m_map_start.y()),
			Eigen::Vector2f(cur_box_3.max().x() + DISTANCE_THRESHOLD * x_add, cur_box_3.max().y() + DISTANCE_THRESHOLD * y_add));

		Eigen::Vector3f next_point = v_building.bounding_box_3d.box.max();
		if(m_motion_status != Motion_status::final_check)
		{
			//cur_box_2.max().x() += 2*DISTANCE_THRESHOLD;
			//cur_box_2.max().y() += 2 * DISTANCE_THRESHOLD;
			//next_point.x() += 2 * DISTANCE_THRESHOLD;
			//next_point.y() += 2 * DISTANCE_THRESHOLD;
		}
		
		cv::Mat ccpp_map((m_map_end.y() - m_map_start.y()) / DISTANCE_THRESHOLD + 1,
			(m_map_end.x() - m_map_start.x()) / DISTANCE_THRESHOLD + 1,
			CV_8UC1, cv::Scalar(0));
		int num_ccpp_cell = 0;
		for (int id_region = 0; id_region < sample_points.size(); ++id_region) {
			const auto& point = sample_points[id_region];
			int y = (point.y() - m_map_start.y()) / DISTANCE_THRESHOLD;
			int x = (point.x() - m_map_start.x()) / DISTANCE_THRESHOLD;
			Eigen::Vector2f pos(point.x(), point.y());
			if (inside_box(pos, cur_box_2) && region_status[id_region]==color_unobserved) {
			//if (inside_box(pos, cur_box_2)) {
				bool already_traveled = false;
				for (const auto& item : topology) {
					if (inside_box(pos, item))
						already_traveled=true;
				}
				if(!already_traveled)
				{
					ccpp_map.at<cv::uint8_t>(y, x) = 255;
					num_ccpp_cell += 1;
				}
				
			}
		}
		if (num_ccpp_cell < v_ccpp_threshold || num_ccpp_cell==0)
			return false;
		dummy2 += num_ccpp_cell;
		topology.push_back(cur_box_2);
		//cv::imwrite(std::to_string(dummy1++)+"_ccpp_map.png", ccpp_map);
		// Find the nearest view point
		Eigen::Vector3f t1 = (v_cur_pos - m_map_start) / DISTANCE_THRESHOLD;

		Eigen::Vector3f t2 = (next_point - m_map_start) / DISTANCE_THRESHOLD;

		Eigen::Vector2i start_pos_on_map(t1.x(), t1.y());
		Eigen::Vector2i end_pos_on_map(t2.x(), t2.y());
		if (v_ccpp_threshold == 0)//Last check
			end_pos_on_map = start_pos_on_map;

		cv::Mat start_end = ccpp_map.clone();
		//start_end.setTo(0);
		start_end.at<cv::uint8_t>(start_pos_on_map.y(), start_pos_on_map.x()) = 255;
		start_end.at<cv::uint8_t>(end_pos_on_map.y(), end_pos_on_map.x()) = 255;
		//debug_img(std::vector<cv::Mat>{ccpp_map, start_end});

		std::vector<Eigen::Vector2i> map_trajectory = perform_ccpp(start_end,
			start_pos_on_map, end_pos_on_map, 2.5);

		//std::cout << "  " << std::endl;
		cv::Mat viz_ccpp = ccpp_map.clone();

		float iter_trajectory = 0;
		for (const Eigen::Vector2i& item : map_trajectory) {
			// todo: invert x,y!!!!!!!!!!!!!
			viz_ccpp.at<cv::uint8_t>(item.x(), item.y()) = iter_trajectory++ * 255. / map_trajectory.size();
			Eigen::Vector3f t3 = m_map_start + DISTANCE_THRESHOLD * Eigen::Vector3f(item.y(), item.x(), 0.f);
			t3.z() = 100;
			m_ccpp_trajectory.emplace_back(
				t3,
				Eigen::Vector3f(0, 0, -1)
			);
			if (iter_trajectory < map_trajectory.size())
				dummy3 += (map_trajectory[iter_trajectory] - map_trajectory[iter_trajectory - 1]).norm();
		}
		next_point.z() = 100;
		m_ccpp_trajectory.emplace_back(
			next_point,
			Eigen::Vector3f(0, 0, -1)
		);
		cv::imwrite("log/ccpp_map/"+std::to_string(dummy1++) + "_ccpp.png", viz_ccpp);
		return true;
	}
	
	void get_next_target(int frame_id,const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override{
		m_ccpp_trajectory.clear();
		m_current_ccpp_trajectory_id = 0;
		// Find next target (building or place) with higher confidence
		std::vector<int> untraveled_buildings;
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].is_divide)
				continue;
			if (v_buildings[i_building].passed_trajectory.size() == 0)
			{
				untraveled_buildings.push_back(i_building);
			}
		}
		if (untraveled_buildings.size() == 0)
		{
			m_motion_status = Motion_status::final_check;
			Building fake_building;
			fake_building.bounding_box_3d = Eigen::AlignedBox3f(m_map_end-Eigen::Vector3f(1,1,1), m_map_end);
			fake_building.trajectory.emplace_back(m_map_end, Eigen::Vector3f(0, 0, 1));
			get_ccpp_trajectory(v_cur_pos.pos_mesh, 
				fake_building, 0);
			return;
		}

		// Find nearest building to existing polygon
		std::sort(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const int& b1, const int& b2) {
			int view1 = v_buildings[b1].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
			int view2 = v_buildings[b2].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
			return (v_buildings[b1].trajectory[view1].first- v_cur_pos.pos_mesh).norm() < (v_buildings[b2].trajectory[view2].first - v_cur_pos.pos_mesh).norm();
			//return (v_buildings[id1].trajectory[view1].first.x()- v_cur_pos.pos_mesh.x()) < (v_buildings[id2].trajectory[view2].first.x() - v_cur_pos.pos_mesh.x());
		});

		int  id_building = 0;
		if(with_exploration)
		{
			bool ccpp_done = get_ccpp_trajectory(v_cur_pos.pos_mesh,v_buildings[untraveled_buildings[id_building]],
				CCPP_CELL_THRESHOLD);
			while(!ccpp_done)
			{
				id_building += 1;
				if (id_building >= untraveled_buildings.size())
				{
					m_motion_status = Motion_status::final_check;
					Building fake_building;
					fake_building.bounding_box_3d = Eigen::AlignedBox3f(m_map_end - Eigen::Vector3f(1, 1, 1), m_map_end);
					fake_building.trajectory.emplace_back(m_map_end, Eigen::Vector3f(0, 0, 1));
					get_ccpp_trajectory(v_cur_pos.pos_mesh,
						fake_building, 0);
					return;
				}
	
				else
					ccpp_done = get_ccpp_trajectory(v_cur_pos.pos_mesh, v_buildings[untraveled_buildings[id_building]],
						CCPP_CELL_THRESHOLD);
			}
		}

		// return trajectory
		//debug_img(std::vector<cv::Mat>{map});
		m_current_building_id = untraveled_buildings[id_building];
		if (with_exploration)
			m_motion_status = Motion_status::exploration;
		else
			m_motion_status = Motion_status::reconstruction_in_exploration;
		return;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		}) - sample_points.begin();
		//if (region_status[nearest_region_id] == color_unobserved)
		//{
			//if (m_motion_status == Motion_status::reconstruction)
			//	region_status[nearest_region_id] = color_reconstruction;
			//else
		//		region_status[nearest_region_id] = region_viz_color[m_current_color_id % (region_viz_color.size() - 2) + 2];
		//}
		if ((m_motion_status == Motion_status::exploration|| m_motion_status == Motion_status::final_check )&& region_status[nearest_region_id]==color_unobserved)
		{
			bool inside = false;
			for(auto item:topology)
				if(inside_box(Eigen::Vector2f(sample_points[nearest_region_id].x(), sample_points[nearest_region_id].y()), item))
					inside = true;
			if(inside)
				region_status[nearest_region_id] = region_viz_color[m_current_color_id % region_viz_color.size()];
		}
		
		for (int i_point = 0; i_point < region_status.size(); i_point++) {
			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

			for (const auto& item_building : v_buildings) {
				
				//Eigen::AlignedBox2f box(Eigen::Vector2f(item_building.bounding_box_3d.min().x(), item_building.bounding_box_3d.min().y()),
				//	Eigen::Vector2f(item_building.bounding_box_3d.max().x(), item_building.bounding_box_3d.max().y()));
				//if (point_box_distance_eigen(p, box) < 20 || inside_box(p, box)) {
				//if (inside_box(p, box)) {
					//region_status[i_point] = color_occupied;
				//	break;
				//}
			}
		}
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration,float v_threshold) override
	{
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;
		if (m_motion_status==Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}
		if (m_motion_status == Motion_status::exploration || m_motion_status == Motion_status::final_check)
		{
			std::vector<int> untraveled_buildings_inside_exist_region;
			Eigen::Vector2f cur_point_cgal(m_ccpp_trajectory[m_current_ccpp_trajectory_id].first.x(), 
				m_ccpp_trajectory[m_current_ccpp_trajectory_id].first.y());
			for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
				if (v_buildings[i_building].is_divide)
					continue;
				if (v_buildings[i_building].passed_trajectory.size() == 0) {
					int id_nearest_view = v_buildings[i_building].find_nearest_trajectory_2d(Eigen::Vector3f(cur_point_cgal.x(), cur_point_cgal.y(), 0));
					//Eigen::Vector2f nearest_view(v_buildings[i_building].trajectory[id_nearest_view].first.x(), 
					//	v_buildings[i_building].trajectory[id_nearest_view].first.y());
					Eigen::Vector2f nearest_view(v_buildings[i_building].bounding_box_3d.box.center().x(),
						v_buildings[i_building].bounding_box_3d.box.center().y());
					
					if ((nearest_view - cur_point_cgal).norm() < DISTANCE_THRESHOLD / 1.5)
						untraveled_buildings_inside_exist_region.push_back(i_building);
				}
			}
			if (untraveled_buildings_inside_exist_region.size() != 0 && m_arg["with_reconstruction"].asBool()) {
				//throw;
				m_motion_status = Motion_status::reconstruction_in_exploration;
				int id_building = std::min_element(untraveled_buildings_inside_exist_region.begin(),
					untraveled_buildings_inside_exist_region.end(),
					[&v_cur_pos, &v_buildings, this](const int& id1, const int& id2) {
					int view1 = v_buildings[id1].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
					int view2 = v_buildings[id2].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
					return (v_buildings[id1].trajectory[view1].first - v_cur_pos.pos_mesh).norm() < (v_buildings[id2].trajectory[view2].first - v_cur_pos.pos_mesh).norm();
					//return (v_buildings[id1].trajectory[view1].first.x()- v_cur_pos.pos_mesh.x()) < (v_buildings[id2].trajectory[view2].first.x() - v_cur_pos.pos_mesh.x());
				}) - untraveled_buildings_inside_exist_region.begin();
				m_current_building_id = untraveled_buildings_inside_exist_region[id_building];
				next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
				return next_pos;

				//m_motion_status = Motion_status::reconstruction;
			}
			else
			{
				if(m_exploration_point.size() == 0)
				{
					m_current_ccpp_trajectory_id += 1;
					if (m_current_ccpp_trajectory_id >= m_ccpp_trajectory.size()) {
						if (m_motion_status == Motion_status::final_check) {
							m_motion_status = Motion_status::done;
							LOG(INFO) << dummy2;
							LOG(INFO) << dummy3;
						}
						else {
							get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
							next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
							//m_motion_status = Motion_status::reconstruction;
							m_current_color_id += 1;
							return next_pos;
						}
					}
					else {
						auto& item = m_ccpp_trajectory[m_current_ccpp_trajectory_id].first;

						std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> start_points;

						start_points.emplace_back(
							Eigen::Vector3f(
								item.x() - DISTANCE_THRESHOLD * 0.3,
								item.y(),
								100),
							Eigen::Vector3f(1,0,0)
						);
						start_points.emplace_back(
							Eigen::Vector3f(
								item.x() + DISTANCE_THRESHOLD * 0.3,
								item.y(),
								100),
							Eigen::Vector3f(-1,0,0)
						);
						start_points.emplace_back(
							Eigen::Vector3f(
								item.x() ,
								item.y() + DISTANCE_THRESHOLD * 0.3,
								100),
							Eigen::Vector3f(0,-1,0)
						);
						start_points.emplace_back(
							Eigen::Vector3f(
								item.x() ,
								item.y() - DISTANCE_THRESHOLD * 0.3,
								100),
							Eigen::Vector3f(0,1,0)
						);

						int nearest_id = std::min_element(start_points.begin(), start_points.end(),
							[&v_cur_pos](auto& a1, auto& a2) {
							return (v_cur_pos.pos_mesh - a1.first).norm() < (v_cur_pos.pos_mesh - a2.first).norm();
						}) - start_points.begin();

						m_exploration_point.emplace(
							start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 0 * start_points[nearest_id].second,
							start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 0 * start_points[nearest_id].second +
							(m_arg["fix_angle_flag"].asBool()?Eigen::Vector3f(std::cos(64.f / 180 * M_PI), 0, -std::sin(64.f / 180 * M_PI)) : Eigen::Vector3f(-0.5, -0.5 * 1.732, -std::tan(64.f / 180 * M_PI)).normalized())
						);
						m_exploration_point.emplace(
							start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 1 * start_points[nearest_id].second,
							start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 1 * start_points[nearest_id].second + 
							(m_arg["fix_angle_flag"].asBool() ? Eigen::Vector3f(std::cos(64.f / 180 * M_PI), 0, -std::sin(64.f / 180 * M_PI)) : Eigen::Vector3f(1, 0, -std::tan(64.f / 180 * M_PI)).normalized())
						);
						m_exploration_point.emplace(
							start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 2 * start_points[nearest_id].second,
							start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 2 * start_points[nearest_id].second + 
							(m_arg["fix_angle_flag"].asBool() ? Eigen::Vector3f(std::cos(64.f / 180 * M_PI), 0, -std::sin(64.f / 180 * M_PI)) : Eigen::Vector3f(-0.5, 0.5 * 1.732, -std::tan(64.f / 180 * M_PI)).normalized())
						);
					
						
						/*m_exploration_point.emplace(
							Eigen::Vector3f(
								item.x() - DISTANCE_THRESHOLD * 0.3,
								item.y(),
								100),
							Eigen::Vector3f(0, 1, -std::tan(64.f / 180 * M_PI)).normalized()
						);
						m_exploration_point.emplace(
							Eigen::Vector3f(
								item.x() - DISTANCE_THRESHOLD * 0.1,
								item.y(),
								100),
							Eigen::Vector3f(1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
						);
						m_exploration_point.emplace(
							Eigen::Vector3f(
								item.x() + DISTANCE_THRESHOLD * 0.1,
								item.y(),
								100),
							Eigen::Vector3f(0, -1, -std::tan(64.f / 180 * M_PI)).normalized()
						);
						m_exploration_point.emplace(
							Eigen::Vector3f(
								item.x() + DISTANCE_THRESHOLD * 0.3,
								item.y(),
								100),
							Eigen::Vector3f(-1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
						); */
						//m_exploration_point.emplace(
						//	Eigen::Vector3f(
						//		item.x(),
						//		item.y(),
						//		100),
						//	Eigen::Vector3f(-1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
						//);
						//next_pos = m_ccpp_trajectory[m_current_ccpp_trajectory_id];
						next_pos = m_exploration_point.front();
						m_exploration_point.pop();
					}
				}
				else
				{
					auto item = m_exploration_point.front();
					m_exploration_point.pop();
					return item;
				}
				//m_motion_status = Motion_status::exploration;
				
			}
		}
		if (m_motion_status == Motion_status::reconstruction_in_exploration)
		{
			const int& cur_building_id = m_current_building_id;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			if (passed_trajectory.size() == 0)
			{
				int id_closest_trajectory = std::min_element(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
					[&v_cur_pos](const auto& tra1, const auto& tra2) {
						return (tra1.first - v_cur_pos.pos_mesh).norm() < (tra2.first - v_cur_pos.pos_mesh).norm();
					}) - v_buildings[cur_building_id].trajectory.begin();
					v_buildings[cur_building_id].closest_trajectory_id = id_closest_trajectory;
			}
			int start_pos_id = 0;
			int one_pass_trajectory_num = v_buildings[cur_building_id].one_pass_trajectory_num;
			std::copy_if(v_buildings[cur_building_id].trajectory.begin() + v_buildings[cur_building_id].closest_trajectory_id, v_buildings[cur_building_id].trajectory.end(),
				std::back_inserter(unpassed_trajectory),
				[&passed_trajectory, v_threshold, &unpassed_trajectory, &start_pos_id, one_pass_trajectory_num](const auto& item_new_trajectory) {
					bool untraveled = true;
					for (auto item_passed_trajectory_iter = passed_trajectory.begin(); item_passed_trajectory_iter < passed_trajectory.end(); ++item_passed_trajectory_iter) {
						auto item_passed_trajectory = *item_passed_trajectory_iter;
						Eigen::Vector3f vector1 = (item_passed_trajectory.second - item_passed_trajectory.first).normalized();
						Eigen::Vector3f vector2 = (item_new_trajectory.second - item_new_trajectory.first).normalized();
						float dot_product = vector1.dot(vector2);
						if (dot_product > 1)
							dot_product = 1;
						float angle = std::acos(dot_product) / M_PI * 180;
						if (angle >= 180)
							angle = 0;
						if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold && angle < 5) {
							untraveled = false;
							/*if (unpassed_trajectory.size() - start_pos_id < one_pass_trajectory_num / 2)
								start_pos_id = unpassed_trajectory.size();
							if ((item_passed_trajectory_iter - passed_trajectory.begin()) == passed_trajectory.size() - 1)
								start_pos_id = 0;*/
							//TODO trajectory merge.
						}
					}
					return untraveled;
				});

			std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.begin() + v_buildings[cur_building_id].closest_trajectory_id,
				std::back_inserter(unpassed_trajectory),
				[&passed_trajectory, v_threshold, &unpassed_trajectory, &start_pos_id, one_pass_trajectory_num](const auto& item_new_trajectory) {
					bool untraveled = true;
					for (auto item_passed_trajectory_iter = passed_trajectory.begin(); item_passed_trajectory_iter < passed_trajectory.end(); ++item_passed_trajectory_iter) {
						auto item_passed_trajectory = *item_passed_trajectory_iter;
						Eigen::Vector3f vector1 = (item_passed_trajectory.second - item_passed_trajectory.first).normalized();
						Eigen::Vector3f vector2 = (item_new_trajectory.second - item_new_trajectory.first).normalized();
						float dot_product = vector1.dot(vector2);
						if (dot_product > 1)
							dot_product = 1;
						float angle = std::acos(dot_product) / M_PI * 180;
						if (angle >= 180)
							angle = 0;
						if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold && angle < 5) {
							untraveled = false;
							/*if (unpassed_trajectory.size() - start_pos_id < one_pass_trajectory_num / 2)
								start_pos_id = unpassed_trajectory.size();
							if ((item_passed_trajectory_iter - passed_trajectory.begin()) == passed_trajectory.size() - 1)
								start_pos_id = 0;*/
								//TODO trajectory merge.
						}
					}
					return untraveled;
				});

			if (unpassed_trajectory.size() == 0) {
				//debug_img(std::vector<cv::Mat>{cv::Mat(1, 1, CV_8UC1)});
				if(with_exploration)
				{
					m_motion_status = Motion_status::exploration;
					next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
				}
				else
				{
					m_motion_status = Motion_status::reconstruction_in_exploration;
					get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
					if(m_motion_status == Motion_status::final_check)
					{
						m_motion_status = Motion_status::done;
						return next_pos;
					}

					next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
				}

				//get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				//LOG(INFO) << "Change target !";
				//return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				if (start_pos_id >= unpassed_trajectory.size())
					start_pos_id = 0;
				next_pos = unpassed_trajectory.at(start_pos_id);
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}
			
			//const int& cur_building_id = m_current_building_id;
			//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			//std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
			//	std::back_inserter(unpassed_trajectory),
			//	[&passed_trajectory, v_threshold](const auto& item_new_trajectory) {
			//	bool untraveled = true;
			//	for (const auto& item_passed_trajectory : passed_trajectory)
			//		if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold) {
			//			untraveled = false;
			//		}
			//	return untraveled;
			//});

			//if (unpassed_trajectory.size() == 0) {
			//	//debug_img(std::vector<cv::Mat>{cv::Mat(1,1,CV_8UC1)});
			//	m_motion_status = Motion_status::exploration;
			//	next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			//}
			//else
			//{
			//	auto it_min_distance = std::min_element(
			//		unpassed_trajectory.begin(), unpassed_trajectory.end(),
			//		[&v_cur_pos](const auto& t1, const auto& t2) {
			//		return (t1.first - v_cur_pos.pos_mesh).norm() < (t2.first - v_cur_pos.pos_mesh).norm();
			//	});
			//	next_pos = *it_min_distance;
			//	passed_trajectory.push_back(next_pos);
			//}
		}
		return next_pos;
	}
};

class Next_best_target_min_max_information :public Next_best_target {
public:
	int m_current_exploration_id = -1;
	cv::Vec3b color_unobserved, color_free, color_occupied;
	
	Next_best_target_min_max_information(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,float v_ccpp_cell_distance)
	:Next_best_target(v_map_start_mesh,v_map_end_mesh, v_ccpp_cell_distance) {
		color_unobserved = region_viz_color[0];
		color_free = region_viz_color[1];
		color_occupied = region_viz_color[2];
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		{
			// Find with distance
			for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
				if (v_buildings[i_building].passed_trajectory.size() == 0)
					untraveled_buildings.emplace_back(i_building, -1);
			}
			if (with_exploration)
				for (int i_point = 0; i_point < sample_points.size(); ++i_point)
					if (region_status[i_point] == color_unobserved)
						untraveled_buildings.emplace_back(-1, i_point);

			std::nth_element(untraveled_buildings.begin(),
				untraveled_buildings.begin() + std::min(5, (int)untraveled_buildings.size()),
				untraveled_buildings.end(),
				[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
				float distance1, distance2;
				if (b1.origin_index_in_building_vector != -1)
					distance1 = (v_buildings[b1.origin_index_in_building_vector].bounding_box_3d.box.center() - v_cur_pos.pos_mesh).norm();
				else
					distance1 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b1.origin_index_in_untraveled_pointset].x(), sample_points[b1.origin_index_in_untraveled_pointset].y())).norm();
				if (b2.origin_index_in_building_vector != -1)
					distance2 = (v_buildings[b2.origin_index_in_building_vector].bounding_box_3d.box.center() - v_cur_pos.pos_mesh).norm();
				else
					distance2 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b2.origin_index_in_untraveled_pointset].x(), sample_points[b2.origin_index_in_untraveled_pointset].y())).norm();

				return  distance1 < distance2;
			});
		}
		if (untraveled_buildings.size() == 0)
		{
			m_motion_status = Motion_status::done;
			return;
		}
		if (!with_exploration)
		{
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[0].origin_index_in_building_vector;
			return;
		}
		int untreavel_size = std::min(5, (int)untraveled_buildings.size());
		// Find next building with higher information gain
		std::vector<float> information_gain(untreavel_size, 0.f);
		{
			for (int i_building = 0; i_building < untreavel_size; ++i_building) {
				if (untraveled_buildings[i_building].origin_index_in_building_vector == -1) {
					if ((sample_points[untraveled_buildings[i_building].origin_index_in_untraveled_pointset]
						- Point_2(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).squared_length() < 500 * 500)
						information_gain[i_building] = 1;
					else
						information_gain[i_building] = 0;
					continue;
				}
				for (int i_point = 0; i_point < region_status.size(); i_point++) {
					//if (region_status[i_point] != color_unobserved)
					if (region_status[i_point] == color_free)
						continue;
					const auto& original_bounding_box_3d = v_buildings[untraveled_buildings[i_building].origin_index_in_building_vector].bounding_box_3d;
					const CGAL::Bbox_2 box(
						original_bounding_box_3d.box.min().x() - DISTANCE_THRESHOLD, original_bounding_box_3d.box.min().y() - DISTANCE_THRESHOLD,
						original_bounding_box_3d.box.max().x() + DISTANCE_THRESHOLD, original_bounding_box_3d.box.max().y() + DISTANCE_THRESHOLD
					);
					{
						const CGAL::Point_2<K>& p = sample_points[i_point];
						if (p.x() > box.xmin() && p.x() < box.xmax() && p.y() > box.ymin() && p.y() > box.ymin())
							information_gain[i_building] += 1;
					}

				}
			}
		}

		int next_target_id = std::max_element(information_gain.begin(), information_gain.end()) - information_gain.begin();
		if (!with_exploration) {
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[0].origin_index_in_building_vector;
			return;
		}

		if (untraveled_buildings[next_target_id].origin_index_in_building_vector == -1) {
			m_motion_status = Motion_status::exploration;
			m_current_exploration_id = untraveled_buildings[next_target_id].origin_index_in_untraveled_pointset;
		}
		else {
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
		}
		return ;

	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		}) - sample_points.begin();
		region_status[nearest_region_id] = color_free;

		for (int i_point = 0; i_point < region_status.size(); i_point++) {
			if (region_status[i_point] != color_unobserved)
				continue;
			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

			for (const auto& item_building : v_buildings) {
				Eigen::Vector3f point(p.x(), p.y(), 0.f);
				if (item_building.bounding_box_3d.inside_2d(point)) {
					region_status[i_point] = color_occupied;
					break;
				}
			}
			continue;
		}
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status == Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if (m_motion_status == Motion_status::exploration) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			Eigen::Vector3f next_pos(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y(), 100);
			return std::make_pair(next_pos, (next_pos - v_cur_pos.pos_mesh).normalized());
		}
		if (m_motion_status == Motion_status::reconstruction) {
			const int& cur_building_id = m_current_building_id;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
				std::back_inserter(unpassed_trajectory),
				[&passed_trajectory, v_threshold](const auto& item_new_trajectory) {
				bool untraveled = true;
				for (const auto& item_passed_trajectory : passed_trajectory)
					if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold) {
						untraveled = false;
					}
				return untraveled;
			});

			if (unpassed_trajectory.size() == 0) {
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				LOG(INFO) << "Change target !";
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				auto it_min_distance = std::min_element(
					unpassed_trajectory.begin(), unpassed_trajectory.end(),
					[&v_cur_pos](const auto& t1, const auto& t2) {
					return (t1.first - v_cur_pos.pos_mesh).norm() < (t2.first - v_cur_pos.pos_mesh).norm();
				});
				next_pos = *it_min_distance;
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}
		}
	}

};


class Next_best_target_order_reconstruction :public Next_best_target_topology_exploration {
public:
	Next_best_target_order_reconstruction(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,
		int v_CCPP_CELL_THRESHOLD, const Polygon_2& m_boundary, float v_ccpp_cell_distance,const Json::Value& v_arg)
	:Next_best_target_topology_exploration(v_map_start_mesh, v_map_end_mesh, v_CCPP_CELL_THRESHOLD, m_boundary, v_ccpp_cell_distance, v_arg){}
	
	//int m_current_exploration_id = -1;
	//cv::Vec3b color_unobserved, color_free, color_occupied;

	//std::vector<bool> already_explored;

	//std::queue<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_exploration_point;

	//Next_best_target_order_reconstruction(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh, float v_ccpp_cell_distance)
	//	:Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance) {
	//	color_unobserved = region_viz_color[0];
	//	color_free = region_viz_color[1];
	//	color_occupied = region_viz_color[2];
	//	already_explored.resize(region_status.size(), false);
	//}

	//void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
	//	// Find next building in current cell
	//	std::vector<Next_target> untraveled_buildings;
	//	for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
	//		if (v_buildings[i_building].passed_trajectory.size() == 0
	//			//&& (v_buildings[i_building].bounding_box_3d.center().block(0,0,2,1)-Eigen::Vector2f(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y())).norm()<DISTANCE_THRESHOLD/2
	//			)
	//			untraveled_buildings.emplace_back(i_building, -1);
	//	}
	//	if (untraveled_buildings.size() == 0 && with_exploration)
	//	{
	//		for (int i_point = 0; i_point < sample_points.size(); ++i_point)
	//			if (!already_explored[i_point])
	//				//if (region_status[i_point]==color_unobserved)
	//				untraveled_buildings.emplace_back(-1, i_point);
	//	}

	//	if (untraveled_buildings.size() == 0) {
	//		m_motion_status = Motion_status::done;
	//		return;
	//	}

	//	int next_target_id = std::min_element(untraveled_buildings.begin(),
	//		untraveled_buildings.end(),
	//		[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
	//			float distance1, distance2;
	//			if (b1.origin_index_in_building_vector != -1)
	//			{
	//				int id_trajectory = v_buildings[b1.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
	//				Eigen::Vector3f pos = v_buildings[b1.origin_index_in_building_vector].trajectory[id_trajectory].first;
	//				Eigen::Vector2f pos_2(pos.x(), pos.y());
	//				distance1 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
	//			}
	//			else
	//			{
	//				distance1 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b1.origin_index_in_untraveled_pointset].x(), sample_points[b1.origin_index_in_untraveled_pointset].y())).norm();
	//			}
	//			if (b2.origin_index_in_building_vector != -1)
	//			{
	//				int id_trajectory = v_buildings[b2.origin_index_in_building_vector].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
	//				Eigen::Vector3f pos = v_buildings[b2.origin_index_in_building_vector].trajectory[id_trajectory].first;
	//				Eigen::Vector2f pos_2(pos.x(), pos.y());
	//				distance2 = (pos_2 - Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y())).norm();
	//			}
	//			else
	//				distance2 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b2.origin_index_in_untraveled_pointset].x(), sample_points[b2.origin_index_in_untraveled_pointset].y())).norm();

	//			return  distance1 < distance2;
	//		}) - untraveled_buildings.begin();

	//		if (!with_exploration) {
	//			m_motion_status = Motion_status::reconstruction;
	//			m_current_building_id = untraveled_buildings[0].origin_index_in_building_vector;
	//			return;
	//		}

	//		if (untraveled_buildings[next_target_id].origin_index_in_building_vector == -1) {
	//			m_motion_status = Motion_status::exploration;
	//			m_current_exploration_id = untraveled_buildings[next_target_id].origin_index_in_untraveled_pointset;

	//			/*m_exploration_point.emplace(
	//				Eigen::Vector3f(
	//					sample_points[m_current_exploration_id].x() - DISTANCE_THRESHOLD * 0.3,
	//					sample_points[m_current_exploration_id].y(),
	//					100),
	//				Eigen::Vector3f(0, 1, -std::tan(64.f / 180 * M_PI)).normalized()
	//			);
	//			m_exploration_point.emplace(
	//				Eigen::Vector3f(
	//					sample_points[m_current_exploration_id].x() - DISTANCE_THRESHOLD * 0.1,
	//					sample_points[m_current_exploration_id].y(),
	//					100),
	//				Eigen::Vector3f(1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
	//			);
	//			m_exploration_point.emplace(
	//				Eigen::Vector3f(
	//					sample_points[m_current_exploration_id].x() + DISTANCE_THRESHOLD * 0.1,
	//					sample_points[m_current_exploration_id].y(),
	//					100),
	//				Eigen::Vector3f(0, -1, -std::tan(64.f / 180 * M_PI)).normalized()
	//			);
	//			m_exploration_point.emplace(
	//				Eigen::Vector3f(
	//					sample_points[m_current_exploration_id].x() + DISTANCE_THRESHOLD * 0.3,
	//					sample_points[m_current_exploration_id].y(),
	//					100),
	//				Eigen::Vector3f(-1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
	//			);*/
	//			m_exploration_point.emplace(
	//				Eigen::Vector3f(
	//					sample_points[m_current_exploration_id].x(),
	//					sample_points[m_current_exploration_id].y(),
	//					100),
	//				Eigen::Vector3f(-1, 0, -std::tan(64.f / 180 * M_PI)).normalized()
	//			);

	//		}
	//		else {
	//			m_motion_status = Motion_status::reconstruction;
	//			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
	//		}
	//		return;
	//}

	//void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
	//	Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
	//	int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
	//		[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
	//			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
	//		}) - sample_points.begin();
	//		region_status[nearest_region_id] = color_free;
	//		if (m_motion_status == Motion_status::exploration)
	//			already_explored[nearest_region_id] = true;

	//		for (int i_point = 0; i_point < region_status.size(); i_point++) {
	//			if (region_status[i_point] != color_unobserved)
	//				continue;
	//			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

	//			for (const auto& item_building : v_buildings) {
	//				Eigen::AlignedBox2f box(Eigen::Vector2f(item_building.bounding_box_3d.min().x(), item_building.bounding_box_3d.min().y()),
	//					Eigen::Vector2f(item_building.bounding_box_3d.max().x(), item_building.bounding_box_3d.max().y()));
	//				if (p.x() > item_building.bounding_box_3d.min().x() && p.x() < item_building.bounding_box_3d.max().x() &&
	//					p.y() > item_building.bounding_box_3d.min().y() && p.y() < item_building.bounding_box_3d.max().y()) {
	//					region_status[i_point] = color_occupied;
	//					break;
	//				}
	//			}
	//			continue;
	//		}
	//}

	//std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
	//	std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

	//	if (m_motion_status == Motion_status::initialization) {
	//		get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
	//		LOG(INFO) << "Initialization target !";
	//	}

	//	if (m_motion_status == Motion_status::exploration) {
	//		if (m_exploration_point.size() > 0)
	//		{
	//			auto item = m_exploration_point.front();
	//			m_exploration_point.pop();
	//			return item;
	//		}
	//		else
	//		{
	//			already_explored[m_current_exploration_id] = true;
	//			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
	//			return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
	//		}

	//	}
	//	if (m_motion_status == Motion_status::reconstruction) {
	//		const int& cur_building_id = m_current_building_id;
	//		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> unpassed_trajectory;
	//		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
	//		int start_pos_id = 0;
	//		std::copy_if(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
	//			std::back_inserter(unpassed_trajectory),
	//			[&passed_trajectory, v_threshold, &unpassed_trajectory, &start_pos_id](const auto& item_new_trajectory) {
	//				bool untraveled = true;
	//				for (auto item_passed_trajectory_iter = passed_trajectory.begin(); item_passed_trajectory_iter < passed_trajectory.end(); ++item_passed_trajectory_iter) {
	//					auto item_passed_trajectory = *item_passed_trajectory_iter;
	//					if ((item_passed_trajectory.first - item_new_trajectory.first).norm() < v_threshold) {
	//						untraveled = false;
	//						start_pos_id = unpassed_trajectory.size();
	//						if ((item_passed_trajectory_iter - passed_trajectory.begin()) == passed_trajectory.size() - 1)
	//							start_pos_id = 0;
	//					}
	//				}
	//				return untraveled;
	//			});

	//		if (unpassed_trajectory.size() == 0) {
	//			//debug_img(std::vector<cv::Mat>{cv::Mat(1, 1, CV_8UC1)});
	//			//m_motion_status = Motion_status::exploration;
	//			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
	//			LOG(INFO) << "Change target !";
	//			return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
	//		}
	//		else {
	//			if (start_pos_id >= unpassed_trajectory.size())
	//				start_pos_id = 0;
	//			next_pos = unpassed_trajectory.at(start_pos_id);
	//			passed_trajectory.push_back(next_pos);
	//			return next_pos;
	//		}
	//	}
	//}

};


class Mapper
{
public:
	Json::Value m_args;
	Polygon_2 m_boundary;
	Mapper(const Json::Value& v_args):m_args(v_args)
	{
		std::vector<Point_2> points;
		if (!v_args["boundary"].isNull()) {
			int num_point = v_args["boundary"].size() / 3;
			for (int i_point = 0; i_point < num_point; ++i_point) {
				Point_2 p_lonlat(v_args["boundary"][i_point * 3 + 0].asFloat(),
					v_args["boundary"][i_point * 3 + 1].asFloat());
				Eigen::Vector2f p_mercator = lonLat2Mercator(Eigen::Vector2f(p_lonlat.x(), p_lonlat.y())) - Eigen::Vector2f(v_args["geo_origin"][0].asFloat(), v_args["geo_origin"][1].asFloat());
				//Eigen::Vector2f p_mercator = lonLat2Mercator(Eigen::Vector2f(p_lonlat.x(), p_lonlat.y())) - lonLat2Mercator(Eigen::Vector2f(v_args["geo_origin"][0].asFloat(), v_args["geo_origin"][1].asFloat()));
				points.emplace_back(p_mercator.x(), p_mercator.y());
			}
			m_boundary = Polygon_2(points.begin(), points.end());
		}
	};
	virtual void get_buildings(std::vector<Building>& v_buildings,
		const Pos_Pack& v_current_pos,
		const int v_cur_frame_id, Height_map& v_height_map)=0;
};

class GT_mapper:public Mapper
{
public:
	std::vector<Building> m_buildings_target;
	std::vector<Building> m_buildings_safe_place;
	GT_mapper(const Json::Value& args): Mapper(args)
	{
		if(false)
		{
			m_buildings_target.resize(3);
			m_buildings_target[0].bounding_box_3d= Rotated_box(Eigen::AlignedBox3f(
				Eigen::Vector3f(130 * 4 - 10, 130 * 6 - 10, 0),
				Eigen::Vector3f(130 * 4 + 10, 130 * 6 + 10, 50)
			));
			m_buildings_target[1].bounding_box_3d= Rotated_box(Eigen::AlignedBox3f(
				Eigen::Vector3f(130 * 8 - 10,130 * 3 - 10,0),
				Eigen::Vector3f(130 * 8 + 10,130 * 3 + 10,50)
			));
			m_buildings_target[2].bounding_box_3d= Rotated_box(Eigen::AlignedBox3f(
				Eigen::Vector3f(130 * 6 - 10, 130 * 1 - 10, 0),
				Eigen::Vector3f(130 * 6 + 10, 130 * 1 + 10, 50)
			));
			
			
			for (int cluster_id = 0; cluster_id < m_buildings_target.size(); ++cluster_id) {
				Building& current_building = m_buildings_target[cluster_id];
				current_building.boxes.push_back(current_building.bounding_box_3d);
			}
			m_buildings_safe_place = m_buildings_target;
		}
		else if (false)
		{
			tinyobj::attrib_t attr;
			std::vector<tinyobj::shape_t> shapes;
			std::vector<tinyobj::material_t> mtl;
			std::tie(attr, shapes, mtl) = load_obj(args["model_path"].asString());

			m_buildings_target.resize(shapes.size());
			for (int cluster_id = 0; cluster_id < shapes.size(); ++cluster_id) {
				Building& current_building = m_buildings_target[cluster_id];
				size_t index_offset = 0;
				for(int i_face=0; i_face <shapes[cluster_id].mesh.num_face_vertices.size();++i_face)
				{
					for(int i_vertice=0;i_vertice< shapes[cluster_id].mesh.num_face_vertices[i_face];++i_vertice)
					{
						tinyobj::index_t idx = shapes[cluster_id].mesh.indices[index_offset + i_vertice];

						current_building.points_world_space.insert(Point_3(
							attr.vertices[3 * idx.vertex_index + 0],
							attr.vertices[3 * idx.vertex_index + 1],
							attr.vertices[3 * idx.vertex_index + 2]
						));
					}
					index_offset += shapes[cluster_id].mesh.num_face_vertices[i_face];
				}
				current_building.bounding_box_3d = get_bounding_box(current_building.points_world_space);
				current_building.boxes.push_back(current_building.bounding_box_3d);
			}
			m_buildings_safe_place = m_buildings_target;

		}
		else if (true)
		{

			CGAL::Point_set_3<Point_3, Vector_3> original_point_cloud;
			std::vector<CGAL::Point_set_3<Point_3, Vector_3>> pcs;
			if (boost::filesystem::is_directory(args["model_path"].asString())) {
				boost::filesystem::directory_iterator end_iter;
				for (fs::directory_iterator iter(args["model_path"].asString()); iter != end_iter; iter++) {
					CGAL::Point_set_3<Point_3, Vector_3> pc_item;
					CGAL::read_ply_point_set(std::ifstream(iter->path().string(), std::ios::binary), pc_item);
					pcs.push_back(pc_item);
				}
				//CGAL::read_ply_point_set(std::ifstream("D:\\datasets\\Realcity\\Shenzhen\\sample_500000.ply", std::ios::binary), original_point_cloud);
				//CGAL::Point_set_3<Point_3, Vector_3> point_cloud(original_point_cloud);

				m_buildings_safe_place.resize(pcs.size());
				for (int cluster_id = 0; cluster_id < pcs.size(); ++cluster_id) {
					Building& current_building = m_buildings_safe_place[cluster_id];
					bool add_as_target = true;
					for (Point_set::Index idx : pcs[cluster_id]) {
						current_building.points_world_space.insert(pcs[cluster_id].point(idx));
						Point_2 p(pcs[cluster_id].point(idx).x(), pcs[cluster_id].point(idx).y());
						if(m_boundary.size()>0)
						{
							for (auto iter_segment = m_boundary.edges_begin(); iter_segment != m_boundary.edges_end(); ++iter_segment)
								if (CGAL::squared_distance(p, *iter_segment) < 00 * 00)
									add_as_target = false;
							if(m_boundary.bounded_side(p)!=CGAL::ON_BOUNDED_SIDE)
								add_as_target = false;
						}
					}
					//current_building.bounding_box_3d = get_bounding_box(current_building.points_world_space);
					current_building.bounding_box_3d = get_bounding_box_rotated(current_building.points_world_space);
					current_building.boxes.push_back(current_building.bounding_box_3d);

					if(add_as_target)
						m_buildings_target.push_back(current_building);
				}
				return;
			}
			else {
				CGAL::read_ply_point_set(std::ifstream(args["model_path"].asString(), std::ios::binary), original_point_cloud);
				CGAL::Point_set_3<Point_3, Vector_3> point_cloud(original_point_cloud);
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

					std::vector<std::pair<std::size_t, std::size_t>> adjacencies;

					nb_clusters = CGAL::cluster_point_set(point_cloud, cluster_map,
						point_cloud.parameters().neighbor_radius(
							args["cluster_radius"].asFloat()).
						adjacencies(std::back_inserter(adjacencies)));
					m_buildings_target.resize(nb_clusters);

					Point_set::Property_map<unsigned char> red = point_cloud.add_property_map<unsigned char>("red", 0).first;
					Point_set::Property_map<unsigned char> green = point_cloud
						.add_property_map<unsigned char>("green", 0).first;
					Point_set::Property_map<unsigned char> blue = point_cloud.add_property_map<unsigned char>("blue", 0).first;
					for (Point_set::Index idx : point_cloud) {
						// One color per cluster
						int cluster_id = cluster_map[idx];
						CGAL::Random rand(cluster_id);
						red[idx] = rand.get_int(64, 192);
						green[idx] = rand.get_int(64, 192);
						blue[idx] = rand.get_int(64, 192);

						Building& current_building = m_buildings_target[cluster_id];
						current_building.points_world_space.insert(point_cloud.point(idx));
					}
				}
				for (int i_building_1 = 0; i_building_1 < m_buildings_target.size(); ++i_building_1) {
					//m_buildings_target[i_building_1].bounding_box_3d = get_bounding_box(m_buildings_target[i_building_1].points_world_space);
					m_buildings_target[i_building_1].bounding_box_3d = get_bounding_box_rotated(m_buildings_target[i_building_1].points_world_space);
					m_buildings_target[i_building_1].bounding_box_3d.box.min().z() -= args["HEIGHT_CLIP"].asFloat();
					m_buildings_target[i_building_1].boxes.push_back(m_buildings_target[i_building_1].bounding_box_3d);
				}

				for (int i_building_1 = m_buildings_target.size() - 1; i_building_1 >= 0; --i_building_1) 
				{
					cv::Point2f points[4];
					m_buildings_target[i_building_1].bounding_box_3d.cv_box.points(points);

					if (m_boundary.size() > 0)
					{
						bool should_delete = false;
						for (int i_point = 0; i_point < 4; ++i_point) {
							Point_2 p(points[i_point].x, points[i_point].y);
							for (auto iter_segment = m_boundary.edges_begin(); iter_segment != m_boundary.edges_end(); ++iter_segment)
								if (CGAL::squared_distance(p, *iter_segment) < args["safe_distance"].asFloat() * args["safe_distance"].asFloat() * 2)
									should_delete = true;
							if (m_boundary.bounded_side(p) != CGAL::ON_BOUNDED_SIDE)
								should_delete = true;
						}
						if(should_delete)
							m_buildings_target.erase(m_buildings_target.begin() + i_building_1);

					}
				}

				
				
				m_buildings_safe_place = m_buildings_target;
			}
		}
		
	}

	void get_buildings(std::vector<Building>& v_buildings,
		const Pos_Pack& v_current_pos,
		const int v_cur_frame_id, Height_map& v_height_map) override
	{
		if (v_buildings.size() == 0)
		{
			v_buildings = m_buildings_target;
			for (auto& item_building : m_buildings_safe_place) {
				v_height_map.update(item_building.bounding_box_3d);
			}
		}
		
		return;
	}
};

class Virtual_mapper:public Mapper {
public:
	Unreal_object_detector* m_unreal_object_detector;
	Synthetic_SLAM* m_synthetic_SLAM;
	Airsim_tools* m_airsim_client;
	std::map<cv::Vec3b, std::string> m_color_to_mesh_name_map;
	
	Virtual_mapper(const Json::Value& args, Airsim_tools* v_airsim_client, std::map<cv::Vec3b, std::string> color_to_mesh_name_map)
	: Mapper(args), m_airsim_client(v_airsim_client){
		m_unreal_object_detector =new Unreal_object_detector;
		m_color_to_mesh_name_map = color_to_mesh_name_map;
		//m_synthetic_SLAM = new Synthetic_SLAM;
	}

	struct ImageCluster
	{
		CGAL::Bbox_2 box;
		std::string name;
		cv::Vec3b color;
		std::vector<int> xs;
		std::vector<int> ys;
	};

	std::vector<ImageCluster> solveCluster(const cv::Mat& vSeg, const std::map<cv::Vec3b, std::string> colorMap, bool& isValid) {

		isValid = true;
		std::map<cv::Vec3b, int> currentColor;
		std::vector<std::pair<std::vector<int>, std::vector<int>>> bbox;

		int background_num = 0;
		for (int y = 0; y < vSeg.size().height; y++) {
			for (int x = 0; x < vSeg.size().width; x++) {
				cv::Vec3b pixel = vSeg.at<cv::Vec3b>(y, x);
				if (pixel == cv::Vec3b(55, 181, 57))
				{
					background_num++;
					continue;
				}

				if (currentColor.find(pixel) == currentColor.end()) {
					currentColor.insert(std::make_pair(pixel, bbox.size()));
					bbox.push_back(std::make_pair(std::vector<int>(), std::vector<int>()));
				}

				bbox[currentColor.at(pixel)].first.push_back(x);
				bbox[currentColor.at(pixel)].second.push_back(y);
			}
		}

		std::vector<ImageCluster> result;
		if (background_num > vSeg.size().height * vSeg.size().width * 0.8)
		{
			isValid = false;
			return result;
		}

		int small_building_num = 0;
		for (auto colorIter = currentColor.begin(); colorIter != currentColor.end(); colorIter++) {
			ImageCluster cluster;
			if (bbox[colorIter->second].first.size() < 30 * 30)
			{
				small_building_num++;
				continue;
			}
			//if (bbox[colorIter->second].first.size() < 30 * 30)
			//{
			//	isValid = false;
			//	break;
			//}
			cluster.box = CGAL::Bbox_2(
				*std::min_element(bbox[colorIter->second].first.begin(), bbox[colorIter->second].first.end()),
				*std::min_element(bbox[colorIter->second].second.begin(), bbox[colorIter->second].second.end()),
				*std::max_element(bbox[colorIter->second].first.begin(), bbox[colorIter->second].first.end()),
				*std::max_element(bbox[colorIter->second].second.begin(), bbox[colorIter->second].second.end())
			);
			cluster.color = colorIter->first;
			if (colorMap.find(cluster.color) == colorMap.end())
				continue;
			cluster.name = std::to_string(std::atoi(colorMap.at(cluster.color).c_str()));
			cluster.xs = bbox[currentColor.at(colorIter->first)].first;
			cluster.ys = bbox[currentColor.at(colorIter->first)].second;
			result.push_back(cluster);
		}
		if (small_building_num > 40)
			isValid = false;
		return result;
	}

	void get_buildings(std::vector<Building>& v_buildings,
		const Pos_Pack& v_current_pos,
		const int v_cur_frame_id, Height_map& v_height_map) override {
		std::vector<Building> current_buildings;
		int num_building_current_frame;
		// Get current image and pose
		// Input: 
		// Output: Image(cv::Mat), Camera matrix(Pos_pack)
		std::map<std::string, cv::Mat> current_image;
		{
			m_airsim_client->adjust_pose(v_current_pos);
			current_image = m_airsim_client->get_images();
			//cv::imwrite("D:/test_data/" + std::to_string(v_cur_frame_id) + ".png", current_image.at("rgb"));
			//std::ofstream pose("D:/test_data/" + std::to_string(v_cur_frame_id) + ".txt");
			//pose << v_current_pos.camera_matrix.matrix();
			//pose.close();
			LOG(INFO) << "Image done";
		}

		// Object detection
		// Input: Vector of building (std::vector<Building>)
		// Output: Vector of building with 2D bounding box (std::vector<Building>)
		std::vector<cv::Vec3b> color_map;
		{
			m_unreal_object_detector->get_bounding_box(current_image, color_map, current_buildings);
			LOG(INFO) << "Object detection done";
		}

		// SLAM
		// Input: Image(cv::Mat), Camera matrix(cv::Iso)
		// Output: Vector of building with Point cloud in camera frames (std::vector<Building>)
		//		   Refined Camera matrix(cv::Iso)
		//		   num of clusters (int)
		{
			m_synthetic_SLAM->get_points(current_image, color_map, current_buildings);
			LOG(INFO) << "Sparse point cloud generation and building cluster done";
		}

		// Post process point cloud
		// Input: Vector of building (std::vector<Building>)
		// Output: Vector of building with point cloud in world space (std::vector<Building>)
		{
			Point_set cur_frame_total_points_in_world_coordinates;
			std::vector<bool> should_delete(current_buildings.size(), false);
			for (auto& item_building : current_buildings) {
				size_t cluster_index = &item_building - &current_buildings[0];
				if (item_building.points_camera_space.points().size() < 200) {
					should_delete[cluster_index] = true;
					continue;
				}
				for (const auto& item_point : item_building.points_camera_space.points()) {
					Eigen::Vector3f point_eigen(item_point.x(), item_point.y(), item_point.z());
					point_eigen = v_current_pos.camera_matrix.inverse() * point_eigen;
					item_building.points_world_space.insert(Point_3(point_eigen.x(), point_eigen.y(), point_eigen.z()));
					cur_frame_total_points_in_world_coordinates.insert(Point_3(point_eigen.x(), point_eigen.y(), point_eigen.z()));
				}
				//CGAL::write_ply_point_set(std::ofstream(std::to_string(cluster_index) + "_world.ply"), item_building.points_world_space);
			}
			current_buildings.erase(std::remove_if(current_buildings.begin(), current_buildings.end(),
				[&should_delete, idx = 0](const auto& item)mutable
			{
				return should_delete[idx++];
			}), current_buildings.end());
			num_building_current_frame = current_buildings.size();
			//CGAL::write_ply_point_set(std::ofstream(std::to_string(v_cur_frame_id) + "_world_points.ply"), cur_frame_total_points_in_world_coordinates);
		}

		// Mapping
		// Input: *
		// Output: Vector of building with 3D bounding box (std::vector<Building>)
		{
			if (m_args["MAP_2D_BOX_TO_3D"].asBool()) {
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
					Eigen::Vector3f point_pos_world = v_current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);

					float final_height = point_pos_world[2];
					// Shorter than camera, recalculate using max distance
					if (final_height < v_current_pos.pos_mesh[2]) {
						distance_candidate = max_distance;
						scale = distance_candidate / point_pos_camera_XZ[2];
						point_pos_world = v_current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);
						final_height = point_pos_world[2];
					}

					item_building.bounding_box_3d = get_bounding_box(item_building.points_world_space);
					item_building.bounding_box_3d.box.min()[2] = 0;
					item_building.bounding_box_3d.box.max()[2] = final_height;
				}
			}
			LOG(INFO) << "2D Bbox to 3D Bbox done";

		}

		// Merging
		// Input: 3D bounding box of current frame and previous frame
		// Output: Total building vectors (std::vector<Building>)
		{
			std::vector<bool> need_register(num_building_current_frame, true);
			for (auto& item_building : v_buildings) {
				for (const auto& item_current_building : current_buildings) {
					size_t index_box = &item_current_building - &current_buildings[0];
					// Bug here
					/*if (item_building.segmentation_color== item_current_building.segmentation_color) {
						need_register[index_box] = false;
						item_building.bounding_box_3d = item_building.bounding_box_3d.merged(item_current_building.bounding_box_3d);
						for (const auto& item_point : item_current_building.points_world_space.points())
							item_building.points_world_space.insert(item_point);
					}
					continue;
					if (item_current_building.bounding_box_3d.intersects(item_building.bounding_box_3d)) {
						float overlap_volume = item_current_building.bounding_box_3d.intersection(item_building.bounding_box_3d).volume();
						if (overlap_volume / item_current_building.bounding_box_3d.volume() > 0.5 || overlap_volume / item_building.bounding_box_3d.volume() > 0.5) {
							need_register[index_box] = false;
							item_building.bounding_box_3d = item_building.bounding_box_3d.merged(item_current_building.bounding_box_3d);
							for (const auto& item_point : item_current_building.points_world_space.points())
								item_building.points_world_space.insert(item_point);
						}
					}*/
				}
			}
			for (int i = 0; i < need_register.size(); ++i) {
				if (need_register[i]) {
					v_buildings.push_back(current_buildings[i]);
				}
			}
			LOG(INFO) << "Building BBox update: DONE!";
		}

		// Update height map
		for (auto& item_building : v_buildings) {
			v_height_map.update(item_building.bounding_box_3d);
		}
	}

};

class Real_mapper :public Mapper
{
public:
	Real_object_detector* m_real_object_detector;
	cv::Ptr<cv::Feature2D> orb;
	Synthetic_SLAM* m_synthetic_SLAM;
	//zxm::ISLAM* slam = GetInstanceOfSLAM();
	Airsim_tools* m_airsim_client;
	Real_mapper(const Json::Value& args, Airsim_tools* v_airsim_client)
		: Mapper(args), m_airsim_client(v_airsim_client) {
			m_real_object_detector = new Real_object_detector;
			orb = cv::ORB::create(MAX_FEATURES);
			m_synthetic_SLAM = new Synthetic_SLAM;
	}
	
	void get_buildings(std::vector<Building>& v_buildings, const Pos_Pack& v_current_pos, const int v_cur_frame_id,
		Height_map& v_height_map) override
	{
		std::vector<Building> current_buildings;
		int num_building_current_frame;
		std::vector<cv::KeyPoint> keypoints;
		// Get current image and pose
		// Input: 
		// Output: Image(cv::Mat), Camera matrix(Pos_pack)
		std::map<std::string, cv::Mat> current_image;
		{
			m_airsim_client->adjust_pose(v_current_pos);
			//demo_move_to_next(*(m_airsim_client->m_agent), v_current_pos.pos_airsim, v_current_pos.yaw, 5, false);
			current_image = m_airsim_client->get_images();
			cv::imwrite("F:\\Sig\\Shanghai\\" + std::to_string(v_cur_frame_id) + "_rgb.jpg", current_image["rgb"]);
			cv::imwrite("F:\\Sig\\Shanghai\\" + std::to_string(v_cur_frame_id) + "_seg.png", current_image["segmentation"]);

			LOG(INFO) << "Image done";
		}

		// Object detection
		// Input: Vector of building (std::vector<Building>)
		// Output: Vector of building with 2D bounding box (std::vector<Building>)
		std::vector<cv::Vec3b> color_map;
		std::vector<cv::Rect2f> detection_boxes;
		{
			m_real_object_detector->get_bounding_box(current_image, color_map, current_buildings);
			LOG(INFO) << "Object detection done";
		}

		// SLAM
		// Input: Image(cv::Mat), Camera matrix(cv::Iso)
		// Output: Vector of building with Point cloud in camera frames (std::vector<Building>)
		//		   Refined Camera matrix(cv::Iso)
		//		   num of clusters (int)
		{
			m_synthetic_SLAM->get_points(current_image, color_map, current_buildings);
			LOG(INFO) << "Sparse point cloud generation and building cluster done";
		}

		cv::Mat rgb = current_image["rgb"].clone();
		auto orb = cv::ORB::create(200);
		//orb->detect(rgb, keypoints, v_img.at("roi_mask"));
		orb->detect(rgb, keypoints);
		cv::drawKeypoints(rgb, keypoints, rgb);
		for (auto box : detection_boxes)
		{
			cv::rectangle(rgb, box, cv::Scalar(0, 0, 255));
		}
		cv::imwrite("F:\\Sig\\demo\\" + std::to_string(v_cur_frame_id) + ".jpg", rgb);
		

		// Post process point cloud
		// Input: Vector of building (std::vector<Building>)
		// Output: Vector of building with point cloud in world space (std::vector<Building>)
		{
			Point_set cur_frame_total_points_in_world_coordinates;
			std::vector<bool> should_delete(current_buildings.size(), false);
			for (auto& item_building : current_buildings) {
				size_t cluster_index = &item_building - &current_buildings[0];
				if (item_building.points_camera_space.points().size() < 5) {
					should_delete[cluster_index] = true;
					continue;
				}
				for (const auto& item_point : item_building.points_camera_space.points()) {
					Eigen::Vector3f point_eigen(item_point.x(), item_point.y(), item_point.z());
					point_eigen = v_current_pos.camera_matrix.inverse() * point_eigen;
					item_building.points_world_space.insert(Point_3(point_eigen.x(), point_eigen.y(), point_eigen.z()));
					cur_frame_total_points_in_world_coordinates.insert(Point_3(point_eigen.x(), point_eigen.y(), point_eigen.z()));
				}
				//CGAL::write_ply_point_set(std::ofstream(std::to_string(cluster_index) + "_world.ply"), item_building.points_world_space);
			}
			current_buildings.erase(std::remove_if(current_buildings.begin(), current_buildings.end(),
				[&should_delete, idx = 0](const auto& item)mutable
			{
				return should_delete[idx++];
			}), current_buildings.end());
			num_building_current_frame = current_buildings.size();
			//CGAL::write_ply_point_set(std::ofstream(std::to_string(v_cur_frame_id) + "_world_points.ply"), cur_frame_total_points_in_world_coordinates);
		}

		// Mapping
		// Input: *
		// Output: Vector of building with 3D bounding box (std::vector<Building>)
		{
			if (m_args["MAP_2D_BOX_TO_3D"].asBool()) {
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
					Eigen::Vector3f point_pos_world = v_current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);

					float final_height = point_pos_world[2];
					// Shorter than camera, recalculate using max distance
					if (final_height < v_current_pos.pos_mesh[2]) {
						distance_candidate = max_distance;
						scale = distance_candidate / point_pos_camera_XZ[2];
						point_pos_world = v_current_pos.camera_matrix.inverse() * (scale * point_pos_camera_XZ);
						final_height = point_pos_world[2];
					}

					item_building.bounding_box_3d = get_bounding_box(item_building.points_world_space);
					item_building.bounding_box_3d.box.min()[2] = 0;
					item_building.bounding_box_3d.box.max()[2] = final_height;
				}
			}
			LOG(INFO) << "2D Bbox to 3D Bbox done";

		}

		// Merging
		// Input: 3D bounding box of current frame and previous frame
		// Output: Total building vectors (std::vector<Building>)
		{
			std::vector<bool> need_register(num_building_current_frame, true);
			for (auto& item_building : v_buildings) {
				for (const auto& item_current_building : current_buildings) {
					// bug here
					/*size_t index_box = &item_current_building - &current_buildings[0];
					if (item_building.segmentation_color == item_current_building.segmentation_color) {
						need_register[index_box] = false;
						item_building.bounding_box_3d = item_building.bounding_box_3d.merged(item_current_building.bounding_box_3d);
						for (const auto& item_point : item_current_building.points_world_space.points())
							item_building.points_world_space.insert(item_point);
					}
					continue;
					if (item_current_building.bounding_box_3d.intersects(item_building.bounding_box_3d)) {
						float overlap_volume = item_current_building.bounding_box_3d.intersection(item_building.bounding_box_3d).volume();
						if (overlap_volume / item_current_building.bounding_box_3d.volume() > 0.5 || overlap_volume / item_building.bounding_box_3d.volume() > 0.5) {
							need_register[index_box] = false;
							item_building.bounding_box_3d = item_building.bounding_box_3d.merged(item_current_building.bounding_box_3d);
							for (const auto& item_point : item_current_building.points_world_space.points())
								item_building.points_world_space.insert(item_point);
						}
					}*/
				}
			}
			for (int i = 0; i < need_register.size(); ++i) {
				if (need_register[i]) {
					v_buildings.push_back(current_buildings[i]);
				}
			}
			LOG(INFO) << "Building BBox update: DONE!";
		}

		// Update height map
		for (auto& item_building : v_buildings) {
			v_height_map.update(item_building.bounding_box_3d);
		}
	}
};

int main(int argc, char** argv){
	// Read arguments
	LOG(INFO) << "Read config "<< argv[2];
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
	Airsim_tools* airsim_client;
	Visualizer* viz = new Visualizer;
	std::map<cv::Vec3b, std::string> color_to_mesh_name_map;
	viz->m_uncertainty_map_distance=args["ccpp_cell_distance"].asFloat();

	LOG(INFO) << "Read safe zone " << args["safe_zone_model_path"].asString();
	CGAL::Point_set_3<Point_3, Vector_3> safe_zone_point_cloud;
	CGAL::read_ply_point_set(std::ifstream(args["safe_zone_model_path"].asString(), std::ios::binary), safe_zone_point_cloud);
	Height_map original_height_map(safe_zone_point_cloud, args["heightmap_resolution"].asFloat(),
		args["heightmap_dilate"].asInt());
	
	{
		LOG(INFO) << "Initialization directory, airsim and reset color";
		if (boost::filesystem::exists(log_root))
			boost::filesystem::remove_all(log_root);
		boost::filesystem::create_directories(log_root);
		boost::filesystem::create_directories(log_root/"img");
		boost::filesystem::create_directories(log_root/"seg");
		boost::filesystem::create_directories(log_root/"point_world");
		boost::filesystem::create_directories(log_root/"ccpp_map");
		boost::filesystem::create_directories(log_root/"wgs_log");
		boost::filesystem::create_directories(log_root/"gradually_results");
		
		map_converter.initDroneStart(UNREAL_START);
		INTRINSIC << 400, 0, 400, 0, 400, 400, 0, 0, 1;

		if(args["mapper"].asString()!="gt_mapper")
		{
			airsim_client = new Airsim_tools(UNREAL_START);
			airsim_client->reset_color("building");
			//	airsim_client.m_agent->simSetSegmentationObjectID("BP_Sky_Sphere", 0);

			const boost::filesystem::path color_map_path(args["color_map"].asString());
			airsim_client->m_color_map = cv::imread(color_map_path.string());
			if (airsim_client->m_color_map.size == 0)
			{
				LOG(ERROR) << "Cannot open color map " << color_map_path << std::endl;
				return 0;
			}
			cv::cvtColor(airsim_client->m_color_map, airsim_client->m_color_map, cv::COLOR_BGR2RGB);

			color_to_mesh_name_map = airsim_client->reset_color([](std::string v_name)
				{
					std::regex rx("^[0-9_]+$");
					bool bl = std::regex_match(v_name.begin(), v_name.end(), rx);
					return bl;
				});
		}
	}

	// Some global structure
	bool end = false;
	bool is_viz=args["is_viz"].asBool();
	float DRONE_STEP = args["DRONE_STEP"].asFloat();
	bool with_interpolated =args["with_interpolated"].asBool();
	const Eigen::Vector3f map_start_unreal(args["MAP_START_UNREAL_X"].asFloat(), args["MAP_START_UNREAL_Y"].asFloat(), args["MAP_START_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_end_unreal(args["MAP_END_UNREAL_X"].asFloat(), args["MAP_END_UNREAL_Y"].asFloat(), args["MAP_END_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_start_mesh(map_start_unreal.x() / 100.f, -map_end_unreal.y() / 100.f, map_start_unreal.z() / 100.f) ;
	const Eigen::Vector3f map_end_mesh(map_end_unreal.x() / 100.f, -map_start_unreal.y() / 100.f, map_end_unreal.z() / 100.f);
	Height_map height_map(map_start_mesh,map_end_mesh,
		args["heightmap_resolution"].asFloat(),
		args["heightmap_dilate"].asInt()
		);
	std::vector<Building> total_buildings;
	Pos_Pack current_pos = map_converter.get_pos_pack_from_unreal(
		Eigen::Vector3f(args["START_X"].asFloat(), 
			args["START_Y"].asFloat(), 
			args["START_Z"].asFloat()),  -M_PI / 2, 0);
	int cur_frame_id = 0;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> total_passed_trajectory;
	std::vector<int> trajectory_flag;
	//debug_img(std::vector<cv::Mat>{height_map.m_map});

	LOG(INFO) << "Mapping building ";
	Mapper* mapper;
	if (args["mapper"] == "gt_mapper")
		mapper = new GT_mapper(args);
	else if (args["mapper"] == "real_mapper")
		mapper = new Real_mapper(args, airsim_client);
	else
		mapper = new Virtual_mapper(args, airsim_client, color_to_mesh_name_map);
	
	Next_best_target* next_best_target;
	if (args["nbv_target"] == "Topology_decomposition")
		next_best_target = new Next_best_target_topology_exploration(map_start_mesh, map_end_mesh, 
			args["CCPP_CELL_THRESHOLD"].asInt(), mapper->m_boundary, args["ccpp_cell_distance"].asFloat(),args);
	else if (args["nbv_target"] == "Min_distance")
		next_best_target = new Next_best_target_min_distance_ccpp(map_start_mesh, map_end_mesh, args["ccpp_cell_distance"].asFloat());
	else if (args["nbv_target"] == "Order_reconstruction")
		next_best_target = new Next_best_target_order_reconstruction(map_start_mesh, map_end_mesh,
			args["CCPP_CELL_THRESHOLD"].asInt(), mapper->m_boundary, args["ccpp_cell_distance"].asFloat(), args);
	else if (args["nbv_target"] == "Random_min_distance")
		next_best_target = new Next_best_target_random_min_distance(map_start_mesh, map_end_mesh, args["ccpp_cell_distance"].asFloat());
	else if (args["nbv_target"] == "Min_max_information")
		next_best_target = new Next_best_target_min_max_information(map_start_mesh, map_end_mesh, args["ccpp_cell_distance"].asFloat());
	else if (args["nbv_target"] == "First_building_next_region")
		next_best_target = new Next_best_target_first_building_next_region(map_start_mesh, map_end_mesh, args["ccpp_cell_distance"].asFloat());
	else if (args["nbv_target"] == "Reconstruction_only")
		next_best_target = new Next_best_target_reconstruction_only(map_start_mesh, map_end_mesh, args["ccpp_cell_distance"].asFloat());
	else if (args["nbv_target"] == "Exploration_only")
		next_best_target = new Next_best_target_exploration_only(map_start_mesh, map_end_mesh, args["ccpp_cell_distance"].asFloat());
	else
		throw;
	bool with_exploration = args["with_exploration"].asBool();
	float reconstruction_length = 0.f;
	float exploration_length = 0.f;
	float max_turn = 0.f;
	int building_num_record = -1;
	int current_building_num= 0;
	float vertical_step, horizontal_step, split_min_distance;
	bool is_interpolated = false;
	std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos_direction;
	//total_passed_trajectory.push_back(std::make_pair(current_pos.pos_mesh, Eigen::Vector3f(0,0,-1)));

	while (!end) {
		LOG(INFO) << "<<<<<<<<<<<<< Frame " << cur_frame_id << " <<<<<<<<<<<<<";

		auto t = recordTime();
		mapper->get_buildings(total_buildings, current_pos, cur_frame_id, height_map);
		next_best_target->update_uncertainty(current_pos, total_buildings);
		profileTime(t, "Height map");

		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> current_trajectory;
		if(!with_interpolated||(with_interpolated&& !is_interpolated))
		{
			// Generating trajectory
			// Input: Building vectors (std::vector<Building>)
			// Output: Modified Building.trajectory and return the whole trajectory
			{
				current_trajectory = generate_trajectory(args, total_buildings, args["mapper"].asString()=="gt_mapper"? original_height_map:height_map,
					vertical_step, horizontal_step, split_min_distance);
				LOG(INFO) << "New trajectory ??!";
			}

			// Determine next position
			{
				next_pos_direction = next_best_target->determine_next_target(cur_frame_id, current_pos,
					total_buildings, with_exploration, horizontal_step / 2);
				LOG(INFO) << "Determine next position ??";
			}
			// End
			if (next_best_target->m_motion_status == Motion_status::done)
				break;

			LOG(INFO) << (boost::format("Current mode: %s. Building progress: %d/%d") % std::to_string(next_best_target->m_motion_status) % current_building_num % total_buildings.size()).str();

		}
		profileTime(t, "Generate trajectory");

		// Statics
		{
			if (cur_frame_id > 1)
			{
				float distance = (next_pos_direction.first - current_pos.pos_mesh).norm();
				if(next_best_target->m_motion_status==Motion_status::exploration || next_best_target->m_motion_status == Motion_status::final_check)
					exploration_length += distance;
				else
					reconstruction_length+= distance;
				max_turn = distance > max_turn ? distance : max_turn;
			}
			if (next_best_target->m_current_building_id != building_num_record) {
				current_building_num += 1;
				building_num_record = next_best_target->m_current_building_id;
			}
		}

		// Visualize
		if(is_viz)
		{
			viz->lock();
			viz->m_buildings = total_buildings;
			//if(next_best_target->m_motion_status==Motion_status::reconstruction)
			viz->m_current_building = next_best_target->m_current_building_id;
			viz->m_uncertainty_map.clear();
			for (const auto& item : next_best_target->sample_points) {
				int index = &item - &next_best_target->sample_points[0];
				viz->m_uncertainty_map.emplace_back(Eigen::Vector2f(item.x(), item.y()), next_best_target->region_status[index]);
			}
			viz->m_pos = current_pos.pos_mesh;
			viz->m_direction = current_pos.direction;
			//viz->m_trajectories = current_trajectory;
			viz->m_trajectories = total_passed_trajectory;
			viz->calculate_pitch();
			viz->m_is_reconstruction_status = trajectory_flag;
			//viz->m_trajectories_spline = total_passed_trajectory;
			//viz.m_polygon = next_best_target->img_polygon;
			viz->unlock();
			//override_sleep(0.1);
			//debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
		}
		profileTime(t, "Viz");

		//
		// Prepare next move
		//
		// Output: current_pos
		{
			Eigen::Vector3f direction = next_pos_direction.first - current_pos.pos_mesh;
			Eigen::Vector3f next_direction;
			Eigen::Vector3f next_pos;
			int interpolated_num = int(direction.norm() /  DRONE_STEP);
			if (direction.norm() < 2 * DRONE_STEP||!with_interpolated)
			{
				//next_direction = next_pos_direction.second.normalized();
				next_direction = next_pos_direction.second;
				next_pos = next_pos_direction.first;
				is_interpolated = false;
			}
			else
			{
				next_direction = direction.normalized();
				next_pos = current_pos.pos_mesh + next_direction * DRONE_STEP;
				Eigen::Vector2f next_2D_direction(next_pos_direction.second.x(), next_pos_direction.second.y());
				Eigen::Vector2f current_2D_direction(current_pos.direction.x(), current_pos.direction.y());
				float current_yaw = atan2(current_2D_direction.y(), current_2D_direction.x());
				float next_direction_yaw = atan2(next_2D_direction.y(), next_2D_direction.x());
				float angle_delta;
				if (abs(current_yaw - next_direction_yaw) > M_PI)
				{
					if (current_yaw > next_direction_yaw)
					{
						angle_delta = (next_direction_yaw - current_yaw + M_PI * 2) / interpolated_num;
						next_direction.x() = cos(current_yaw + angle_delta);
						next_direction.y() = sin(current_yaw + angle_delta);
					}
					else
					{
						angle_delta = (current_yaw - next_direction_yaw + M_PI * 2) / interpolated_num;
						next_direction.x() = cos(current_yaw - angle_delta + M_PI * 2);
						next_direction.y() = sin(current_yaw - angle_delta + M_PI * 2);
					}
				}
				else
				{
					if (current_yaw > next_direction_yaw)
					{
						angle_delta = (current_yaw - next_direction_yaw) / interpolated_num;
						next_direction.x() = cos(current_yaw - angle_delta);
						next_direction.y() = sin(current_yaw - angle_delta);
					}
					else
					{
						angle_delta = (next_direction_yaw - current_yaw) / interpolated_num;
						next_direction.x() = cos(current_yaw + angle_delta);
						next_direction.y() = sin(current_yaw + angle_delta);
					}
				}
				next_direction.z() = -std::sqrt(next_direction.x() * next_direction.x() + next_direction.y() * next_direction.y()) * std::tan(45.f / 180 * M_PI);
				next_direction.normalize();
				is_interpolated = true;
			}
			total_passed_trajectory.push_back(std::make_pair(next_pos, next_direction));
			std::ofstream pose("D:/test_data/" + std::to_string(cur_frame_id) + ".txt");
			pose << next_pos << next_direction;
			pose.close();
			if(next_best_target->m_motion_status==Motion_status::exploration|| next_best_target->m_motion_status == Motion_status::final_check)
				trajectory_flag.push_back(0);
			else
				trajectory_flag.push_back(1);
			float pitch = -std::atan2f(next_direction[2], std::sqrtf(next_direction[0] * next_direction[0] + next_direction[1] * next_direction[1]));
			float yaw = std::atan2f(next_direction[1], next_direction[0]);
			current_pos = map_converter.get_pos_pack_from_mesh(next_pos, yaw, pitch);
			cur_frame_id++;
		}
		profileTime(t, "Find next move");
		LOG(INFO) << "<<<<<<<<<<<<< Frame " << cur_frame_id - 1 << " done! <<<<<<<<<<<<<";
		LOG(INFO) << "";
		//if (cur_frame_id > 1000)
		//	break;

		std::vector<Rotated_box> boxes;
		if(cur_frame_id == 50 || cur_frame_id == 100 || cur_frame_id==150|| cur_frame_id == 450||cur_frame_id == 750)
		{
			for (const auto& item : total_buildings)
				boxes.push_back(item.bounding_box_3d);
			//Surface_mesh mesh = get_box_mesh(boxes);
			Surface_mesh mesh = get_rotated_box_mesh(boxes);
			CGAL::write_ply(std::ofstream("log/gradually_results/box" + std::to_string(cur_frame_id) + ".ply"), mesh);
			write_normal_path_with_flag(total_passed_trajectory, 
				"log/gradually_results/camera_normal_" + std::to_string(cur_frame_id) + ".log", 
				trajectory_flag);
		}
		
	}
	total_passed_trajectory.pop_back();

	write_unreal_path(total_passed_trajectory, "camera_after_transaction.log");
	write_normal_path(total_passed_trajectory, "camera_normal.log");
	write_smith_path(total_passed_trajectory, "camera_smith_invert_x.log");
	write_normal_path_with_flag(total_passed_trajectory, "camera_with_flag.log", trajectory_flag);
	std::vector<Rotated_box> boxes;
	for (const auto& item : total_buildings)
		boxes.push_back(item.bounding_box_3d);
	//Surface_mesh mesh = get_box_mesh(boxes);
	Surface_mesh mesh = get_rotated_box_mesh(boxes);

	CGAL::write_ply(std::ofstream("proxy.ply"), mesh);

	boxes.clear();
	// Uncertainty
	std::vector<Eigen::AlignedBox3f> boxess1;
	std::vector<cv::Vec3b> boxess1_color;
	std::vector<Eigen::AlignedBox3f> boxess2;
	std::vector<cv::Vec3b> boxess2_color;
	for (const auto& item : next_best_target->sample_points) {
		int index = &item - &next_best_target->sample_points[0];
		if (next_best_target->region_status[index] == cv::Vec3b(0, 255, 0))
		{
			boxess1.push_back(Eigen::AlignedBox3f(Eigen::Vector3f(item.x() - args["ccpp_cell_distance"].asFloat() / 2, item.y() - args["ccpp_cell_distance"].asFloat() / 2, -1),
				Eigen::Vector3f(item.x() + args["ccpp_cell_distance"].asFloat() / 2, item.y() + args["ccpp_cell_distance"].asFloat() / 2, 1)));
			boxess1_color.push_back(next_best_target->region_status[index]);
		}
		else
		{
			boxess2_color.push_back(next_best_target->region_status[index]);
			boxess2.push_back(Eigen::AlignedBox3f(Eigen::Vector3f(item.x() - args["ccpp_cell_distance"].asFloat() / 2, item.y() - args["ccpp_cell_distance"].asFloat() / 2, -1),
				Eigen::Vector3f(item.x() + args["ccpp_cell_distance"].asFloat() / 2, item.y() + args["ccpp_cell_distance"].asFloat() / 2, 1)));
		}
	}
	get_box_mesh_with_colors(boxess1, boxess1_color, "uncertainty_map1.obj");
	get_box_mesh_with_colors(boxess2, boxess2_color, "uncertainty_map2.obj");
	
	LOG(ERROR) <<"Total path num: "<< total_passed_trajectory.size();
	LOG(ERROR) <<"Total path length: "<< evaluate_length(total_passed_trajectory);
	LOG(ERROR) <<"Total exploration length: "<< exploration_length;
	LOG(ERROR) <<"Total reconstruction length: "<< reconstruction_length;
	LOG(ERROR) <<"Max_turn: "<< max_turn;
	LOG(ERROR) << "Vertical step: " << vertical_step;
	LOG(ERROR) << "Horizontal step: " << horizontal_step;
	LOG(ERROR) << "Split minimum distance: " << split_min_distance;
	LOG(ERROR) << "Write trajectory done!";
	height_map.save_height_map_tiff("height_map.tiff");
	debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});

	total_passed_trajectory = ensure_global_safe(total_passed_trajectory, original_height_map, args["safe_distance"].asFloat(), mapper->m_boundary);

	// Change focus point into direction
	for (auto& item : total_passed_trajectory)
	{
		item.second = (item.second - item.first);
		if (item.second.z() > 0)
			item.second.z() = 0;
		item.second = item.second.normalized();
	}

	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> safe_global_trajectory;
	if (args["output_waypoint"].asBool())
	{

		safe_global_trajectory = simplify_path_reduce_waypoints(total_passed_trajectory);
		write_wgs_path(args, safe_global_trajectory, "./log/wgs_log/");
		LOG(ERROR) << "Total waypoint length: " << evaluate_length(safe_global_trajectory);
	}
	
	{
		viz->lock();
		viz->m_buildings = total_buildings;
		viz->m_pos = total_passed_trajectory[0].first;
		viz->m_direction = total_passed_trajectory[0].second;
		viz->m_trajectories.clear();
		if (args["output_waypoint"].asBool())
		{
			viz->m_trajectories = safe_global_trajectory;
		}
		else
		{
			viz->m_trajectories = total_passed_trajectory;
		}
		viz->m_uncertainty_map.clear();
		for (const auto& item : next_best_target->sample_points) {
			int index = &item - &next_best_target->sample_points[0];
			viz->m_uncertainty_map.emplace_back(Eigen::Vector2f(item.x(), item.y()), next_best_target->region_status[index]);
		}
		viz->unlock();
		//override_sleep(100);
		//debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
	}
	debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});

	return 0;
}
