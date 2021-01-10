#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
//#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <CGAL/cluster_point_set.h>
#include <CGAL/random_selection.h>
#include <CGAL/Boolean_set_operations_2.h>
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
#include "common_util.h"


//Path
boost::filesystem::path log_root("log");
const Eigen::Vector3f UNREAL_START(0.f, 0.f, 0.f);
//Camera
const cv::Vec3b BACKGROUND_COLOR(57,181,55);
const cv::Vec3b SKY_COLOR(161, 120, 205);
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
			cv::Rect2f rect = cv::boundingRect(pixel_points);
			cv::rectangle(seg, rect, cv::Scalar(0, 0, 255));
			size_t id = &pixel_points - &*bboxes_points.begin();
			v_buildings[id].bounding_box_2d = CGAL::Bbox_2(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
			v_buildings[id].segmentation_color = v_color_map[id];
		}
		//debug_img(std::vector{ seg });
	}
};

class Synthetic_SLAM {
public:

	Synthetic_SLAM() {

	}

	void get_points(const std::map<std::string, cv::Mat>& v_img,const std::vector<cv::Vec3b>& v_color_map, std::vector<Building>& v_buildings) {
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat rgb = v_img.at("rgb").clone();
		auto orb = cv::ORB::create(3000);
		orb->detect(rgb, keypoints, v_img.at("roi_mask"));
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
				throw "";
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

enum Motion_status { initialization,exploration,reconstruction,done};

class Next_best_target {
public:
	Motion_status m_motion_status;
	int m_current_building_id = -1;
	
	Next_best_target()
	{
		m_motion_status = Motion_status::initialization;
	}

	virtual void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) = 0;

	virtual void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) = 0;
	
	virtual std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration,float v_threshold) = 0;
};

class Next_best_target_min_max_information:public Next_best_target
{
public:
	float DISTANCE_THRESHOLD = 20.f;
	float VISIBLE_DISTANCE = DISTANCE_THRESHOLD * 2;
	
	std::vector<CGAL::Point_2<K>> sample_points;
	std::vector<Region_status> region_status;

	int m_current_exploration_id = -1;
	
	Next_best_target_min_max_information(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh):Next_best_target()
	{
		for (float y = v_map_start_mesh.y(); y < v_map_end_mesh.y(); y += DISTANCE_THRESHOLD)
			for (float x = v_map_start_mesh.x(); x < v_map_end_mesh.x(); x += DISTANCE_THRESHOLD)
				sample_points.push_back(CGAL::Point_2<K>(x, y));
		region_status.resize(sample_points.size(), Unobserved);
	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings,bool with_exploration) override
	{
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		{
			// Find with distance
			for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
				if (v_buildings[i_building].passed_trajectory.size() == 0)
					untraveled_buildings.emplace_back(i_building, -1);
			}
			if(with_exploration)
				for (int i_point = 0; i_point < sample_points.size(); ++i_point)
					if (region_status[i_point]==Unobserved)
						untraveled_buildings.emplace_back(-1, i_point);

			std::nth_element(untraveled_buildings.begin(),
				untraveled_buildings.begin() + std::min(5, (int)untraveled_buildings.size()),
				untraveled_buildings.end(),
				[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
				float distance1, distance2;
				if (b1.origin_index_in_building_vector != -1)
					distance1 = (v_buildings[b1.origin_index_in_building_vector].bounding_box_3d.center() - v_cur_pos.pos_mesh).norm();
				else
					distance1 = (Eigen::Vector2f(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y()) - Eigen::Vector2f(sample_points[b1.origin_index_in_untraveled_pointset].x(), sample_points[b1.origin_index_in_untraveled_pointset].y())).norm();
				if (b2.origin_index_in_building_vector != -1)
					distance2 = (v_buildings[b2.origin_index_in_building_vector].bounding_box_3d.center() - v_cur_pos.pos_mesh).norm();
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

		// Find next building with higher information gain
		std::vector<float> information_gain(untraveled_buildings.size(), 0.f);
		{
			for (int i_building = 0; i_building < untraveled_buildings.size(); ++i_building) {
				if (untraveled_buildings[i_building].origin_index_in_building_vector == -1) {
					information_gain[i_building] = 1;
					continue;
				}
				for (int i_point = 0; i_point < region_status.size(); i_point++) {
					if (region_status[i_point]!=Unobserved)
						continue;
					const auto& original_bounding_box_3d = v_buildings[untraveled_buildings[i_building].origin_index_in_building_vector].bounding_box_3d;
					const CGAL::Bbox_2 box(
						original_bounding_box_3d.min().x() - DISTANCE_THRESHOLD, original_bounding_box_3d.min().y() - DISTANCE_THRESHOLD,
						original_bounding_box_3d.max().x() + DISTANCE_THRESHOLD, original_bounding_box_3d.max().y() + DISTANCE_THRESHOLD
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
		if(untraveled_buildings[next_target_id].origin_index_in_building_vector==-1)
		{
			m_motion_status = Motion_status::exploration;
			m_current_exploration_id = untraveled_buildings[next_target_id].origin_index_in_untraveled_pointset;
		}
		else
		{
			m_motion_status = Motion_status::reconstruction;
			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
		}
		return ;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override
	{
		Eigen::Vector2f cur_point_cgal(v_cur_pos.pos_mesh.x(), v_cur_pos.pos_mesh.y());
		int nearest_region_id=std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
			return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
		})- sample_points.begin();
		region_status[nearest_region_id] = Free;
		
		for (int i_point = 0; i_point < region_status.size(); i_point++) {
			if (region_status[i_point]!= Unobserved)
				continue;
			const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

			for (const auto& item_building : v_buildings) {
				Eigen::AlignedBox2f box(Eigen::Vector2f(item_building.bounding_box_3d.min().x(), item_building.bounding_box_3d.min().y()),
					Eigen::Vector2f(item_building.bounding_box_3d.max().x(), item_building.bounding_box_3d.max().y()));
				if (p.x() > item_building.bounding_box_3d.min().x() && p.x() < item_building.bounding_box_3d.max().x() &&
					p.y() > item_building.bounding_box_3d.min().y() && p.y() < item_building.bounding_box_3d.max().y()) {
					region_status[i_point] = Occupied;
					break;
				}
			}
			if(region_status[i_point] != Unobserved)
				continue;

			Eigen::Vector2f direction = p - cur_point_cgal;
			float squared_distance = direction.norm();

			if (squared_distance > VISIBLE_DISTANCE)
				continue;

			if (direction.normalized().dot(Eigen::Vector2f(v_cur_pos.direction.x(), v_cur_pos.direction.y()).normalized()) < M_SQRT2 / 2)
				continue;

			region_status[i_point] = Free;
		}
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status == Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if (m_motion_status == Motion_status::exploration) {
			Eigen::Vector3f next_pos(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y(), v_cur_pos.pos_mesh.z());
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

class Next_best_target_topology_exploration: public Next_best_target {
public:
	float DISTANCE_THRESHOLD = 20.f;
	float VISIBLE_DISTANCE = DISTANCE_THRESHOLD * 2;

	std::vector<Eigen::AlignedBox2f> topology;
	std::vector<cv::Vec3b> topology_viz_color;

	Eigen::Vector3f m_map_start;
	Eigen::Vector3f m_map_end;

	int m_current_ccpp_trajectory_id;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_ccpp_trajectory;

	Next_best_target_topology_exploration(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh):
		Next_best_target(), m_map_start(v_map_start_mesh), m_map_end(v_map_end_mesh)
	{
		topology.emplace_back(Eigen::Vector2f(m_map_start.x(), m_map_start.y()), 
			Eigen::Vector2f(m_map_start.x()+1, m_map_start.y()+1));
		topology_viz_color= get_color_table_bgr();
	}

	void get_next_target(int frame_id,const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override{
		m_ccpp_trajectory.clear();
		m_current_ccpp_trajectory_id = 0;
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].passed_trajectory.size() == 0)
				untraveled_buildings.emplace_back(i_building, -1);
		}
		if (untraveled_buildings.size() == 0)
		{
			m_motion_status = Motion_status::done;
			return;
		}

		// Find nearest building to existing polygon
		int id_building=std::min_element(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const Next_target& b1, const Next_target& b2) {
			const Eigen::Vector2f building_center1(v_buildings[b1.origin_index_in_building_vector].bounding_box_3d.center().x(), v_buildings[b1.origin_index_in_building_vector].bounding_box_3d.center().y());
				const Eigen::Vector2f building_center2(v_buildings[b2.origin_index_in_building_vector].bounding_box_3d.center().x(), v_buildings[b2.origin_index_in_building_vector].bounding_box_3d.center().y());
				return point_box_distance_eigen(building_center1, topology.back()) < point_box_distance_eigen(building_center2, topology.back());
		}) - untraveled_buildings.begin();

		// Add to topology graph
		const Eigen::Vector3f& target_center = v_buildings[untraveled_buildings[id_building].origin_index_in_building_vector].bounding_box_3d.center();
		Eigen::AlignedBox2f cur_box(Eigen::Vector2f(m_map_start.x(), m_map_start.y()), Eigen::Vector2f(target_center.x(), target_center.y()));
		
		// Perform complete coverage path planning in this topology
		cv::Mat map((m_map_end.y() - m_map_start.y()) / DISTANCE_THRESHOLD + 1,
			(m_map_end.x() - m_map_start.x()) / DISTANCE_THRESHOLD + 1,
			CV_8UC3,cv::Scalar(0,0,0));

		topology.push_back(cur_box);
		topology_viz_color.push_back(cv::Vec3b((unsigned char)(rand() / (RAND_MAX / 255.0)), (unsigned char)(rand() / (RAND_MAX / 255.0)), (unsigned char)(rand() / (RAND_MAX / 255.0))));
		for (int y = 0; y < map.rows; y++) {
			for (int x = 0; x < map.cols; x++) {
				for(int i_topology= topology.size()-1; i_topology>=0; i_topology--)
				{
					if (inside_box(Eigen::Vector2f(x * DISTANCE_THRESHOLD + m_map_start.x(),
						y * DISTANCE_THRESHOLD + m_map_start.y()), topology[i_topology])) {
						map.at<cv::Vec3b>(y, x)= topology_viz_color[i_topology%(topology_viz_color.size()-1)];
					}
				}
			}
		}
		cv::imwrite(std::to_string(frame_id) + "_global_map.png", map);
		if(with_exploration)
			m_ccpp_trajectory = perform_ccpp(map, v_cur_pos.pos_mesh, target_center);

		// return trajectory
		//debug_img(std::vector<cv::Mat>{map});
		m_current_building_id = untraveled_buildings[id_building].origin_index_in_building_vector;
		m_motion_status = Motion_status::exploration;
		return ;
	}

	void update_uncertainty(const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings) override {
		
	}

	std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration,float v_threshold) override
	{
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos;

		if (m_motion_status==Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if(m_motion_status == Motion_status::exploration)
		{
			if (m_current_ccpp_trajectory_id >= m_ccpp_trajectory.size()) {
				m_motion_status = Motion_status::reconstruction;
			}
			else
				next_pos=m_ccpp_trajectory[m_current_ccpp_trajectory_id++];
		}
		if(m_motion_status == Motion_status::reconstruction)
		{
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
				next_pos = determine_next_target(v_frame_id,v_cur_pos,v_buildings,with_exploration, v_threshold);
			}
			else
			{
				auto it_min_distance = std::min_element(
					unpassed_trajectory.begin(), unpassed_trajectory.end(),
					[&v_cur_pos](const auto& t1, const auto& t2) {
					return (t1.first - v_cur_pos.pos_mesh).norm() < (t2.first - v_cur_pos.pos_mesh).norm();
				});
				next_pos = *it_min_distance;
				passed_trajectory.push_back(next_pos);
			}
		}
		return next_pos;
	}

};

class Synthetic_building {
public:

	Synthetic_building() {

	}

	void get_buildings() {
		/*
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

		else {
				const float distance = 30.f;
				Eigen::AlignedBox3f drone_box(current_pos.pos_mesh - Eigen::Vector3f(distance, distance, distance), current_pos.pos_mesh + Eigen::Vector3f(distance, distance, distance));
				for (int i_building = 0; i_building < fake_buildings.size(); ++i_building) {
					if (fake_buildings[i_building].bounding_box_3d.intersects(drone_box))
						current_buildings.push_back(fake_buildings[i_building]);
				}
				num_building_current_frame = current_buildings.size();
			}
		*/
	}
};

class Mapper
{
public:
	Json::Value m_args;
	
	Mapper(const Json::Value& v_args):m_args(v_args){};
	virtual void get_buildings(std::vector<Building>& v_buildings,
		const Pos_Pack& v_current_pos,
		const int v_cur_frame_id)=0;
};

class GT_mapper:public Mapper
{
public:
	std::vector<Building> m_buildings;
	GT_mapper(const Json::Value& args): Mapper(args)
	{
		CGAL::Point_set_3<Point_3, Vector_3> original_point_cloud;
		CGAL::read_ply_point_set(std::ifstream(args["model_path"].asString()), original_point_cloud);
		CGAL::Point_set_3<Point_3, Vector_3> point_cloud(original_point_cloud);
		Height_map height_map(point_cloud, args["heightmap_resolution"].asFloat());
		height_map.save_height_map_png("map.png", args["HEIGHT_CLIP"].asFloat());
		height_map.save_height_map_tiff("map.tiff");

		// Delete ground planes
		{
			for (int idx = point_cloud.size() - 1; idx >= 0; idx--)
			{
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
			m_buildings.resize(nb_clusters);

			Point_set::Property_map<unsigned char> red = point_cloud.add_property_map<unsigned char>("red", 0).first;
			Point_set::Property_map<unsigned char> green = point_cloud
			                                               .add_property_map<unsigned char>("green", 0).first;
			Point_set::Property_map<unsigned char> blue = point_cloud.add_property_map<unsigned char>("blue", 0).first;
			for (Point_set::Index idx : point_cloud)
			{
				// One color per cluster
				int cluster_id = cluster_map[idx];
				CGAL::Random rand(cluster_id);
				red[idx] = rand.get_int(64, 192);
				green[idx] = rand.get_int(64, 192);
				blue[idx] = rand.get_int(64, 192);

				Building& current_building = m_buildings[cluster_id];
				current_building.points_world_space.insert(point_cloud.point(idx));
			}
		}
		for (int i_building_1 = 0; i_building_1 < m_buildings.size(); ++i_building_1)
		{
			m_buildings[i_building_1].bounding_box_3d = get_bounding_box(m_buildings[i_building_1].points_world_space);
			m_buildings[i_building_1].bounding_box_3d.min().z() -= args["HEIGHT_CLIP"].asFloat();
			m_buildings[i_building_1].boxes.push_back(m_buildings[i_building_1].bounding_box_3d);
		}
	}

	void get_buildings(std::vector<Building>& v_buildings,
		const Pos_Pack& v_current_pos,
		const int v_cur_frame_id) override
	{
		if (v_buildings.size() == 0)
			v_buildings = m_buildings;
		return;
	}
};

class Virtual_mapper:public Mapper {
public:
	Unreal_object_detector* m_unreal_object_detector;
	Synthetic_SLAM* m_synthetic_SLAM;
	Airsim_tools* m_airsim_client;
	
	Virtual_mapper(const Json::Value& args, Airsim_tools* v_airsim_client)
	: Mapper(args), m_airsim_client(v_airsim_client){
		m_unreal_object_detector =new Unreal_object_detector;
		m_synthetic_SLAM = new Synthetic_SLAM;
	}
	
	void get_buildings(std::vector<Building>& v_buildings,
		const Pos_Pack& v_current_pos,
		const int v_cur_frame_id) override {
		std::vector<Building> current_buildings;
		int num_building_current_frame;
		// Get current image and pose
		// Input: 
		// Output: Image(cv::Mat), Camera matrix(Pos_pack)
		std::map<std::string, cv::Mat> current_image;
		{
			m_airsim_client->adjust_pose(v_current_pos);
			current_image = m_airsim_client->get_images();
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
					item_building.bounding_box_3d.min()[2] = 0;
					item_building.bounding_box_3d.max()[2] = final_height;
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
					if (item_building.segmentation_color== item_current_building.segmentation_color) {
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
					}
				}
			}
			for (int i = 0; i < need_register.size(); ++i) {
				if (need_register[i]) {
					v_buildings.push_back(current_buildings[i]);
				}
			}
			LOG(INFO) << "Building BBox update: DONE!";
		}
	}
};


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
		if (boost::filesystem::exists(log_root))
			boost::filesystem::remove_all(log_root);
		boost::filesystem::create_directories(log_root);
		boost::filesystem::create_directories(log_root/"img");
		boost::filesystem::create_directories(log_root/"seg");
		boost::filesystem::create_directories(log_root/"point_world");
		
		map_converter.initDroneStart(UNREAL_START);
		INTRINSIC << 400, 0, 400, 0, 400, 400, 0, 0, 1;

		if(args["mapper"].asString()=="Virtual_mapper")
		{
			airsim_client = new Airsim_tools(UNREAL_START);
			airsim_client->reset_color("building");
		}
		LOG(INFO) << "Initialization done";
	}
	
	// Some global structure
	bool end = false;
	const Eigen::Vector3f map_start_unreal(args["MAP_START_UNREAL_X"].asFloat(), args["MAP_START_UNREAL_Y"].asFloat(), args["MAP_START_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_end_unreal(args["MAP_END_UNREAL_X"].asFloat(), args["MAP_END_UNREAL_Y"].asFloat(), args["MAP_END_UNREAL_Z"].asFloat());
	const Eigen::Vector3f map_start_mesh(map_start_unreal.x() / 100.f, -map_end_unreal.y() / 100.f, map_start_unreal.z() / 100.f) ;
	const Eigen::Vector3f map_end_mesh(map_end_unreal.x() / 100.f, -map_start_unreal.y() / 100.f, map_end_unreal.z() / 100.f);
	Height_map height_map(map_start_mesh,map_end_mesh,args["heightmap_resolution"].asFloat());
	std::vector<Building> total_buildings;
	Pos_Pack current_pos = map_converter.get_pos_pack_from_unreal(
		Eigen::Vector3f(args["START_X"].asFloat(), 
			args["START_Y"].asFloat(), 
			args["START_Z"].asFloat()),  -M_PI / 2, 0);
	int cur_frame_id = 0;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> total_passed_trajectory;

	Mapper* mapper;
	if (args["mapper"] == "gt_mapper")
		mapper = new GT_mapper(args);
	else
		mapper = new Virtual_mapper(args,airsim_client);
	
	Next_best_target* next_best_target;
	if(args["nbv_target"] == "Topology_decomposition")
		next_best_target=new Next_best_target_topology_exploration(map_start_mesh, map_end_mesh);
	else
		next_best_target = new Next_best_target_min_max_information(map_start_mesh, map_end_mesh);

	bool with_exploration = args["with_exploration"].asBool();
	while (!end) {
		LOG(INFO) << "<<<<<<<<<<<<< Frame " << cur_frame_id << " <<<<<<<<<<<<<";
		
		mapper->get_buildings(total_buildings, current_pos, cur_frame_id);

		// Update height map
		for (auto& item_building : total_buildings) {
			height_map.update(item_building.bounding_box_3d);
		}
		next_best_target->update_uncertainty(current_pos, total_buildings);
		//debug_img(std::vector{ height_map.m_map });


		// Generating trajectory
		// Input: Building vectors (std::vector<Building>)
		// Output: Modified Building.trajectory and return the whole trajectory
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> current_trajectory;
		{
			Trajectory_params params;
			params.view_distance = args["BOUNDS_MIN"].asFloat();
			params.z_down_bounds = args["Z_DOWN_BOUND"].asFloat();
			params.z_up_bounds = args["Z_UP_BOUNDS"].asFloat();
			params.xy_angle = args["xy_angle"].asFloat();
			params.with_continuous_height = args["with_continuous_height"].asBool();
			params.with_ray_test = args["with_ray_test"].asBool();
			params.double_flag = args["double_flag"].asBool();
			params.step = args["step"].asFloat();
			current_trajectory = generate_trajectory(params,total_buildings,height_map, params.z_up_bounds);
			LOG(INFO) << "New trajectory ¡Ì!";
		}

		// Determine next position
		std::pair<Eigen::Vector3f, Eigen::Vector3f> next_pos_direction;
		{
			next_pos_direction=next_best_target->determine_next_target(cur_frame_id, current_pos, 
				total_buildings, with_exploration, args["step"].asFloat()-.5f);
			LOG(INFO) << "Determine next position ¡Ì";
		}

		// End
		if (next_best_target->m_motion_status==Motion_status::done)
			break;
		
		total_passed_trajectory.push_back(next_pos_direction);

		// Visualize
		{
			viz.lock();
			viz.m_buildings = total_buildings;
			if(next_best_target->m_motion_status==Motion_status::reconstruction)
				viz.m_current_building = next_best_target->m_current_building_id;
			viz.m_uncertainty_map.clear();
			//for (const auto& item : next_best_target->sample_points)
			//{
			//	int index = &item - &next_best_target->sample_points[0];
			//	viz.m_uncertainty_map.emplace_back(Eigen::Vector2f(item.x(), item.y()), next_best_target->region_status[index]);
			//}
			viz.m_pos = current_pos.pos_mesh;
			viz.m_direction = current_pos.direction;
			viz.m_trajectories = total_passed_trajectory;
			//viz.m_polygon = next_best_target->img_polygon;
			viz.unlock();
			//override_sleep(100);
			//debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
		}

		//
		// Prepare next move
		//
		// Output: current_pos
		{
			LOG(INFO) << (boost::format("Current mode: %s. Building progress: %d/%d")%std::to_string(next_best_target->m_motion_status)% next_best_target->m_current_building_id%total_buildings.size()).str();

			Eigen::Vector3f next_direction = next_pos_direction.second.normalized();
			Eigen::Vector3f next_pos = next_pos_direction.first;
			float pitch = -std::atan2f(next_direction[2], std::sqrtf(next_direction[0] * next_direction[0] + next_direction[1] * next_direction[1]));
			float yaw = std::atan2f(next_direction[1], next_direction[0]);
			current_pos = map_converter.get_pos_pack_from_mesh(next_pos, yaw, pitch);
			cur_frame_id++;
		}
		LOG(INFO) << "<<<<<<<<<<<<< Frame " << cur_frame_id - 1 << " done! <<<<<<<<<<<<<";
		LOG(INFO) << "";

	}
	debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});

	write_unreal_path(total_passed_trajectory, "camera_after_transaction.log");
	write_normal_path(total_passed_trajectory, "camera_normal.log");
	write_smith_path(total_passed_trajectory, "camera_smith_invert_x.log");
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> simplified_trajectory = simplify_path_reduce_waypoints(total_passed_trajectory);
	
	LOG(INFO) <<"Total path num: "<< total_passed_trajectory.size();
	LOG(INFO) <<"Total path length: "<< evaluate_length(total_passed_trajectory);
	LOG(INFO) <<"Total path length: "<< evaluate_length(simplified_trajectory);
	LOG(INFO) << "Write trajectory done!";

	{
		viz.lock();
		viz.m_buildings = total_buildings;
		viz.m_pos = total_passed_trajectory[0].first;
		viz.m_direction = total_passed_trajectory[0].second;
		viz.m_trajectories.clear();
		viz.m_trajectories_spline= simplified_trajectory;
		viz.unlock();
		//override_sleep(100);
		//debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});
	}
	debug_img(std::vector<cv::Mat>{height_map.m_map_dilated});

	return 0;
}
