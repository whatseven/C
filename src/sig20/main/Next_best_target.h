#pragma once

#include "map_util.h"
#include "building.h"
#include "cgal_tools.h"
#include "common_util.h"

enum Motion_status { initialization, exploration, reconstruction, done, final_check, reconstruction_in_exploration };

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

	Next_best_target(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh, float v_ccpp_cell_distance) :m_map_start(v_map_start_mesh), m_map_end(v_map_end_mesh)
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

	virtual Viewpoint determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) = 0;
};


class Next_best_target_topology_exploration : public Next_best_target {
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
	int m_isFirst = false;
	std::vector<Viewpoint> m_ccpp_trajectory;
	std::queue<Viewpoint> m_exploration_point;

	float memory_y = -1.f;
	int dummy1 = 0;
	int dummy2 = 0;
	float dummy3 = 0;

	Next_best_target_topology_exploration(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,
		int v_CCPP_CELL_THRESHOLD, const Polygon_2& v_boundary, float v_ccpp_cell_distance, const Json::Value& v_arg) :CCPP_CELL_THRESHOLD(v_CCPP_CELL_THRESHOLD), m_arg(v_arg),
		Next_best_target(v_map_start_mesh, v_map_end_mesh, v_ccpp_cell_distance)
	{
		//color_reconstruction = region_viz_color[2];
		//color_occupied = region_viz_color[1];
		//color_unobserved = region_viz_color[0];
		color_reconstruction = cv::Vec3b(0, 255, 0);
		color_occupied = cv::Vec3b(0, 255, 0);
		color_unobserved = cv::Vec3b(205, 205, 209);
		this->m_boundary = v_boundary;
		region_status.clear();
		region_status.resize(sample_points.size(), color_unobserved);

		if (v_boundary.size() != 0) {
			for (int i_sample_point = 0; i_sample_point < sample_points.size(); ++i_sample_point) {
				if (v_boundary.bounded_side(sample_points[i_sample_point]) != CGAL::ON_BOUNDED_SIDE) {
					//LOG(INFO) << v_boundary.is_simple();
					region_status[i_sample_point] = color_occupied;
				}
			}
		}

		m_current_color_id = 0;
		m_current_ccpp_trajectory_id = 0;
		//region_status[0] = region_viz_color[m_current_color_id];

		//topology.emplace_back(Eigen::Vector2f(m_map_start.x(), m_map_start.y()), 
		//	Eigen::Vector2f(m_map_start.x()+1, m_map_start.y()+1));
		//topology_viz_color= get_color_table_bgr();
	}

	bool get_ccpp_trajectory(const Eigen::Vector3f& v_cur_pos, const Building& v_building, int v_ccpp_threshold)
	{
		const Eigen::AlignedBox3f& cur_box_3 = v_building.bounding_box_3d.box;
		// Ensure even
		Eigen::AlignedBox2f last_topology;
		Eigen::Vector2i last_map_pos;
		int x_add = 0, y_add = 0;
		if (topology.size() == 0)
		{
			last_map_pos = Eigen::Vector2i(0, 0);
			Eigen::Vector2i now_map_pos(int((cur_box_3.max().x() - m_map_start.x()) / DISTANCE_THRESHOLD), int((cur_box_3.max().y() - m_map_start.y()) / DISTANCE_THRESHOLD));
			if (std::abs(now_map_pos.x() - last_map_pos.x()) % 2 == 0)
				x_add = 1;
			if (std::abs(now_map_pos.y() - last_map_pos.y()) % 2 == 0)
				y_add = 1;
		}
		else
		{
			last_topology = topology.at(topology.size() - 1);
			last_map_pos = Eigen::Vector2i(int((last_topology.max().x() - m_map_start.x()) / DISTANCE_THRESHOLD), int((last_topology.max().y() - m_map_start.y()) / DISTANCE_THRESHOLD));
			Eigen::Vector2i now_map_pos(int((cur_box_3.max().x() - m_map_start.x()) / DISTANCE_THRESHOLD), int((cur_box_3.max().y() - m_map_start.y()) / DISTANCE_THRESHOLD));
			if (std::abs(now_map_pos.x() - last_map_pos.x()) % 2 == 1)
				x_add = 1;
			if (std::abs(now_map_pos.y() - last_map_pos.y()) % 2 == 1)
				y_add = 1;
		}

		float bbox_x = std::min(cur_box_3.max().x() + DISTANCE_THRESHOLD * x_add, m_map_end.x());
		float bbox_y = std::min(cur_box_3.max().y() + DISTANCE_THRESHOLD * y_add, m_map_end.y());

		Eigen::AlignedBox2f cur_box_2(Eigen::Vector2f(m_map_start.x(), m_map_start.y()),
			Eigen::Vector2f(bbox_x, bbox_y));

		Eigen::Vector3f next_point = v_building.bounding_box_3d.box.max();

		Eigen::Vector3f t1 = (v_cur_pos - m_map_start) / DISTANCE_THRESHOLD;

		Eigen::Vector3f t2 = (next_point - m_map_start) / DISTANCE_THRESHOLD;

		//Eigen::Vector2i start_pos_on_map(t1.x(), t1.y());
		Eigen::Vector2i start_pos_on_map = last_map_pos;
		Eigen::Vector2i end_pos_on_map(t2.x() + x_add, t2.y() + y_add);
		end_pos_on_map.x() = (end_pos_on_map.x() * DISTANCE_THRESHOLD + m_map_start.x() < m_map_end.x()) ? end_pos_on_map.x() : (m_map_end.x() - m_map_start.x()) / DISTANCE_THRESHOLD;
		end_pos_on_map.y() = (end_pos_on_map.y() * DISTANCE_THRESHOLD + m_map_start.y() < m_map_end.y()) ? end_pos_on_map.y() : (m_map_end.y() - m_map_start.y()) / DISTANCE_THRESHOLD;
		if (end_pos_on_map.x() <= start_pos_on_map.x() && end_pos_on_map.y() <= start_pos_on_map.y())
			return false;
		if (m_motion_status != Motion_status::final_check)
		{
			//cur_box_2.max().x() += 2*DISTANCE_THRESHOLD;
			//cur_box_2.max().y() += 2 * DISTANCE_THRESHOLD;
			//next_point.x() += 2 * DISTANCE_THRESHOLD;
			//next_point.y() += 2 * DISTANCE_THRESHOLD;
		}

		cv::Mat ccpp_map((m_map_end.y() - m_map_start.y()) / DISTANCE_THRESHOLD,
			(m_map_end.x() - m_map_start.x()) / DISTANCE_THRESHOLD,
			CV_8UC1, cv::Scalar(0));
		int num_ccpp_cell = 0;
		for (int id_region = 0; id_region < sample_points.size(); ++id_region) {
			const auto& point = sample_points[id_region];
			int y = (point.y() - m_map_start.y()) / DISTANCE_THRESHOLD;
			int x = (point.x() - m_map_start.x()) / DISTANCE_THRESHOLD;
			Eigen::Vector2f pos(point.x(), point.y());
			if (inside_box(pos, cur_box_2) && region_status[id_region] == color_unobserved) {
				//if (inside_box(pos, cur_box_2)) {
				bool already_traveled = false;
				for (const auto& item : topology) {
					if (inside_box(pos, item))
						already_traveled = true;
				}
				if (!already_traveled)
				{
					ccpp_map.at<cv::uint8_t>(y, x) = 255;
					num_ccpp_cell += 1;
				}

			}
		}
		dummy2 += num_ccpp_cell;
		topology.push_back(cur_box_2);
		//cv::imwrite(std::to_string(dummy1++)+"_ccpp_map.png", ccpp_map);
		// Find the nearest view point

		cv::Mat start_end = ccpp_map.clone();
		//start_end.setTo(0);
		if (topology.size() == 1)
			start_end.at<cv::uint8_t>(start_pos_on_map.y(), start_pos_on_map.x()) = 255;
		start_end.at<cv::uint8_t>(end_pos_on_map.y(), end_pos_on_map.x()) = 255;
		//debug_img(std::vector<cv::Mat>{ccpp_map, start_end});

		std::vector<Eigen::Vector2i> map_trajectory = perform_ccpp(start_end,
			start_pos_on_map, end_pos_on_map, m_arg["CCPP_Obstacle_weight"].asFloat());

		//std::cout << "  " << std::endl;
		//cv::Mat viz_ccpp = ccpp_map.clone();

		float iter_trajectory = 0;
		for (const Eigen::Vector2i& item : map_trajectory) {
			// todo: invert x,y!!!!!!!!!!!!!
			//viz_ccpp.at<cv::uint8_t>(item.x(), item.y()) = iter_trajectory++ * 255. / map_trajectory.size();
			Eigen::Vector3f t3 = m_map_start + DISTANCE_THRESHOLD * Eigen::Vector3f(item.y(), item.x(), 0.f);
			t3.z() = 100;
			m_ccpp_trajectory.emplace_back(
				t3,
				Eigen::Vector3f(0, 0, -1)
			);
			//if (iter_trajectory < map_trajectory.size())
			//	dummy3 += (map_trajectory[iter_trajectory] - map_trajectory[iter_trajectory - 1]).norm();
		}
		// Bug:
		//if(m_boundary.bounded_side(Point_2(
		//	m_ccpp_trajectory[m_ccpp_trajectory.size() - 1].first.x(),
		//	m_ccpp_trajectory[m_ccpp_trajectory.size() - 1].first.y()
		//)) != CGAL::ON_BOUNDED_SIDE)
		//	m_ccpp_trajectory.pop_back();
		next_point.z() = 100;
		//m_ccpp_trajectory.emplace_back(
		//	next_point,
		//	Eigen::Vector3f(0, 0, -1)
		//);
		//cv::imwrite("log/ccpp_map/"+std::to_string(dummy1++) + "_ccpp.png", viz_ccpp);
		// BUG
		return true;
	}

	Building default_expand()
	{
		int num_already_travelled_cell = 0;
		int max_x = 0;
		int max_x_id = -1;
		int max_y = 0;
		int max_y_id = -1;

		for (int id_region = 0; id_region < sample_points.size(); ++id_region) {
			const auto& point = sample_points[id_region];
			Eigen::Vector2f pos(point.x(), point.y());
			if (region_status[id_region] == color_unobserved) {
				bool already_traveled = false;
				for (const auto& item : topology) {
					if (inside_box(pos, item))
						already_traveled = true;
					if (max_x <= item.max().x() - m_map_start.x())
					{
						max_x = item.max().x() - m_map_start.x();
						max_x_id = &item - &topology[0];
					}
					if (max_y <= item.max().y() - m_map_start.y())
					{
						max_y = item.max().y() - m_map_start.y();
						max_y_id = &item - &topology[0];
					}
				}
				if (!already_traveled)
				{
					num_already_travelled_cell += 1;
				}
			}
		}

		Building fake_building;
		if (num_already_travelled_cell > 0.5 * sample_points.size() && (m_map_start.x() + max_x + DISTANCE_THRESHOLD * CCPP_CELL_THRESHOLD < m_map_end.x()
			|| m_map_start.y() + max_y + DISTANCE_THRESHOLD * CCPP_CELL_THRESHOLD < m_map_end.y())) // Expand in default size
		{
			m_motion_status = Motion_status::exploration;

			if (max_x_id != max_y_id && false) // Current traveled is not forming a rectangle 
			{
				Eigen::Vector3f max_point(m_map_start.x() + max_x, m_map_start.y() + max_y, 100);
				fake_building.bounding_box_3d = Eigen::AlignedBox3f(max_point - Eigen::Vector3f(1, 1, 1), max_point);
				fake_building.trajectory.emplace_back(max_point, Eigen::Vector3f(0, 0, 1));
			}
			else // Current traveled is forming a rectangle 
			{
				max_x = (m_map_start.x() + max_x + DISTANCE_THRESHOLD * CCPP_CELL_THRESHOLD < m_map_end.x()) ? max_x : (m_map_end.x() - m_map_start.x() - DISTANCE_THRESHOLD * (CCPP_CELL_THRESHOLD + 0.5));
				max_y = (m_map_start.y() + max_y + DISTANCE_THRESHOLD * CCPP_CELL_THRESHOLD < m_map_end.y()) ? max_y : (m_map_end.y() - m_map_start.y() - DISTANCE_THRESHOLD * (CCPP_CELL_THRESHOLD + 0.5));
				Eigen::Vector3f max_point(m_map_start.x() + max_x + DISTANCE_THRESHOLD * CCPP_CELL_THRESHOLD, m_map_start.y() + max_y + DISTANCE_THRESHOLD * CCPP_CELL_THRESHOLD, 100);
				fake_building.bounding_box_3d = Eigen::AlignedBox3f(max_point - Eigen::Vector3f(1, 1, 1), max_point);
				fake_building.trajectory.emplace_back(max_point, Eigen::Vector3f(0, 0, 1));
			}
		}
		else // Final check 
		{
			m_motion_status = Motion_status::final_check;
			//Building fake_building;
			fake_building.bounding_box_3d = Eigen::AlignedBox3f(m_map_end - Eigen::Vector3f(1 + DISTANCE_THRESHOLD * 0.5, 1 + DISTANCE_THRESHOLD * 0.5, 1 + DISTANCE_THRESHOLD * 0.5), m_map_end - Eigen::Vector3f(DISTANCE_THRESHOLD * 0.5, DISTANCE_THRESHOLD * 0.5, DISTANCE_THRESHOLD * 0.5));
			fake_building.trajectory.emplace_back(m_map_end, Eigen::Vector3f(0, 0, 1));
		}
		return fake_building;

	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		m_ccpp_trajectory.clear();
		m_current_ccpp_trajectory_id = -1;
		// Find next target (building or place) with higher confidence
		std::vector<int> untraveled_buildings;
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].is_divide)
				continue;
			else if (v_buildings[i_building].passed_trajectory.size() == 0)
			{
				untraveled_buildings.push_back(i_building);
			}
		}
		if (untraveled_buildings.size() == 0)
		{
			get_ccpp_trajectory(v_cur_pos.pos_mesh,
				default_expand(), 0);
			return;
		}

		// Find nearest building to existing polygon
		std::sort(untraveled_buildings.begin(),
			untraveled_buildings.end(),
			[&v_cur_pos, &v_buildings, this](const int& b1, const int& b2) {
				int view1 = v_buildings[b1].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
				int view2 = v_buildings[b2].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
				return (v_buildings[b1].trajectory[view1].pos_mesh - v_cur_pos.pos_mesh).norm() < (v_buildings[b2].trajectory[view2].pos_mesh - v_cur_pos.pos_mesh).norm();
				//return (v_buildings[id1].trajectory[view1].first.x()- v_cur_pos.pos_mesh.x()) < (v_buildings[id2].trajectory[view2].first.x() - v_cur_pos.pos_mesh.x());
			});
		//std::sort(untraveled_buildings.begin(),
		//	untraveled_buildings.end(),
		//	[&v_cur_pos, &v_buildings, this](const int& b1, const int& b2) {
		//		return (v_buildings[b1].bounding_box_3d.box.center() - v_cur_pos.pos_mesh).norm() < (v_buildings[b2].bounding_box_3d.box.center() - v_cur_pos.pos_mesh).norm();
		//		//return (v_buildings[id1].trajectory[view1].first.x()- v_cur_pos.pos_mesh.x()) < (v_buildings[id2].trajectory[view2].first.x() - v_cur_pos.pos_mesh.x());
		//});

		int id_building = 0;
		if (with_exploration)
		{
			bool ccpp_done = get_ccpp_trajectory(v_cur_pos.pos_mesh, v_buildings[untraveled_buildings[id_building]],
				CCPP_CELL_THRESHOLD);
			while (!ccpp_done)
			{
				id_building += 1;
				if (id_building >= untraveled_buildings.size())
				{
					get_ccpp_trajectory(v_cur_pos.pos_mesh,
						default_expand(), 0);
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
			if ((m_motion_status == Motion_status::exploration || m_motion_status == Motion_status::final_check) && region_status[nearest_region_id] == color_unobserved)
			{
				bool inside = false;
				for (auto item : topology)
					if (inside_box(Eigen::Vector2f(sample_points[nearest_region_id].x(), sample_points[nearest_region_id].y()), item))
						inside = true;

				if (inside && m_boundary.bounded_side(sample_points[nearest_region_id]) == CGAL::ON_BOUNDED_SIDE)
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

	Viewpoint determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override
	{
		bool with_reconstruction = m_arg["with_reconstruction"].asBool();

		Viewpoint next_pos;
		if (m_motion_status == Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}
		if (with_exploration && (m_motion_status == Motion_status::exploration || m_motion_status == Motion_status::final_check))
		{
			std::vector<int> untraveled_buildings_inside_exist_region;
			Eigen::Vector2f cur_point_cgal(m_ccpp_trajectory[m_current_ccpp_trajectory_id].pos_mesh.x(),
				m_ccpp_trajectory[m_current_ccpp_trajectory_id].pos_mesh.y());
			for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
				if (v_buildings[i_building].is_divide)
					continue;
				if (v_buildings[i_building].passed_trajectory.size() == 0) {
					//int id_nearest_view = v_buildings[i_building].find_nearest_trajectory_2d(Eigen::Vector3f(cur_point_cgal.x(), cur_point_cgal.y(), 0));
					//Eigen::Vector2f nearest_view(v_buildings[i_building].trajectory[id_nearest_view].first.x(), 
					//	v_buildings[i_building].trajectory[id_nearest_view].first.y());
					Eigen::Vector2f nearest_view(v_buildings[i_building].bounding_box_3d.box.center().x(),
						v_buildings[i_building].bounding_box_3d.box.center().y());

					if ((nearest_view - cur_point_cgal).norm() < DISTANCE_THRESHOLD / 1.4)
						untraveled_buildings_inside_exist_region.push_back(i_building);
				}
			}
			if (untraveled_buildings_inside_exist_region.size() != 0 && !m_isFirst) {
				//throw;
				m_motion_status = Motion_status::reconstruction_in_exploration;
				int id_building = std::min_element(untraveled_buildings_inside_exist_region.begin(),
					untraveled_buildings_inside_exist_region.end(),
					[&v_cur_pos, &v_buildings, this](const int& id1, const int& id2) {
						int view1 = v_buildings[id1].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
						int view2 = v_buildings[id2].find_nearest_trajectory_2d(v_cur_pos.pos_mesh);
						return (v_buildings[id1].trajectory[view1].pos_mesh - v_cur_pos.pos_mesh).norm() < (v_buildings[id2].trajectory[view2].pos_mesh - v_cur_pos.pos_mesh).norm();
						//return (v_buildings[id1].trajectory[view1].first.x()- v_cur_pos.pos_mesh.x()) < (v_buildings[id2].trajectory[view2].first.x() - v_cur_pos.pos_mesh.x());
					}) - untraveled_buildings_inside_exist_region.begin();
					m_current_building_id = untraveled_buildings_inside_exist_region[id_building];
					next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
					return next_pos;

					//m_motion_status = Motion_status::reconstruction;
			}
			else
			{
				if (m_exploration_point.size() == 0)
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
							m_isFirst = true;
							next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
							m_isFirst = false;
							//m_motion_status = Motion_status::reconstruction;
							m_current_color_id += 1;
							return next_pos;
						}
					}
					else {
						auto& item = m_ccpp_trajectory[m_current_ccpp_trajectory_id].pos_mesh;

						std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> start_points;

						start_points.emplace_back(
							Eigen::Vector3f(
								item.x() - DISTANCE_THRESHOLD * 0.25,
								item.y(),
								100),
							Eigen::Vector3f(1, 0, 0)
						);
						start_points.emplace_back(
							Eigen::Vector3f(
								item.x() + DISTANCE_THRESHOLD * 0.25,
								item.y(),
								100),
							Eigen::Vector3f(-1, 0, 0)
						);
						start_points.emplace_back(
							Eigen::Vector3f(
								item.x(),
								item.y() + DISTANCE_THRESHOLD * 0.25,
								100),
							Eigen::Vector3f(0, -1, 0)
						);
						start_points.emplace_back(
							Eigen::Vector3f(
								item.x(),
								item.y() - DISTANCE_THRESHOLD * 0.25,
								100),
							Eigen::Vector3f(0, 1, 0)
						);

						int nearest_id = std::min_element(start_points.begin(), start_points.end(),
							[&v_cur_pos](auto& a1, auto& a2) {
								return (v_cur_pos.pos_mesh - a1.first).norm() < (v_cur_pos.pos_mesh - a2.first).norm();
							}) - start_points.begin();

							if (m_arg["explore_multi_angle"].asBool())
								m_exploration_point.emplace(
									start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 0 * start_points[nearest_id].second,
									start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 0 * start_points[nearest_id].second +
									(m_arg["fix_angle_flag"].asBool() ? Eigen::Vector3f(std::cos(64.f / 180 * M_PI), 0, -std::sin(64.f / 180 * M_PI)) : Eigen::Vector3f(-0.5, -0.5 * 1.732, -std::tan(64.f / 180 * M_PI)).normalized())
								);
							m_exploration_point.emplace(
								start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 1 * start_points[nearest_id].second,
								start_points[nearest_id].first + DISTANCE_THRESHOLD * 0.25 * 1 * start_points[nearest_id].second +
								(m_arg["fix_angle_flag"].asBool() ? Eigen::Vector3f(std::cos(64.f / 180 * M_PI), 0, -std::sin(64.f / 180 * M_PI)) : Eigen::Vector3f(1, 0, -std::tan(64.f / 180 * M_PI)).normalized())
							);
							if (m_arg["explore_multi_angle"].asBool())
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
			if (!with_reconstruction)
			{
				if (with_exploration)
				{
					m_motion_status = Motion_status::exploration;
					next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
				}
				else
					throw;
			}
			if (true)
			{
				const int& cur_building_id = m_current_building_id;
				std::vector<Viewpoint> unpassed_trajectory;
				std::vector<Viewpoint>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
				if (passed_trajectory.size() == 0)
				{
					int id_closest_trajectory = std::min_element(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
						[&v_cur_pos](const auto& tra1, const auto& tra2) {
							return (tra1.pos_mesh - v_cur_pos.pos_mesh).norm() < (tra2.pos_mesh - v_cur_pos.pos_mesh).norm();
						}) - v_buildings[cur_building_id].trajectory.begin();
						v_buildings[cur_building_id].closest_trajectory_id = id_closest_trajectory;
				}
				int start_pos_id = 0;
				int one_pass_trajectory_num = v_buildings[cur_building_id].one_pass_trajectory_num;
				std::copy_if(v_buildings[cur_building_id].trajectory.begin() + v_buildings[cur_building_id].closest_trajectory_id, v_buildings[cur_building_id].trajectory.end(),
					std::back_inserter(unpassed_trajectory),
					[&passed_trajectory, v_threshold, &unpassed_trajectory, &start_pos_id, one_pass_trajectory_num](const Viewpoint& item_new_trajectory) {
						bool untraveled = true;
						for (auto item_passed_trajectory_iter = passed_trajectory.begin(); item_passed_trajectory_iter < passed_trajectory.end(); ++item_passed_trajectory_iter) {
							auto item_passed_trajectory = *item_passed_trajectory_iter;
							Eigen::Vector3f vector1 = item_passed_trajectory.direction;
							Eigen::Vector3f vector2 = item_new_trajectory.direction;
							float dot_product = vector1.dot(vector2);
							if (dot_product > 1)
								dot_product = 1;
							float angle = std::acos(dot_product) / M_PI * 180;
							if (angle >= 180)
								angle = 0;
							if ((item_passed_trajectory.pos_mesh - item_new_trajectory.pos_mesh).norm() < v_threshold && angle < 0.1) {
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
					[&passed_trajectory, v_threshold, &unpassed_trajectory, &start_pos_id, one_pass_trajectory_num](const Viewpoint& item_new_trajectory) {
						bool untraveled = true;
						for (auto item_passed_trajectory_iter = passed_trajectory.begin(); item_passed_trajectory_iter < passed_trajectory.end(); ++item_passed_trajectory_iter) {
							auto item_passed_trajectory = *item_passed_trajectory_iter;
							Eigen::Vector3f vector1 = item_passed_trajectory.direction;
							Eigen::Vector3f vector2 = item_new_trajectory.direction;
							float dot_product = vector1.dot(vector2);
							if (dot_product > 1)
								dot_product = 1;
							float angle = std::acos(dot_product) / M_PI * 180;
							if (angle >= 180)
								angle = 0;
							if ((item_passed_trajectory.pos_mesh - item_new_trajectory.pos_mesh).norm() < v_threshold && angle < 0.1) {
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
					if (with_exploration)
					{
						m_motion_status = Motion_status::exploration;
						next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
					}
					else
					{
						m_motion_status = Motion_status::reconstruction_in_exploration;
						get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
						if (m_motion_status == Motion_status::final_check)
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
			}
			else
			{
				const int& cur_building_id = m_current_building_id;
				std::vector<Viewpoint>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
				if (passed_trajectory.size() == 0)
				{
					int id_closest_trajectory = std::min_element(v_buildings[cur_building_id].trajectory.begin(), v_buildings[cur_building_id].trajectory.end(),
						[&v_cur_pos](const auto& tra1, const auto& tra2) {
							return (tra1.pos_mesh - v_cur_pos.pos_mesh).norm() < (tra2.pos_mesh - v_cur_pos.pos_mesh).norm();
						}) - v_buildings[cur_building_id].trajectory.begin();
						v_buildings[cur_building_id].closest_trajectory_id = id_closest_trajectory;
				}

				if (passed_trajectory.size() == v_buildings[cur_building_id].trajectory.size()) // Exit reconstruction mode
				{
					if (with_exploration)
					{
						m_motion_status = Motion_status::exploration;
						next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
					}
					else
					{
						m_motion_status = Motion_status::reconstruction_in_exploration;
						get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
						if (m_motion_status == Motion_status::final_check)
						{
							m_motion_status = Motion_status::done;
							return next_pos;
						}

						next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
					}
				}
				else
				{
					int id = v_buildings[cur_building_id].closest_trajectory_id + passed_trajectory.size();
					if (id >= v_buildings[cur_building_id].trajectory.size())
						id -= v_buildings[cur_building_id].trajectory.size();
					next_pos = v_buildings[cur_building_id].trajectory[id];
					passed_trajectory.push_back(next_pos);
					return next_pos;

				}
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


/*
 * Ablation study
 */
class Next_best_target_random_min_distance :public Next_best_target
{
public:
	int m_current_exploration_id = -1;
	cv::Vec3b color_unobserved, color_free, color_occupied;
	std::vector<bool> already_explored;

	Next_best_target_random_min_distance(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh, float v_ccpp_cell_distance)
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

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override
	{
		// Find next target (building or place) with higher confidence
		std::vector<Next_target> untraveled_buildings;
		// Find with distance
		for (int i_building = 0; i_building < v_buildings.size(); ++i_building) {
			if (v_buildings[i_building].passed_trajectory.size() == 0 && v_buildings[i_building].is_divide!=true)
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
		int nearest_region_id = std::min_element(sample_points.begin(), sample_points.end(),
			[&cur_point_cgal](const CGAL::Point_2<K>& p1, const CGAL::Point_2<K>& p2) {
				return std::pow(p1.x() - cur_point_cgal.x(), 2) + std::pow(p1.y() - cur_point_cgal.y(), 2) < std::pow(p2.x() - cur_point_cgal.x(), 2) + std::pow(p2.y() - cur_point_cgal.y(), 2);
			}) - sample_points.begin();
			if (m_motion_status == Motion_status::exploration)
			{
				region_status[nearest_region_id] = region_viz_color[1];
				already_explored[nearest_region_id] = true;
			}

			/*for (int i_point = 0; i_point < region_status.size(); i_point++) {
				const Eigen::Vector2f p(sample_points[i_point].x(), sample_points[i_point].y());

				for (const auto& item_building : v_buildings) {
					Eigen::Vector3f point(p.x(), p.y(), 0.f);
					if (item_building.bounding_box_3d.inside_2d(point)) {
						region_status[i_point] = color_occupied;
						break;
					}
				}
			}*/
	}

	Viewpoint determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override {
		Viewpoint next_pos;

		if (m_motion_status == Motion_status::initialization) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			LOG(INFO) << "Initialization target !";
		}

		if (m_motion_status == Motion_status::exploration) {
			get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
			Eigen::Vector3f next_pos(sample_points[m_current_exploration_id].x(), sample_points[m_current_exploration_id].y(), 100);
			return Viewpoint(next_pos, next_pos);
		}
		if (m_motion_status == Motion_status::reconstruction) {
			const int& cur_building_id = m_current_building_id;
			std::vector<Viewpoint> unpassed_trajectory;
			std::vector<Viewpoint>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;

			if (passed_trajectory.size() == v_buildings[cur_building_id].trajectory.size()) {
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				LOG(INFO) << "Change target !";
				return determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else {
				next_pos = v_buildings[cur_building_id].trajectory[passed_trajectory.size()];
				passed_trajectory.push_back(next_pos);
				return next_pos;
			}

		}
	}

};

class Next_best_target_exploration_first :public Next_best_target_topology_exploration
{
public:
	Next_best_target_exploration_first(const Eigen::Vector3f& v_map_start_mesh, const Eigen::Vector3f& v_map_end_mesh,
		int v_CCPP_CELL_THRESHOLD, const Polygon_2& m_boundary, float v_ccpp_cell_distance, const Json::Value& v_arg)
		:Next_best_target_topology_exploration(v_map_start_mesh, v_map_end_mesh, v_CCPP_CELL_THRESHOLD, m_boundary, v_ccpp_cell_distance, v_arg)
	{

	}

	void get_next_target(int frame_id, const Pos_Pack& v_cur_pos, const std::vector<Building>& v_buildings, bool with_exploration) override {
		m_ccpp_trajectory.clear();
		m_current_ccpp_trajectory_id = 0;
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

			m_current_building_id = untraveled_buildings[next_target_id].origin_index_in_building_vector;
			m_motion_status = Motion_status::reconstruction;
			return;
	}
	Viewpoint determine_next_target(int v_frame_id, const Pos_Pack& v_cur_pos, std::vector<Building>& v_buildings, bool with_exploration, float v_threshold) override
	{

		Viewpoint next_pos;
		if (m_motion_status == Motion_status::done)
			return next_pos;
		if (m_motion_status == Motion_status::initialization) {
			m_motion_status = Motion_status::exploration;
			Building fake_building;
			fake_building.bounding_box_3d = Eigen::AlignedBox3f(m_map_end - Eigen::Vector3f(1 + DISTANCE_THRESHOLD * 0.5, 1 + DISTANCE_THRESHOLD * 0.5, 1 + DISTANCE_THRESHOLD * 0.5), m_map_end - Eigen::Vector3f(DISTANCE_THRESHOLD * 0.5, DISTANCE_THRESHOLD * 0.5, DISTANCE_THRESHOLD * 0.5));
			fake_building.trajectory.emplace_back(m_map_end, Eigen::Vector3f(0, 0, 1));
			get_ccpp_trajectory(v_cur_pos.pos_mesh, fake_building, 0);
			next_pos = Viewpoint(Eigen::Vector3f(0, 0, 100),
				Eigen::Vector3f(0, 0, 100)+Eigen::Vector3f(1, 0, 0));
			LOG(INFO) << "Initialization target !";
		}
		if (m_motion_status == Motion_status::exploration)
		{
			std::vector<int> untraveled_buildings_inside_exist_region;
			Eigen::Vector2f cur_point_cgal(m_ccpp_trajectory[m_current_ccpp_trajectory_id].pos_mesh.x(),
				m_ccpp_trajectory[m_current_ccpp_trajectory_id].pos_mesh.y());
			m_current_ccpp_trajectory_id += 1;
			if (m_current_ccpp_trajectory_id >= m_ccpp_trajectory.size()) {
				m_motion_status = Motion_status::reconstruction;
				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
				//m_motion_status = Motion_status::reconstruction;
				m_current_color_id += 1;
				return next_pos;
			}
			else {
				auto& item = m_ccpp_trajectory[m_current_ccpp_trajectory_id].pos_mesh;

				Viewpoint start_points = Viewpoint(Eigen::Vector3f(
					item.x(),
					item.y(),
					100),
					Eigen::Vector3f(
						item.x(),
						item.y(),
						100) + Eigen::Vector3f(1, 0, 0));
				next_pos = start_points;
			}
		}
		if (m_motion_status == Motion_status::reconstruction)
		{
			const int& cur_building_id = m_current_building_id;
			std::vector<Viewpoint>& passed_trajectory = v_buildings[cur_building_id].passed_trajectory;
			if (passed_trajectory.size() == v_buildings[cur_building_id].trajectory.size()) // Exit reconstruction mode
			{

				get_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration);
				next_pos = determine_next_target(v_frame_id, v_cur_pos, v_buildings, with_exploration, v_threshold);
			}
			else
			{
				next_pos = v_buildings[cur_building_id].trajectory[passed_trajectory.size()];
				passed_trajectory.push_back(next_pos);
				return next_pos;

			}

		}
		return next_pos;
	}

};