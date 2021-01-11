#include <argparse/argparse.hpp>

#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/cluster_point_set.h>
#include <CGAL/Random.h>
#include <CGAL/IO/OBJ_reader.h>
#include <CGAL/Surface_mesh/IO.h>
#include <CGAL/point_generators_3.h>

#include <boost/format.hpp>

#include "model_tools.h"
#include "intersection_tools.h"
#include "../main/building.h"
#include "../main/trajectory.h"
#include "../main/viz.h"
#include "common_util.h"
#include "metrics.h"

const float BOUNDS = 5;
const float Z_UP_BOUNDS = 5;
const float Z_DOWN_BOUND = 5;
const float STEP = 3;
const float CENTER_Z1 = 0.5;
const float CENTER_Z2 = 0.25;
const bool DOUBLE_FLAG = true;

//const std::string trajectory_path = "D:\\SIG21\\asia_path\\test.utj";
//const std::string trajectory_path = "C:\\repo\\C\\build\\src\\sig20\\main\\camera_normal.log";
//const std::string trajectory_path = "C:\\repo\\C\\build\\src\\sig20\\generate_trajectory_with_prior\\camera_normal.log";
//std::string trajectory_path = "C:\\repo\\C\\temp\\path0.drone_path";
//const std::string sample_points_path = "C:\\repo\\C\\temp\\bridge_sample_points.ply";

//
//const std::string trajectory_path = "C:\\repo\\C\\temp\\adjacent\\10_single_upper.log";
const std::string trajectory_path = "D:\\SIG21_Local\\2_2_error_map\\asia18_bridge_565.txt";
const std::string recon_sample_points_path = "D:\\SIG21_Local\\2_2_error_map\\asia18_bridge_565_align_points.ply";
const std::string gt_sample_points_path = "D:\\SIG21_Local\\2_2_error_map\\bridge_points_wo_building.ply";
const std::string recon_mesh_path = "D:\\SIG21_Local\\2_2_error_map\\asia18_bridge_565_align.ply";
const std::string gt_mesh_path = "D:\\SIG21_Local\\2_2_error_map\\bridge_mesh.ply";

const std::string output_data_path = "D:\\SIG21_Local\\2_2_error_map\\asia18.txt";


int main(int argc, char** argv){
	// Read arguments
	argparse::ArgumentParser program("Evaluate trajectory");
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory = read_normal_trajectory(trajectory_path);
	std::cout << "Total length: " << evaluate_length(trajectory) << std::endl;
	Surface_mesh gt_mesh;
	Surface_mesh recon_mesh;
	CGAL::read_ply(std::ifstream(recon_mesh_path,std::ios_base::binary), recon_mesh);
	CGAL::read_ply(std::ifstream(gt_mesh_path, std::ios_base::binary), gt_mesh);
	std::cout << recon_mesh.faces().size();
	Point_set sample_recon_points, sample_gt_points;
	CGAL::read_ply_point_set(std::ifstream(recon_sample_points_path, std::ios_base::binary), sample_recon_points);
	CGAL::read_ply_point_set(std::ifstream(gt_sample_points_path, std::ios_base::binary), sample_gt_points);
	
	Tree gt_tree,recon_tree;
	gt_tree.insert(CGAL::faces(gt_mesh).first, CGAL::faces(gt_mesh).second, gt_mesh);
	gt_tree.build();
	recon_tree.insert(CGAL::faces(recon_mesh).first, CGAL::faces(recon_mesh).second, recon_mesh);
	recon_tree.build();

	std::cout << "Build done!" << std::endl;

	std::vector<std::vector<bool>> point_view_visibility;
	std::vector<std::array<float,5>> reconstructability_recon_split = reconstructability_hueristic(trajectory, sample_recon_points, gt_mesh, point_view_visibility);
	std::vector<std::array<float,5>> reconstructability_gt_split = reconstructability_hueristic(trajectory, sample_gt_points, gt_mesh, point_view_visibility);
	std::vector<float> reconstructability_recon, reconstructability_gt;
	for (const auto& item : reconstructability_recon_split)
		reconstructability_recon.push_back(item[4]);
	for (const auto& item : reconstructability_gt_split)
		reconstructability_gt.push_back(item[4]);
	
	
	std::vector<float> recon_nearest_error;
	for(const Point_3& p: sample_recon_points.points())
	{
		Point_3 closest_recon_point= recon_tree.closest_point(p);
		Point_3 closest_gt_point= gt_tree.closest_point(closest_recon_point);
		float error = std::sqrt((closest_recon_point - closest_gt_point).squared_length());
		recon_nearest_error.push_back(error);
	}
	std::vector<float> gt_nearest_error;
	for(const Point_3& p: sample_gt_points.points())
	{
		Point_3 closest_gt_point= gt_tree.closest_point(p);
		Point_3 closest_recon_point= recon_tree.closest_point(closest_gt_point);
		float error = std::sqrt((closest_recon_point - closest_gt_point).squared_length());
		gt_nearest_error.push_back(error);
	}
	
	std::cout << "Error done!" << std::endl;

	std::string reconstructability_recon_string = "";
	std::string reconstructability_gt_string = "";
	std::string accuracy_string = "";
	std::string completeness_string = "";
	for(int i=0;i< sample_recon_points.size();++i)
	{
		reconstructability_recon_string += std::to_string(reconstructability_recon[i]) + ",";
		accuracy_string += std::to_string(recon_nearest_error[i]) + ",";
		reconstructability_gt_string += std::to_string(reconstructability_gt[i]) + ",";
		completeness_string += std::to_string(gt_nearest_error[i]) + ",";
	}
	std::ofstream f_out(output_data_path);
	f_out << reconstructability_recon_string << std::endl;
	f_out << accuracy_string << std::endl;
	f_out << reconstructability_gt_string << std::endl;
	f_out << completeness_string << std::endl;
	f_out.close();
	
	//Point_set point_set;
	//Surface_mesh mesh = convert_obj_from_tinyobjloader_to_surface_mesh(load_obj(mesh_path));
	//CGAL::read_ply_point_set(std::ifstream(sample_points_path), point_set);
	//std::vector<std::vector<bool>> point_view_visibility;
	//auto reconstructability = reconstructability_hueristic(trajectory, point_set, mesh, point_view_visibility);
	//std::vector<std::string> name{ "w1","w2","w3","cos","recon" };
	//for(int i=0;i<5;++i)
	//{
	//	float quartile1, quartile2, quartile3,mean,max;
	//	std::sort(reconstructability.begin(), reconstructability.end(), [&i](auto i1, auto i2) {return i1[i]>i2[i]; });
	//	quartile3 = reconstructability[reconstructability.size() / 4*3][i];
	//	mean = std::accumulate(reconstructability.begin(), reconstructability.end(), 0.f, [&i](float sum, auto& item) {return sum + item[i]; })/ reconstructability.size();
	//	max = (*std::max_element(reconstructability.begin(), reconstructability.end(), [&i](auto& item1, auto& item2) {return item1[i]< item2[i]; }))[i];
	//	std::cout<<(boost::format("*%s*: Quartile 3/4: %f. Max: %f. Mean: %f\n") % name[i].c_str() % quartile3% max% mean).str();
	//	std::cout<<(boost::format("%f %f %f\n") % quartile3% max% mean).str();
	//}
	//
	//std::vector<std::vector<bool>> view_point_visibility(point_view_visibility[0].size(),std::vector<bool>(point_view_visibility.size(),false));
	//for (int i_point = 0; i_point < point_view_visibility.size(); ++i_point)
	//	for (int i_trajectory = 0; i_trajectory < point_view_visibility[0].size(); ++i_trajectory)
	//		view_point_visibility[i_trajectory][i_point] = point_view_visibility[i_point][i_trajectory];

	Visualizer vizer;
	//for (int i = 0; i < view_point_visibility.size(); ++i)
	{
		//std::vector<Eigen::Vector4f> colors(view_point_visibility[i].size(), Eigen::Vector4f(0.f,0.f,0.f,1.f));

		//for (int i_point = 0; i_point < view_point_visibility[i].size(); ++i_point)
		//	if (view_point_visibility[i][i_point])
		//		colors[i_point] = Eigen::Vector4f(1.f, 0.f, 0.f, 1.f);

		//for (int i_point = 0; i_point < view_point_visibility[i].size(); ++i_point)
		//	if (reconstructability[i_point][4]>0)
		//		colors[i_point] = Eigen::Vector4f(1.f, 0.f, 0.f, 1.f);
		
		vizer.lock();
		vizer.m_trajectories = trajectory;
		vizer.m_points = sample_recon_points;
		//vizer.m_points_color = colors;
		vizer.m_pos = trajectory[0].first;
		vizer.unlock();
		debug_img(std::vector<cv::Mat>{cv::Mat(50, 50, CV_8UC3, cv::Scalar(255, 0, 0))});
	}

	override_sleep(3000);
	return 0;
}
