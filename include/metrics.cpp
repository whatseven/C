#include "metrics.h"


int k1 = 32;
float alpha1 = 3.1415926f / 16;
int k3 = 8;
float alpha3 = 3.1415926f / 4;
float dmax = 60;

//std::vector<std::array<float, 5>> reconstructability_hueristic(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory,
//    const Point_set& point_set,
//    const Surface_mesh& v_mesh,std::vector<std::vector<bool>>& point_view_visibility)
//{
//    std::vector<std::array<float, 5>> reconstructabilities(point_set.size());
//    point_view_visibility.resize(point_set.size(), std::vector<bool>(trajectory.size(), false));
//    // Build AABB tree
//    Tree tree;
//    tree.insert(CGAL::faces(v_mesh).first, CGAL::faces(v_mesh).second, v_mesh);
//    tree.build();
//
//#pragma omp parallel for
//    for (int i_point=0;i_point<point_set.size();++i_point)
//    {
//        const Point_3& point = point_set.point(i_point);
//        const Vector_3& normal = point_set.normal(i_point) / std::sqrt(point_set.normal(i_point).squared_length());
//    	
//        for (int j = 0; j < trajectory.size(); j++)
//        {
//            const Point_3 view_pos(trajectory[j].first.x(), trajectory[j].first.y(), trajectory[j].first.z());
//            Vector_3 view_direction(trajectory[j].second.x(), trajectory[j].second.y(), trajectory[j].second.z());
//            view_direction /= std::sqrt(view_direction.squared_length());
//        	
//            Vector_3 view_to_point = point - view_pos;
//            Vector_3 view_to_point_norm = view_to_point/std::sqrt(view_to_point.squared_length());
//            float distance = std::sqrt(view_to_point.squared_length());
//
//            if(!(CGAL::scalar_product(view_to_point_norm,view_direction)>0.707))
//                continue;
//
//            // Intersection Test
//            Ray ray_query(view_pos, view_to_point_norm);
//            auto intersection = tree.first_intersection(ray_query);
//            float intersection_point_distance=0.f;
//            if (intersection&& boost::get<Point_3>(&(intersection->first))) {
//                const Point_3* p = boost::get<Point_3>(&(intersection->first));
//                intersection_point_distance = std::sqrt((*p - view_pos).squared_length());
//            }
//            else
//                continue;
//
//            if (std::abs(intersection_point_distance - distance)<0.1)
//                point_view_visibility[i_point][j] = true;
//        }
//
//        std::vector<std::array<float, 5>> point_recon;
//        for (int id_view1 = 0; id_view1 < trajectory.size(); id_view1++)
//            for (int id_view2 = id_view1+1; id_view2 < trajectory.size(); id_view2++)
//            {
//                const Eigen::Vector3f point_eigen(point.x(), point.y(), point.z());
//                Eigen::Vector3f normal_eigen(normal.x(), normal.y(), normal.z());
//	            if(!(point_view_visibility[i_point][id_view2] && point_view_visibility[i_point][id_view1]))
//                    continue;
//                Eigen::Vector3f view_to_point1 = point_eigen - trajectory[id_view1].first;
//                Eigen::Vector3f view_direction1 = trajectory[id_view1].second;
//            	Eigen::Vector3f view_to_point2 = point_eigen - trajectory[id_view2].first;
//                Eigen::Vector3f view_direction2 = trajectory[id_view2].second;
//
//                float alpha = std::acos(view_to_point1.normalized().dot(view_to_point2.normalized()));
//                float omega1 = 1. / (1 + exp(-k1 * (alpha - alpha1)));
//                float omega2 = 1 - std::min(std::max(view_to_point1.norm(), view_to_point2.norm()) / dmax, 1.f);
//                float omega3 = 1. - 1. / (1 + exp(-k3 * (alpha - alpha3)));
//                float Theta1 =(-view_to_point1).normalized().dot(normal_eigen);
//                float Theta2 =(-view_to_point2).normalized().dot(normal_eigen);
//                float cosTheta = std::min(Theta1, Theta2);
//                float value = omega1 * omega2 * omega3 * cosTheta;
//                point_recon.push_back(std::array<float, 5>{omega1, omega2, omega3, cosTheta, value > 0 ? value : 0});
//            }
//        std::array<float,5> total_recon = std::accumulate(point_recon.begin(), point_recon.end(), std::array<float,5>{0.f,0.f,0.f,0.f,0.f}, [](std::array<float,5> sum, auto item)
//        {
//            sum[0] += std::get<0>(item);
//            sum[1] += std::get<1>(item);
//            sum[2] += std::get<2>(item);
//            sum[3] += std::get<3>(item);
//            sum[4] += std::get<4>(item);
//            return sum;
//        });
//
//        //float num_view_pair = trajectory.size()* (trajectory.size()-1) / 2;
//        float num_view_pair = 1;
//        reconstructabilities[i_point] = std::array<float, 5>{
//            total_recon[0] / num_view_pair,
//                total_recon[1] / num_view_pair,
//                total_recon[2] / num_view_pair,
//                total_recon[3] / num_view_pair,
//                (total_recon[4] > 0 ? total_recon[4] : 0) / num_view_pair};
//    }
//
//    return reconstructabilities;
//}
