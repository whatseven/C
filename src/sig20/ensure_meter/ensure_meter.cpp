#include <boost/format.hpp>
#include <CGAL/Point_set_3/IO.h>
#include <json/reader.h>
#include "../main/trajectory.h"

int main(int argc, char** argv){
	boost::filesystem::recursive_directory_iterator end_iter;

	int delete_index = 0;
	int preserve_index = 0;
	int global_file_index = 0;
	for(boost::filesystem::recursive_directory_iterator it("C:\\Users\\yilin\\Documents\\Tencent Files\\1787609016\\FileRecv\\tansuo_problemPath_Split\\tansuo_problemPath_Split\\split");it!=end_iter;++it)
	{
		if(it->path().extension()==".drone_path")
		{
			fs::path path_file(it->path());
			auto trajectories = read_wgs84_trajectory(path_file.string());

			std::vector<Eigen::Vector3f> exist_pos;
			std::vector<Eigen::Vector3f> exist_direction;
			 
			//std::ofstream f(std::to_string(global_index) + "_0.txt");
			std::ofstream f(std::to_string(global_file_index)+".txt");
			int num = 1;
			for (int i_trajectory1 = 0;i_trajectory1 < trajectories.size();++i_trajectory1)
			{

				if (trajectories[i_trajectory1].first.z() > 119)
					delete_index += 1;
				else
				{
					preserve_index += 1;
					f << (boost::format("%f %f %f %f %f") % trajectories[i_trajectory1].first.x() % trajectories[i_trajectory1].first.y() % trajectories[i_trajectory1].first.z() % trajectories[i_trajectory1].second.y() % trajectories[i_trajectory1].second.x()).str() << std::endl;
				}
				continue;
				Eigen::Vector2f first_pos = lonLat2Mercator(Eigen::Vector2f(
					trajectories[i_trajectory1].first.x(),
					trajectories[i_trajectory1].first.y()
				));
				Eigen::Vector3f first_pos_3(first_pos.x(), first_pos.y(), trajectories[i_trajectory1].first.z());


				first_pos_3.x() -= 1.26826e7;
				first_pos_3.y() -= 2.57652e6;
				

				bool accept = true;
				for (const auto& item : exist_pos) {
					if ((first_pos_3 - item).norm() < 1)
					{
						f.close();
						f = std::ofstream(std::to_string(global_file_index) + "_" + std::to_string(num) + ".txt");
						num += 1;
						exist_pos.clear();
						exist_pos.push_back(first_pos_3);
						accept = false;
						break;
					}
				}
				if (accept)
					exist_pos.push_back(first_pos_3);

				f << (boost::format("%f %f %f %f %f") % trajectories[i_trajectory1].first.x() % trajectories[i_trajectory1].first.y() % trajectories[i_trajectory1].first.z() % trajectories[i_trajectory1].second.y() % trajectories[i_trajectory1].second.x()).str() << std::endl;
			}
			f.close();
			global_file_index += 1;
		}
	}
	std::cout << delete_index << std::endl;
	std::cout << preserve_index << std::endl;
	return 0;
}
