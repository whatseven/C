#include<iostream>
#include<random>
#include<algorithm>
#include<iterator>

#include "model_tools.h"

const std::string file_path = "F:\\AutoCAD\\Map_Data\\district1.txt";
const std::string output_path = "F:\\AutoCAD\\Map_Data\\district1\\";
int main()
{
	std::vector<Polygon_2> polygons = get_polygons(file_path);
	int polygon_number = 0;
	for (auto polygon : polygons)
	{
		std::ofstream output_file;
		output_file.open(output_path + std::to_string(polygon_number) + ".foot_print", std::ios::out);
		output_file << "0 1" << std::endl;
		for (auto vertex : polygon)
		{
			output_file << std::to_string(vertex.x()) << " " << std::to_string(vertex.y()) << " ";
		}
		polygon_number += 1;
	}
	std::cout << "done" << std::endl;
}