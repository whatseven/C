#include "metrics.h"

float reconstructability_hueristic(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>)
{

}

void read_point_set(std::string path, std::vector<std::pair<SC_Point, SC_Point>> point_set)
{
    std::ifstream ObjFile(path);
    std::string line;
    while (getline(ObjFile, line))
    {
        SC_Point vertex, normal;
        std::vector<std::string> vData;
        if (line.substr(0, line.find(" ")) == "v")
        {
            line = line.substr(line.find(" ") + 1);
            for (int i = 0; i < 3; i++)
            {
                vData.push_back(line.substr(0, line.find(" ")));
                line = line.substr(line.find(" ") + 1);
            }
            //vData[2] = vData[2].substr(0, vData[2].find("\n"));
            vertex = Point(atof(vData[0].c_str()), atof(vData[1].c_str()), atof(vData[2].c_str()));
        }
        vData.clear();
        if (line.substr(0, line.find(" ")) == "vn")
        {
            line = line.substr(line.find(" ") + 1);
            for (int i = 0; i < 3; i++)
            {
                vData.push_back(line.substr(0, line.find(" ")));
                line = line.substr(line.find(" ") + 1);
            }
            //vData[2] = vData[2].substr(0, vData[2].find("\n"));
            normal = Point(atof(vData[0].c_str()), atof(vData[1].c_str()), atof(vData[2].c_str()));
        }
        point_set.push_back(std::make_pair(vertex, normal));
    }
}