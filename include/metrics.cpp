#include "metrics.h"

float reconstructability_hueristic(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> trajectory, std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> point_set, std::list<SC_Triangle>& triangles)
{
    Eigen::Matrix3f INTRINSIC;
    INTRINSIC << 400, 0, 400, 0, 400, 400, 0, 0, 1;

    std::vector<float> reconstructabilities;

    // Build AABB tree
    SC_Tree tree(triangles.begin(), triangles.end());
    
    for (int i = 0; i < point_set.size(); i++)
    {
        float distance_V1 = 999;
        float distance_V2 = 999;
        std::pair<Eigen::Vector3f, Eigen::Vector3f> V1, V2;
        for (int j = 0; j < trajectory.size(); j++)
        {
            float distance = (point_set[i].first - trajectory[j].first).norm();
            // Get V1 V2
            Eigen::Vector3f pixel_coord = INTRINSIC * (point_set[i].first - trajectory[j].first);
            Eigen::Vector3f segment1 = point_set[i].first - trajectory[j].first;
            Eigen::Vector3f segment2 = trajectory[j].second;
            float temp = segment1.dot(segment2);
            float depth = temp / segment2.norm();
            pixel_coord /= depth;

            // Intersection Test
            bool visibility = true;
            SC_Point temp1(point_set[i].first.x(), point_set[i].first.y(), point_set[i].first.z());
            SC_Point temp2(trajectory[j].first.x(), trajectory[j].first.y(), trajectory[j].first.z());
            SC_Ray ray_query(temp2, temp1);
            SC_Ray_intersection intersection = tree.any_intersection(ray_query);
            Eigen::Vector3f intersection_point;
            if (intersection) {
                if (boost::get<SC_Point>(&(intersection->first))) {
                    const SC_Point* p = boost::get<SC_Point>(&(intersection->first));
                    intersection_point[0] = p->x();
                    intersection_point[1] = p->y();
                    intersection_point[2] = p->z();
                }
            }
            float intersection_point_distance = (intersection_point - point_set[i].first).norm();

            // Update V1 V2
            if (intersection_point_distance <= 0.1)
            {
                if (pixel_coord[0] >= 0 && pixel_coord[0] <= 800 && pixel_coord[1] >= 0 && pixel_coord[1] <= 800)
                {
                    if (distance < distance_V1)
                    {
                        V2 = V1;
                        distance_V2 = distance_V1;
                        distance_V1 = distance;
                        V1 = trajectory[j];
                    }
                    else if (distance < distance_V2)
                    {
                        distance_V2 = distance;
                        V2 = trajectory[j];
                    }
                }
            }
        }
        // Set parameters
        int k1 = 32;
        float alpha1 = PI / 16;
        int k3 = 8;
        float alpha3 = PI / 4;
        float alpha = acos((V1.first - point_set[i].first).dot((V2.first - point_set[i].first)) / (V1.first - point_set[i].first).norm() / (V2.first - point_set[i].first).norm());
        float omega1 = 1. / (1 + exp(-k1 * (alpha - alpha1)));
        float dmax = 50;
        float omega2 = 1 - std::min(distance_V2 / dmax, 1.f);
        float omega3 = 1. - 1. / (1 + exp(-k3 * (alpha - alpha3)));
        float Theta1 = acos((V1.first - point_set[i].first).dot(point_set[i].second) / (V1.first - point_set[i].first).norm() / point_set[i].second.norm());
        float Theta2 = acos((V2.first - point_set[i].first).dot(point_set[i].second) / (V2.first - point_set[i].first).norm() / point_set[i].second.norm());
        float cosTheta = cos(std::max(Theta1, Theta2));
        float value = omega1 * omega2 * omega3 * cosTheta;
        reconstructabilities.push_back(value);
    }

    return 0;
}

void read_point_set(std::string path, std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& point_set, CGAL::Bbox_3 bounding_box_3d)
{
    float xmin = bounding_box_3d.xmin();
    float ymin = bounding_box_3d.ymin();
    float xmax = bounding_box_3d.xmax();
    float ymax = bounding_box_3d.ymax();
    std::ifstream ObjFile(path);
    std::string line;
    while (getline(ObjFile, line))
    {
        Eigen::Vector3f vertex, normal;
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
            vertex = Eigen::Vector3f(atof(vData[0].c_str()), atof(vData[1].c_str()), atof(vData[2].c_str()));
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
            normal = Eigen::Vector3f(atof(vData[0].c_str()), atof(vData[1].c_str()), atof(vData[2].c_str()));
        }
        if (vertex.x() >= xmin && vertex.x() <= xmax && vertex.y() >= ymin && vertex.y() <= ymax) {
            point_set.push_back(std::make_pair(vertex, normal));
        }
    }
}

void readObj(std::string path, std::list<SC_Triangle>& faces)
{
    std::ifstream ObjFile(path);
    std::string line;
    std::vector<SC_Point> vertices;
    std::vector<std::vector<int>> faceIdx;
    while (getline(ObjFile, line))
    {
        std::vector<std::string> vData;
        std::vector<int> fData;
        if (line.substr(0, line.find(" ")) == "v")
        {
            line = line.substr(line.find(" ") + 1);
            for (int i = 0; i < 3; i++)
            {
                vData.push_back(line.substr(0, line.find(" ")));
                line = line.substr(line.find(" ") + 1);
            }
            //vData[2] = vData[2].substr(0, vData[2].find("\n"));
            vertices.push_back(SC_Point(atof(vData[0].c_str()), atof(vData[1].c_str()), atof(vData[2].c_str())));
        }
        if (line.substr(0, line.find(" ")) == "f")
        {
            line = line.substr(line.find(" ") + 1);
            for (int i = 0; i < 3; i++)
            {
                vData.push_back(line.substr(0, line.find(" ")).substr(0, line.find("/")));
                line = line.substr(line.find(" ") + 1);
            }
            vData[2] = vData[2].substr(0, vData[2].find("\n"));
            for (int i = 0; i < 3; i++)
            {
                fData.push_back(atoi(vData[i].c_str()));
            }
            faceIdx.push_back(fData);
        }
    }
    for (int i = 0; i < faceIdx.size(); i++)
    {
        std::vector<double> planePara;
        faces.push_back(SC_Triangle(vertices[faceIdx[i][0] - 1], vertices[faceIdx[i][1] - 1], vertices[faceIdx[i][2] - 1]));
    }
}