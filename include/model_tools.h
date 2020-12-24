#ifndef MODEL_TOOLS_H
#define MODEL_TOOLS_H
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <eigen3/Eigen/Core>

#include "cgal_tools.h"


/*
Some useful function
*/

// @brief: Get vertex, faces, normals, texcoords, textures from obj file. TinyobjLoader store the vertex, normals, texcoords in the `attrib_t`. Index to vertex for every face is stored in `shapt_t`.
// @notice:
// @param: File path; mtl file directory
// @ret: `attrib_t, shape_t, material_t`
std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> load_obj(const std::string& v_mesh_name,bool v_log=true, const std::string& v_mtl_dir = "./");

// @brief: Store mesh into obj file
// @notice: Every shape in the vector will have a group name
// @param: File path; attrib_t; shape_t; material_t;
// @ret:
bool write_obj(const std::string& filename, const tinyobj::attrib_t& attributes, const std::vector<tinyobj::shape_t>& shapes, const std::vector<tinyobj::material_t>& materials);

// @brief: Meshlab save obj file with an overiding material name. e.g. material_0, material_1... This function change the original mtl file into the meshlab style material name
// @notice: 
// @param: mtl file path
// @ret:
void fix_mtl_from_unreal(const std::string& filename);

// @brief: Clean duplicated face and vertex
// @notice: Currently implementation is only support shape with normals and texcoods!
// @param: attrib_t, shape_t
// @ret:
void clean_vertex(tinyobj::attrib_t& attrib, tinyobj::shape_t& shape);

/*
Get split mesh with a big whole mesh
*/
void merge_obj(const std::string& v_file,
    const std::vector<tinyobj::attrib_t>& v_attribs, const std::vector<tinyobj::shape_t>& saved_shapes,
    const std::vector<tinyobj::_material_t>& materials);

// @brief: Split the whole obj into small object and store them separatly. 
//         Each Object is store separatly and normalize near origin. The transformation is also stored in the txt
//         We also get a whole obj with "g" attribute to indicate each component. This obj can be imported into unreal with separate actor
// @notice: Be careful about the material file, you may adjust them manually
// @param: 
//          Directory that contains the obj file. The splited obj will also be stored in this directory
//          OBJ file name
//          Resolution indicates the resolution of the height map. (How far will the two building is considered to be one component)
// @ret:
void split_obj(const std::string& file_dir, const std::string& file_name, const float resolution, const float v_filter_height=-99999);

// @brief: Rename the material and image name
//         Unreal can not cope with complicate image name
// @notice: 
// @param: 
//          Directory that contains the obj file. 
//          OBJ file name
//          Output directory
// @ret:
void rename_material(const std::string& file_dir, const std::string& file_name, const std::string& v_output_dir);

class Height_map {
public:
    Height_map(const Point_set& v_point_cloud, const float v_resolution) :m_resolution(v_resolution) {
        Eigen::AlignedBox3f bounds = get_bounding_box(v_point_cloud);
        m_start = bounds.min();
        Eigen::Vector3f end(bounds.max());
        Eigen::Vector3f delta = (end - m_start) / m_resolution;
        m_map = cv::Mat((int)delta[1] + 1, (int)delta[0] + 1,CV_32FC1);
        m_map.setTo(-99999);
        m_map_dilated = cv::Mat((int)delta[1] + 1, (int)delta[0] + 1, CV_32FC1);
        m_map_dilated.setTo(-99999);
    	
        for (const auto& id_point : v_point_cloud) {
            const Point_3& point = v_point_cloud.point(id_point);
            float cur_height = m_map.at<float>((int)((point.y() - m_start[1])/m_resolution), (int)((point.x() - m_start[0])/m_resolution));
            if (cur_height < point.z())
                m_map.at<float>((int)((point.y() - m_start[1]) / m_resolution), (int)((point.x() - m_start[0]) / m_resolution)) = point.z();
        }
        cv::dilate(m_map, m_map_dilated, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 3);
    }

    Height_map(const Eigen::Vector3f& v_min, const Eigen::Vector3f& v_max, const float v_resolution)
	:m_resolution(v_resolution),m_start(v_min) {
        Eigen::Vector3f delta = (v_max - m_start) / m_resolution;
        m_map = cv::Mat((int)delta[1] + 1, (int)delta[0] + 1, CV_32FC1);
        m_map.setTo(std::numeric_limits<float>::lowest());
    }

	float get_height(float x,float y) const
    {
        int m_y = (int)((y - m_start[1]) / m_resolution);
        int m_x = (int)((x - m_start[0]) / m_resolution);
        if (!(0 <= m_y && 0 <= m_x && m_y < m_map.rows && m_x < m_map.cols))
            return std::numeric_limits<float>::lowest();
        return m_map_dilated.at<float>(m_y, m_x);
    }

    void update(const Eigen::AlignedBox3f& v_box)
    {
        int xmin = (v_box.min()[0] - m_start[0]) / m_resolution;
        int ymin = (v_box.min()[1] - m_start[1]) / m_resolution;
    	int xmax = (v_box.max()[0] - m_start[0]) / m_resolution+1;
        int ymax = (v_box.max()[1] - m_start[1]) / m_resolution+1;
        xmin = std::max(xmin, 0);
        xmax = std::min(xmax, m_map.cols-1);
        ymin = std::max(ymin, 0);
        ymax = std::min(ymax, m_map.rows-1);
        for (int y = ymin; y <= ymax; ++y)
            for (int x = xmin; x <= xmax; ++x)
                m_map.at<float>(y, x) = m_map.at<float>(y, x) > v_box.max()[2] ? m_map.at<float>(y, x) : v_box.max()[2];
        cv::dilate(m_map, m_map_dilated, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3)),cv::Point(-1,-1),3);
    }
	
    void save_height_map_png(const std::string& v_path,const float v_threshold=0.f)
    {
        //std::cout << m_map.rows() << "," << m_map.cols() << std::endl;
        cv::Mat map(m_map.rows, m_map.cols, CV_8UC3);
        for (int y = 0; y < m_map.rows; ++y)
            for (int x = 0; x < m_map.cols; ++x)
                if (m_map.at<float>(y, x) > v_threshold)
                    map.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
        cv::imwrite(v_path, map);
    }

    void save_height_map_tiff(const std::string& v_path) {
        //std::cout << m_map.rows() << "," << m_map.cols() << std::endl;
        cv::imwrite(v_path, m_map);
    }
	
    Eigen::Vector3f m_start;
    float m_resolution;
    cv::Mat m_map;
    cv::Mat m_map_dilated;
};

#endif // MODEL_TOOLS_H
