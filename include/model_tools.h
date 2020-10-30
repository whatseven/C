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
std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> load_obj(const std::string& v_mesh_name, const std::string& v_mtl_dir = "./");

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

std::string GetFileBasename(const std::string& FileName);

// @brief: Split the whole obj into small object and store them separatly. 
//         Each Object is store separatly and normalize near origin. The transformation is also stored in the txt
//         We also get a whole obj with "g" attribute to indicate each component. This obj can be imported into unreal with separate actor
// @notice: Be careful about the material file, you may adjust them manually
// @param: 
//          Directory that contains the obj file. The splited obj will also be stored in this directory
//          OBJ file name
//          Resolution indicates the resolution of the height map. (How far will the two building is considered to be one component)
// @ret:
void split_obj(const std::string& file_dir, const std::string& file_name, const float resolution);
#endif // MODEL_TOOLS_H
