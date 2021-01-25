#include "model_tools.h"
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

/*
Some useful function
*/

std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> load_obj(
	const std::string& v_mesh_name, bool v_log, const std::string& v_mtl_dir)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;
	bool ret;

	if (v_log)
		std::cout << "Read mesh " << v_mesh_name << " with TinyOBJLoader" << std::endl;
	ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
	                       v_mesh_name.c_str(), v_mtl_dir.c_str(), true);

	if (!warn.empty())
	{
		std::cout << warn << std::endl;
	}
	if (!err.empty())
	{
		std::cerr << err << std::endl;
	}
	if (!ret)
	{
		exit(1);
	}
	if (v_log)
	{
		int num_face = 0;
		for (const auto& shape : shapes)
			num_face += shape.mesh.indices.size() / 3;
		std::cout << "Read with " << attrib.vertices.size() / 3 << " vertices," << attrib.normals.size() / 3 << " normals," <<
			num_face << " faces" << std::endl;
	}
	return {attrib, shapes, materials};
}

Proxy load_footprint(const std::string& v_footprint_path)
{
	std::ifstream file_in(v_footprint_path);
	if (!file_in.is_open()) {
		std::cout << "No such file " << v_footprint_path << std::endl;
		return Proxy();
	}
	Proxy proxy;
	std::string line;
	std::getline(file_in, line);
	std::vector<std::string> tokens;
	boost::split(tokens, line, boost::is_any_of(" "));
	proxy.height = std::atof(tokens[1].c_str());
	std::getline(file_in, line);

	tokens.clear();
	boost::split(tokens, line, boost::is_any_of(" "));
	for (int i = 0; i < tokens.size(); i += 2)
	{
		if(tokens[i]=="")
			break;
		float x = std::atof(tokens[i].c_str());
		float y = std::atof(tokens[i+1].c_str());
		proxy.polygon.push_back(CGAL::Point_2<K>(x, y));
	}
	
	file_in.close();
	return proxy;
}

bool WriteMat(const std::string& filename, const std::vector<tinyobj::material_t>& materials)
{
	FILE* fp = fopen(filename.c_str(), "w");
	if (!fp)
	{
		fprintf(stderr, "Failed to open file [ %s ] for write.\n", filename.c_str());
		return false;
	}

	for (size_t i = 0; i < materials.size(); i++)
	{
		tinyobj::material_t mat = materials[i];

		fprintf(fp, "newmtl %s\n", mat.name.c_str());
		fprintf(fp, "Ka %f %f %f\n", mat.ambient[0], mat.ambient[1], mat.ambient[2]);
		fprintf(fp, "Kd %f %f %f\n", mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
		fprintf(fp, "Ks %f %f %f\n", mat.specular[0], mat.specular[1], mat.specular[2]);
		fprintf(fp, "Kt %f %f %f\n", mat.transmittance[0], mat.specular[1], mat.specular[2]);
		fprintf(fp, "Ke %f %f %f\n", mat.emission[0], mat.emission[1], mat.emission[2]);
		fprintf(fp, "d %f\n", mat.dissolve);
		fprintf(fp, "Ns %f\n", mat.shininess);
		fprintf(fp, "Ni %f\n", mat.ior);
		fprintf(fp, "illum %d\n", mat.illum);
		if (mat.diffuse_texname.size() > 2)
			fprintf(fp, "map_Kd %s\n", mat.diffuse_texname.c_str());
		if (mat.ambient_texname.size() > 2)
			fprintf(fp, "map_Ka %s\n", mat.ambient_texname.c_str());
		//fprintf(fp, "map_Kd %s\n", mat.diffuse_texname.c_str());
		fprintf(fp, "\n");
		// @todo { texture }
	}

	fclose(fp);

	return true;
}

bool write_obj(const std::string& filename, const tinyobj::attrib_t& attributes,
               const std::vector<tinyobj::shape_t>& shapes, const std::vector<tinyobj::material_t>& materials)
{
	FILE* fp = fopen(filename.c_str(), "w");
	if (!fp)
	{
		fprintf(stderr, "Failed to open file [ %s ] for write.\n", filename.c_str());
		return false;
	}

	boost::filesystem::path output_file(filename);
	std::string basename = output_file.filename().stem().string() + ".mtl";
	std::string material_filename = (output_file.parent_path()/ basename).string();

	int prev_material_id = -1;

	fprintf(fp, "mtllib %s\n\n", basename.c_str());

	// facevarying vtx
	for (size_t k = 0; k < attributes.vertices.size(); k += 3)
	{
		fprintf(fp, "v %f %f %f\n",
		        attributes.vertices[k + 0],
		        attributes.vertices[k + 1],
		        attributes.vertices[k + 2]);
	}

	fprintf(fp, "\n");

	// facevarying normal
	for (size_t k = 0; k < attributes.normals.size(); k += 3)
	{
		fprintf(fp, "vn %f %f %f\n",
		        attributes.normals[k + 0],
		        attributes.normals[k + 1],
		        attributes.normals[k + 2]);
	}

	fprintf(fp, "\n");

	// facevarying texcoord
	for (size_t k = 0; k < attributes.texcoords.size(); k += 2)
	{
		fprintf(fp, "vt %f %f\n",
		        attributes.texcoords[k + 0],
		        attributes.texcoords[k + 1]);
	}

	for (size_t i = 0; i < shapes.size(); i++)
	{
		fprintf(fp, "\n");

		if (shapes[i].name.empty())
		{
			fprintf(fp, "g %s\n", std::to_string(i));
		}
		else
		{
			//fprintf(fp, "use %s\n", shapes[i].name.c_str());
			fprintf(fp, "g %s\n", shapes[i].name.c_str());
		}

		bool has_vn = false;
		bool has_vt = false;
		// Assumes normals and textures are set shape-wise.
		if (shapes[i].mesh.indices.size() > 0)
		{
			has_vn = shapes[i].mesh.indices[0].normal_index != -1;
			has_vt = shapes[i].mesh.indices[0].texcoord_index != -1;
		}

		// face
		int face_index = 0;
		for (size_t k = 0; k < shapes[i].mesh.indices.size(); k += shapes[i].mesh.num_face_vertices[face_index++])
		{
			// Check Materials
			int material_id = shapes[i].mesh.material_ids[face_index];
			if (material_id != prev_material_id && material_id >= 0)
			{
				std::string material_name = materials[material_id].name;
				fprintf(fp, "usemtl %s\n", material_name.c_str());
				prev_material_id = material_id;
			}

			unsigned char v_per_f = shapes[i].mesh.num_face_vertices[face_index];
			// Imperformant, but if you want to have variable vertices per face, you need some kind of a dynamic loop.
			fprintf(fp, "f");
			for (int l = 0; l < v_per_f; l++)
			{
				const tinyobj::index_t& ref = shapes[i].mesh.indices[k + l];
				if (has_vn && has_vt)
				{
					// v0/t0/vn0
					fprintf(fp, " %d/%d/%d", ref.vertex_index + 1, ref.texcoord_index + 1, ref.normal_index + 1);
					continue;
				}
				if (has_vn && !has_vt)
				{
					// v0//vn0
					fprintf(fp, " %d//%d", ref.vertex_index + 1, ref.normal_index + 1);
					continue;
				}
				if (!has_vn && has_vt)
				{
					// v0/vt0
					fprintf(fp, " %d/%d", ref.vertex_index + 1, ref.texcoord_index + 1);
					continue;
				}
				if (!has_vn && !has_vt)
				{
					// v0 v1 v2
					fprintf(fp, " %d", ref.vertex_index + 1);
					continue;
				}
			}
			fprintf(fp, "\n");
		}
	}

	bool ret = WriteMat(material_filename, materials);

	fclose(fp);
	int num_face = 0;
	for (const auto& shape : shapes)
		num_face += shape.mesh.indices.size() / 3;
	std::cout << "Write with " << attributes.vertices.size() / 3 << " vertices," << attributes.normals.size() / 3 << " normals," <<
		num_face << " faces" << std::endl;
	return 1;
}

void fix_mtl_from_unreal(const std::string& filename)
{
	std::ifstream f_in(filename);
	std::string buffer;

	if (!f_in.is_open())
	{
		std::cout << "No such file " << filename << std::endl;
		return;
	}

	int material_idx = -1;
	while (!f_in.eof())
	{
		std::string line;
		std::getline(f_in, line);
		if (line.find("newmtl") != line.npos)
		{
			material_idx += 1;
			buffer += "\n newmtl material_" + std::to_string(material_idx);
		}
		else
		{
			if (material_idx == -1) //First line
				continue;
			else
				buffer += "\n" + line;
		}
	}
	f_in.close();
	std::ofstream f_out(filename);
	f_out << buffer;
	f_out.close();
}

void clean_vertex(tinyobj::attrib_t& attrib, tinyobj::shape_t& shape)
{
	// Find out used vertex, mark true
	std::vector<bool> vertex_used(attrib.vertices.size() / 3, false);
	std::vector<bool> tex_used(attrib.texcoords.size() / 2, false);
	for (size_t face_id = 0; face_id < shape.mesh.num_face_vertices.size(); face_id++)
	{
		size_t index_offset = 3 * face_id;
		tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
		tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
		tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
		vertex_used[idx0.vertex_index] = true;
		vertex_used[idx1.vertex_index] = true;
		vertex_used[idx2.vertex_index] = true;
		tex_used[idx0.texcoord_index] = true;
		tex_used[idx1.texcoord_index] = true;
		tex_used[idx2.texcoord_index] = true;
	}
	// Filter out vertex, normals, texcoords
	attrib.vertices.erase(std::remove_if(attrib.vertices.begin(), attrib.vertices.end(), [&](const tinyobj::real_t& idx)
	{
		return vertex_used[(&idx - &*attrib.vertices.begin()) / 3] == false;
	}), attrib.vertices.end());
	attrib.normals.erase(std::remove_if(attrib.normals.begin(), attrib.normals.end(), [&](const tinyobj::real_t& idx)
	{
		return vertex_used[(&idx - &*attrib.normals.begin()) / 3] == false;
	}), attrib.normals.end());
	attrib.texcoords.erase(std::remove_if(attrib.texcoords.begin(), attrib.texcoords.end(),
	                                      [&](const tinyobj::real_t& idx)
	                                      {
		                                      return tex_used[(&idx - &*attrib.texcoords.begin()) / 2] == false;
	                                      }), attrib.texcoords.end());

	// Create redirect index map
	std::vector<size_t> vertex_redirect;
	std::vector<size_t> tex_redirect;
	size_t current_vertex_id = 0;
	size_t current_tex_id = 0;
	for (size_t vertex_id = 0; vertex_id < vertex_used.size(); vertex_id++)
	{
		if (vertex_used[vertex_id])
		{
			vertex_redirect.push_back(current_vertex_id + 1);
			current_vertex_id += 1;
		}
		else
			vertex_redirect.push_back(0);
	}
	for (size_t tex_id = 0; tex_id < tex_used.size(); tex_id++)
	{
		if (tex_used[tex_id])
		{
			tex_redirect.push_back(current_tex_id + 1);
			current_tex_id += 1;
		}
		else
			tex_redirect.push_back(0);
	}

	// Adjust index from face to vertex according to the vertex_redirect array
	// Also delete duplicated faces
	std::set<std::tuple<int, int, int, int, int, int, int, int, int>> face_already_assigned;
	std::vector<bool> face_should_delete;
	for (size_t face_id = 0; face_id < shape.mesh.num_face_vertices.size(); face_id++)
	{
		if (shape.mesh.num_face_vertices[face_id] != 3)
			throw;
		size_t index_offset = 3 * face_id;
		tinyobj::index_t& idx0 = shape.mesh.indices[index_offset + 0];
		tinyobj::index_t& idx1 = shape.mesh.indices[index_offset + 1];
		tinyobj::index_t& idx2 = shape.mesh.indices[index_offset + 2];

		auto key = std::make_tuple(idx0.vertex_index, idx1.vertex_index, idx2.vertex_index,
			idx0.normal_index, idx1.normal_index, idx2.normal_index,
			idx0.texcoord_index, idx1.texcoord_index, idx2.texcoord_index);

		if (face_already_assigned.insert(key).second)
			face_should_delete.push_back(false);
		else
			face_should_delete.push_back(true);
		
		idx0.vertex_index = vertex_redirect[idx0.vertex_index] - 1;
		idx0.normal_index = vertex_redirect[idx0.normal_index] - 1;
		idx0.texcoord_index = tex_redirect[idx0.texcoord_index] - 1;
		idx1.vertex_index = vertex_redirect[idx1.vertex_index] - 1;
		idx1.normal_index = vertex_redirect[idx1.normal_index] - 1;
		idx1.texcoord_index = tex_redirect[idx1.texcoord_index] - 1;
		idx2.vertex_index = vertex_redirect[idx2.vertex_index] - 1;
		idx2.normal_index = vertex_redirect[idx2.normal_index] - 1;
		idx2.texcoord_index = tex_redirect[idx2.texcoord_index] - 1;

		assert(idx0.vertex_index != -1);
		assert(idx0.normal_index != -1);
		assert(idx0.texcoord_index != -1);
		assert(idx1.vertex_index != -1);
		assert(idx1.normal_index != -1);
		assert(idx1.texcoord_index != -1);
		assert(idx2.vertex_index != -1);
		assert(idx2.normal_index != -1);
		assert(idx2.texcoord_index != -1);
		
	}
	shape.mesh.indices.erase(
		std::remove_if(shape.mesh.indices.begin(), shape.mesh.indices.end(),
		               [&,index=0](const tinyobj::index_t& idx) mutable
		               {
			               return face_should_delete[index++ / 3] == true;
		               }),
		shape.mesh.indices.end());
	shape.mesh.material_ids.erase(
		std::remove_if(shape.mesh.material_ids.begin(), shape.mesh.material_ids.end(),
			[&, index = 0](const auto& idx) mutable
	{
		return face_should_delete[index++] == true;
	}),
		shape.mesh.material_ids.end());
	shape.mesh.num_face_vertices.erase(
		std::remove_if(shape.mesh.num_face_vertices.begin(), shape.mesh.num_face_vertices.end(),
			[&, index = 0](const auto& idx) mutable
	{
		return face_should_delete[index++] == true;
	}),
		shape.mesh.num_face_vertices.end());
	
}


/*
Get split mesh with a big whole mesh
*/

void merge_obj(const std::string& v_file,
               const std::vector<tinyobj::attrib_t>& v_attribs, const std::vector<tinyobj::shape_t>& saved_shapes,
               const std::vector<tinyobj::_material_t>& materials,
			   const int start_id)
{
	FILE* fp = fopen(v_file.c_str(), "w");
	if (!fp)
	{
		fprintf(stderr, "Failed to open file [ %s ] for write.\n", v_file.c_str());
		return;
	}

	boost::filesystem::path output_file(v_file);
	std::string basename = output_file.filename().stem().string() + ".mtl";
	std::string material_filename = (output_file.parent_path() / basename).string();

	int prev_material_id = -1;

	fprintf(fp, "mtllib %s\n\n", basename.c_str());
	
	size_t vertex_already_assigned = 0;
	size_t tex_already_assigned = 0;
	for (int i_mesh = 0; i_mesh < v_attribs.size(); i_mesh += 1)
	{
		const auto& attributes = v_attribs[i_mesh];
		// vertex
		for (size_t k = 0; k < attributes.vertices.size(); k += 3)
		{
			fprintf(fp, "v %f %f %f\n",
			        attributes.vertices[k + 0],
			        attributes.vertices[k + 1],
			        attributes.vertices[k + 2]);
		}

		fprintf(fp, "\n");

		// normal
		for (size_t k = 0; k < attributes.normals.size(); k += 3)
		{
			fprintf(fp, "vn %f %f %f\n",
			        attributes.normals[k + 0],
			        attributes.normals[k + 1],
			        attributes.normals[k + 2]);
		}

		fprintf(fp, "\n");

		// facevarying texcoord
		for (size_t k = 0; k < attributes.texcoords.size(); k += 2)
		{
			fprintf(fp, "vt %f %f\n",
			        attributes.texcoords[k + 0],
			        attributes.texcoords[k + 1]);
		}
	}
	for (int i_mesh = 0; i_mesh < v_attribs.size(); i_mesh += 1)
	{
		// Mesh
		const auto& attributes = v_attribs[i_mesh];
		const auto& shapes = saved_shapes[i_mesh];

		fprintf(fp, "\n");

		fprintf(fp, "g %s\n", std::to_string(i_mesh + start_id).c_str());

		// face
		int face_index = 0;
		for (size_t k = 0; k < shapes.mesh.indices.size(); k += shapes.mesh.num_face_vertices[face_index++])
		{
			// Check Materials
			int material_id = shapes.mesh.material_ids[face_index];
			if (material_id != prev_material_id)
			{
				std::string material_name = materials[material_id].name;
				fprintf(fp, "usemtl %s\n", material_name.c_str());
				prev_material_id = material_id;
			}

			unsigned char v_per_f = shapes.mesh.num_face_vertices[face_index];
			// Imperformant, but if you want to have variable vertices per face, you need some kind of a dynamic loop.
			fprintf(fp, "f");
			for (int l = 0; l < v_per_f; l++)
			{
				const tinyobj::index_t& ref = shapes.mesh.indices[k + l];
				// v0/t0/vn0
				fprintf(fp, " %d/%d/%d", ref.vertex_index + 1 + vertex_already_assigned,
				        ref.texcoord_index + 1 + tex_already_assigned, ref.normal_index + 1 + vertex_already_assigned);
			}
			fprintf(fp, "\n");
		}
		vertex_already_assigned += attributes.vertices.size() / 3;
		tex_already_assigned += attributes.texcoords.size() / 2;
	}
	fclose(fp);
	bool ret = WriteMat(material_filename, materials);
}

void split_obj(const std::string& file_dir, const std::string& file_name, const float resolution,const float v_filter_height, const int obj_max_builidng_num)
{
	std::cout << "----------Start split obj----------" << std::endl;

	const float Z_THRESHOLD = v_filter_height;
	std::cout << "1/6 Read mesh" << std::endl;
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::tie(attrib, shapes, materials) = load_obj(file_dir + "/" + file_name, true, file_dir);

	// Calculate bounding box
	std::cout << "2/6 Calculate bounding box" << std::endl;
	float xmin = 1e8, ymin = 1e8, zmin = 1e8;
	float xmax = -1e8, ymax = -1e8, zmax = -1e8;
	for (size_t s = 0; s < shapes.size(); s++)
	{
		size_t index_offset = 0;
		for (size_t face_id = 0; face_id < shapes[s].mesh.num_face_vertices.size(); face_id++)
		{
			if (shapes[s].mesh.num_face_vertices[face_id] != 3)
				throw;
			for (size_t v = 0; v < 3; v++)
			{
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

				xmin = xmin < vx ? xmin : vx;
				ymin = ymin < vy ? ymin : vy;
				zmin = zmin < vz ? zmin : vz;

				xmax = xmax > vx ? xmax : vx;
				ymax = ymax > vy ? ymax : vy;
				zmax = zmax > vz ? zmax : vz;
			}
			index_offset += 3;
		}
	}

	// Construct height map
	std::cout << "3/6 Construct height map" << std::endl;
	cv::Mat img((int)((ymax - ymin) / resolution) + 1, (int)((xmax - xmin) / resolution) + 1, CV_32FC1,
	            cv::Scalar(0.f));
	std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> records(img.rows,
	                                                                         std::vector<std::vector<std::pair<
		                                                                         size_t, size_t>>>(
		                                                                         img.cols,
		                                                                         std::vector<std::pair<size_t, size_t>
		                                                                         >()));

	for (size_t s = 0; s < shapes.size(); s++)
	{
		for (size_t face_id = 0; face_id < shapes[s].mesh.num_face_vertices.size(); face_id++)
		{
			size_t index_offset = 3 * face_id;

			tinyobj::index_t idx0 = shapes[s].mesh.indices[index_offset + 0];
			tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset + 1];
			tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + 2];
			const auto& vertex0_x = attrib.vertices[3 * idx0.vertex_index + 0];
			const auto& vertex0_y = attrib.vertices[3 * idx0.vertex_index + 1];
			const auto& vertex1_x = attrib.vertices[3 * idx1.vertex_index + 0];
			const auto& vertex1_y = attrib.vertices[3 * idx1.vertex_index + 1];
			const auto& vertex2_x = attrib.vertices[3 * idx2.vertex_index + 0];
			const auto& vertex2_y = attrib.vertices[3 * idx2.vertex_index + 1];
			int ymin_item = (std::min({vertex0_y, vertex1_y, vertex2_y}) - ymin) / resolution;
			int xmin_item = (std::min({vertex0_x, vertex1_x, vertex2_x}) - xmin) / resolution;
			int xmax_item = (std::max({vertex0_x, vertex1_x, vertex2_x}) - xmin) / resolution;
			int ymax_item = (std::max({vertex0_y, vertex1_y, vertex2_y}) - ymin) / resolution;

			typedef CGAL::Simple_cartesian<int> K;
			CGAL::Triangle_2<K> t1(
				CGAL::Point_2<K>((vertex0_x - xmin) / resolution, (vertex0_y - ymin) / resolution),
				CGAL::Point_2<K>((vertex1_x - xmin) / resolution, (vertex1_y - ymin) / resolution),
				CGAL::Point_2<K>((vertex2_x - xmin) / resolution, (vertex2_y - ymin) / resolution)
			);
			for (int x = xmin_item; x < xmax_item + 1; ++x)
				for (int y = ymin_item; y < ymax_item + 1; ++y)
				{
					if (t1.has_on_bounded_side(CGAL::Point_2<K>(x, y))
						|| t1.has_on_boundary(CGAL::Point_2<K>(x, y)))
					{
						img.at<float>(y, x) = 255;
						records[y][x].push_back(std::make_pair(s, face_id));
					}
				}
		}
	}
	cv::imwrite(file_dir + "/bird_view.jpg", img);

	// Travel to find connect component
	std::cout << "4/6 Travel to find connect component" << std::endl;
	cv::Mat img_traveled(img.rows, img.cols,CV_32FC1, cv::Scalar(0.f));
	struct Building_Cluster
	{
		std::vector<int> xs;
		std::vector<int> ys;
	};
	std::vector<Building_Cluster> buildings;

	std::queue<std::pair<int, int>> node;
	for (int x = 0; x < img.cols; ++x)
	{
		for (int y = 0; y < img.rows; ++y)
		{
			if (img_traveled.at<float>(y, x) == 0)
			{
				Building_Cluster building;
				node.push(std::make_pair(x, y));
				while (!node.empty())
				{
					auto cur = node.front();
					node.pop();
					int cur_x = cur.first;
					int cur_y = cur.second;
					if (cur_x < 0 || cur_y < 0 || cur_x >= img.cols || cur_y >= img.rows)
						continue;;
					if (img.at<float>(cur_y, cur_x) == 0 || img_traveled.at<float>(cur_y, cur_x) != 0)
						continue;;

					building.xs.push_back(cur_x);
					building.ys.push_back(cur_y);
					img_traveled.at<float>(cur_y, cur_x) = 255;
					node.push(std::make_pair(cur_x - 1, cur_y - 1));
					node.push(std::make_pair(cur_x - 1, cur_y));
					node.push(std::make_pair(cur_x - 1, cur_y + 1));
					node.push(std::make_pair(cur_x, cur_y - 1));
					node.push(std::make_pair(cur_x, cur_y + 1));
					node.push(std::make_pair(cur_x + 1, cur_y - 1));
					node.push(std::make_pair(cur_x + 1, cur_y));
					node.push(std::make_pair(cur_x + 1, cur_y + 1));
				}
				if (building.xs.size() > 0)
					buildings.push_back(building);
			}
		}
	}

	std::cout << buildings.size() << " in total\n";

	// Save
	std::cout << "5/6 Save splited model" << std::endl;
	std::vector<tinyobj::shape_t> saved_shapes;
	std::vector<tinyobj::attrib_t> saved_attrib;
	int building_num = 0;
	int obj_num = 0;
	for (int building_idx = 0; building_idx < buildings.size(); building_idx++)
	{
		tinyobj::attrib_t cur_attr = tinyobj::attrib_t(attrib);
		tinyobj::shape_t cur_shape;

		int area_2d = buildings[building_idx].xs.size();
		if (area_2d <= 1)
			continue;

		int x_center = *std::max_element(buildings[building_idx].xs.begin(), buildings[building_idx].xs.end()) + (*
			std::min_element(buildings[building_idx].xs.begin(), buildings[building_idx].xs.end()));
		int y_center = *std::max_element(buildings[building_idx].ys.begin(), buildings[building_idx].ys.end()) + (*
			std::min_element(buildings[building_idx].ys.begin(), buildings[building_idx].ys.end()));
		x_center = x_center / 2 * resolution + xmin;
		y_center = y_center / 2 * resolution + ymin;

		std::map<Eigen::VectorXf, int> vertex_already_assigned;

		for (int pixel_id = 0; pixel_id < area_2d; pixel_id += 1)
		{
			int x = buildings[building_idx].xs[pixel_id];
			int y = buildings[building_idx].ys[pixel_id];

			for (const auto mesh_id : records[y][x])
			{
				tinyobj::index_t idx0 = shapes[mesh_id.first].mesh.indices[mesh_id.second * 3 + 0];
				tinyobj::index_t idx1 = shapes[mesh_id.first].mesh.indices[mesh_id.second * 3 + 1];
				tinyobj::index_t idx2 = shapes[mesh_id.first].mesh.indices[mesh_id.second * 3 + 2];

				cur_shape.mesh.num_face_vertices.push_back(3);
				cur_shape.mesh.material_ids.push_back(shapes[mesh_id.first].mesh.material_ids[mesh_id.second]);
				cur_shape.mesh.indices.push_back(idx0);
				cur_shape.mesh.indices.push_back(idx1);
				cur_shape.mesh.indices.push_back(idx2);
			}
		}

		cur_shape.name = std::to_string(building_num);
		clean_vertex(cur_attr, cur_shape);
		bool preserve_flag = false;
		for (int i = 0; i < cur_attr.vertices.size() / 3; ++i)
		{
			float cur_z = cur_attr.vertices[i * 3 + 2];
			if (cur_z > Z_THRESHOLD)
			{
				preserve_flag = true;
				break;
			}
		}
		if (!preserve_flag)
			continue;

		saved_shapes.push_back(cur_shape);
		saved_attrib.push_back(cur_attr);

		if (obj_max_builidng_num > 0 && saved_shapes.size() >= obj_max_builidng_num)
		{
			std::cout << "5.5/6 Save max num split obj" << std::endl;
			merge_obj(file_dir + "/" + "total_split" + std::to_string(obj_num) + ".obj", saved_attrib, saved_shapes, materials, building_num - obj_max_builidng_num + 1);
			std::cout << "----------Partial Save done----------" << std::endl << std::endl;
			saved_shapes.clear();
			saved_attrib.clear();
			obj_num++;
		}

		for (int i_vertex = 0; i_vertex < cur_attr.vertices.size(); i_vertex += 1)
		{
			if (i_vertex % 3 == 0)
				cur_attr.vertices[i_vertex] -= x_center;
			else if (i_vertex % 3 == 1)
				cur_attr.vertices[i_vertex] -= y_center;
		}

		std::ofstream f_out(file_dir + "/" + std::to_string(building_num) + ".txt");
		f_out << x_center << "," << y_center << std::endl;
		f_out.close();

		write_obj(file_dir + "/" + std::to_string(building_num) + ".obj", cur_attr,
		          std::vector<tinyobj::shape_t>{cur_shape}, materials);
		building_num += 1;
	}
	std::cout << "6/6 Save whole split obj" << std::endl;
	merge_obj(file_dir + "/" + "total_split" + std::to_string(obj_num) + ".obj", saved_attrib, saved_shapes, materials, building_num - saved_attrib.size() + 1);
	std::cout << "----------Split obj done----------" << std::endl << std::endl;

}

void rename_material(const std::string& file_dir, const std::string& file_name, const std::string& v_output_dir)
{
	std::cout << "----------Start rename material----------" << std::endl;

	boost::filesystem::path output_root(v_output_dir);
	boost::system::error_code ec;
	try
	{
		if (boost::filesystem::exists(output_root))
			boost::filesystem::remove_all(output_root, ec);
		boost::filesystem::create_directories(v_output_dir);
		boost::filesystem::create_directories(v_output_dir + "/mat");
	}
	catch (...)
	{
		std::cout << ec << std::endl;
	}

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::tie(attrib, shapes, materials) = load_obj(file_dir + "/" + file_name, true, file_dir);
	std::map<std::string, std::string> texture_set;
	for (int i = 0; i < materials.size(); ++i)
	{
		//continue;
		auto& material = materials[i];

		if (material.diffuse_texname.size() <= 2)
		{
			//std::cout << material.name << std::endl;
			continue;
		}
		material.name = (boost::format("m_%d") % i).str();


		std::string img_name_old = material.diffuse_texname;
		std::string img_name_new;
		if (texture_set.find(img_name_old) != texture_set.end())
		{
			img_name_new = texture_set.at(img_name_old);
		}
		else
		{
			std::string extension_name = boost::filesystem::path(img_name_old).extension().string();

			img_name_new = (boost::format("mat/tex_%s%s") % texture_set.size() % extension_name).str();
			texture_set.insert(std::make_pair(img_name_old, img_name_new));

			boost::filesystem::path img_path_old(file_dir);
			img_path_old = img_path_old / img_name_old;
			if (!boost::filesystem::exists(img_path_old))
				throw;
			boost::filesystem::copy_file(img_path_old, output_root / img_name_new,
			                             boost::filesystem::copy_option::overwrite_if_exists);
		}
		material.diffuse_texname = img_name_new;
		if (material.ambient_texname == img_name_old)
			material.ambient_texname = img_name_new;
	}
	write_obj((output_root / "renamed_material.obj").string(), attrib, shapes, materials);
	std::cout << "----------Rename material done----------" << std::endl << std::endl;

	return;
}

std::vector<tinyobj::shape_t> split_obj_according_to_footprint(const tinyobj::attrib_t& v_attribs, const std::vector<tinyobj::shape_t>& v_shapes, const std::vector<Proxy>& v_proxys,const float v_squared_threshold)
{
	std::vector<tinyobj::shape_t> out_shapes(v_proxys.size());
	tinyobj::shape_t ground_shape;

	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		const tinyobj::mesh_t& shape = v_shapes[i_shape].mesh;
		size_t num_faces = shape.num_face_vertices.size();
		for (size_t face_id = 0; face_id < num_faces; face_id++) {
			size_t index_offset = 3 * face_id;

			tinyobj::index_t idx0 = shape.indices[index_offset + 0];
			tinyobj::index_t idx1 = shape.indices[index_offset + 1];
			tinyobj::index_t idx2 = shape.indices[index_offset + 2];
			CGAL::Point_2<K> point1(v_attribs.vertices[3 * idx0.vertex_index + 0], v_attribs.vertices[3 * idx0.vertex_index + 1]);
			CGAL::Point_2<K> point2(v_attribs.vertices[3 * idx1.vertex_index + 0], v_attribs.vertices[3 * idx1.vertex_index + 1]);
			CGAL::Point_2<K> point3(v_attribs.vertices[3 * idx2.vertex_index + 0], v_attribs.vertices[3 * idx2.vertex_index + 1]);

			int preserved = -1;
			bool close_to_boundary = -1;
			for (int i_proxy = 0; i_proxy < v_proxys.size(); i_proxy++) {
				const Proxy& proxy = v_proxys[i_proxy];
				for (auto iter_edge = proxy.polygon.edges_begin(); iter_edge != proxy.polygon.edges_end(); ++iter_edge)
				{
					if (CGAL::squared_distance(point1, (*iter_edge)) < v_squared_threshold ||
						CGAL::squared_distance(point2, (*iter_edge)) < v_squared_threshold ||
						CGAL::squared_distance(point3, (*iter_edge)) < v_squared_threshold) {
						close_to_boundary = i_proxy;
						break;
					}
				}
				
				if(proxy.polygon.bounded_side(point1) == CGAL::Bounded_side::ON_BOUNDED_SIDE||
					proxy.polygon.bounded_side(point2) == CGAL::Bounded_side::ON_BOUNDED_SIDE||
					proxy.polygon.bounded_side(point3) == CGAL::Bounded_side::ON_BOUNDED_SIDE)
				{
					preserved = i_proxy;
					break;
				}
			}

			tinyobj::mesh_t* mesh;

			if (preserved!=-1) 
				mesh = &out_shapes[preserved].mesh;
			//else if(close_to_boundary!=-1)
			//	mesh = &out_shapes[close_to_boundary].mesh;
			else
				mesh = &ground_shape.mesh;
			mesh->material_ids.push_back(shape.material_ids[face_id]);
			mesh->num_face_vertices.push_back(shape.num_face_vertices[face_id]);
			mesh->indices.push_back(shape.indices[3 * face_id + 0]);
			mesh->indices.push_back(shape.indices[3 * face_id + 1]);
			mesh->indices.push_back(shape.indices[3 * face_id + 2]);
		}
	}
	out_shapes.push_back(ground_shape);

	return out_shapes;
}

/*
	xmin xmax ymin ymax zmax in sequence
*/
std::vector<float> get_bounds(std::string path, float v_bounds)
{
	std::vector<float> output;
	std::ifstream ObjFile(path);
	std::string line;
	float xmin = 99999;
	float xmax = -99999;
	float ymin = 99999;
	float ymax = -99999;
	float zmax = -99999;
	while (getline(ObjFile, line))
	{
		std::vector<std::string> vData;
		if (line.substr(0, line.find(" ")) == "v")
		{
			line = line.substr(line.find(" ") + 1);
			for (int i = 0; i < 3; i++)
			{
				vData.push_back(line.substr(0, line.find(" ")));
				line = line.substr(line.find(" ") + 1);
			}
			vData[2] = vData[2].substr(0, vData[2].find("\n"));
			float x = atof(vData[0].c_str());
			float y = atof(vData[1].c_str());
			float z = atof(vData[2].c_str()) + v_bounds;
			xmin = std::min(x, xmin);
			xmax = std::max(x, xmax);
			ymin = std::min(y, ymin);
			ymax = std::max(y, ymax);
			zmax = std::max(z, zmax);
		}
	}
	output.push_back(xmin);
	output.push_back(xmax);
	output.push_back(ymin);
	output.push_back(ymax);
	output.push_back(zmax);
	return output;
}


std::vector<Polygon_2> get_polygons(std::string file_path)
{
	std::ifstream inputFile;
	inputFile.open(file_path);
	if (!inputFile)
	{
		std::cout << "can't find file" << std::endl;
	}
	std::vector<std::vector<float>> temp_data;
	while (!inputFile.eof())
	{
		std::string x_temp, y_temp, index;
		inputFile >> x_temp >> y_temp >> index;
		temp_data.push_back(std::vector<float>{float(atof(x_temp.c_str())), float(atof(y_temp.c_str())), float(atof(index.c_str()))});
	}
	inputFile.close();
	float now_index = temp_data[0][2];
	std::vector<Polygon_2> polygon_vector;
	Polygon_2 polygon;
	for (auto point : temp_data)
	{
		if (point[2] == now_index)
		{
			polygon.push_back(Point_2(point[0], point[1]));
		}
		else
		{
			if (polygon.is_simple())
				polygon_vector.push_back(polygon);
			polygon = Polygon_2();
			polygon.push_back(Point_2(point[0], point[1]));
			now_index = point[2];
		}
	}
	return polygon_vector;
}


float point_box_distance_eigen(const Eigen::Vector2f& v_pos, const Eigen::AlignedBox2f& v_box) {
	float sqDist = 0.0f;
	float x = v_pos.x();
	if (x < v_box.min().x()) sqDist += (v_box.min().x() - x) * (v_box.min().x() - x);
	if (x > v_box.max().x()) sqDist += (x - v_box.max().x()) * (x - v_box.max().x());

	float y = v_pos.y();
	if (y < v_box.min().y()) sqDist += (v_box.min().y() - y) * (v_box.min().y() - y);
	if (y > v_box.max().y()) sqDist += (y - v_box.max().y()) * (y - v_box.max().y());
	return std::sqrt(sqDist);
}


bool inside_box(const Eigen::Vector2f& v_pos, const Eigen::AlignedBox2f& v_box) {
	return v_pos.x() >= v_box.min().x() && v_pos.x() <= v_box.max().x() && v_pos.y() >= v_box.min().y() && v_pos.y() <= v_box.max().y();
}

Surface_mesh get_box_mesh(const std::vector<Eigen::AlignedBox3f>& v_boxes)
{
	Surface_mesh mesh;

	for(const auto& item: v_boxes)
	{
		auto v0=mesh.add_vertex(Point_3(item.min().x(), item.min().y(), item.min().z()));
		auto v1=mesh.add_vertex(Point_3(item.min().x(), item.max().y(), item.min().z()));
		auto v2=mesh.add_vertex(Point_3(item.max().x(), item.max().y(), item.min().z()));
		auto v3=mesh.add_vertex(Point_3(item.max().x(), item.min().y(), item.min().z()));
		auto v4 = mesh.add_vertex(Point_3(item.min().x(), item.min().y(), item.max().z()));
		auto v5 = mesh.add_vertex(Point_3(item.min().x(), item.max().y(), item.max().z()));
		auto v6 = mesh.add_vertex(Point_3(item.max().x(), item.max().y(), item.max().z()));
		auto v7 = mesh.add_vertex(Point_3(item.max().x(), item.min().y(), item.max().z()));

		mesh.add_face(v0, v1, v2,v3);
		mesh.add_face(v4, v7, v6,v5);
		mesh.add_face(v0, v4, v5,v1);
		mesh.add_face(v3, v2, v6,v7);
		mesh.add_face(v0, v3, v7,v4);
		mesh.add_face(v1, v5, v6,v2);
	}
	return mesh;
}

void get_box_mesh_with_colors(const std::vector<Eigen::AlignedBox3f>& v_boxes,
	const std::vector<cv::Vec3b>& v_colors,const std::string& v_name) {
	std::ofstream f(v_name);
	int index = 0;
	for (const auto& item : v_boxes) {
		f << (boost::format("v %s %s %s %s %s %s\n") % item.min().x() % item.min().y() % item.min().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.min().x() % item.max().y() % item.min().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.max().x() % item.max().y() % item.min().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.max().x() % item.min().y() % item.min().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.min().x() % item.min().y() % item.max().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.min().x() % item.max().y() % item.max().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.max().x() % item.max().y() % item.max().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();
		f << (boost::format("v %s %s %s %s %s %s\n") % item.max().x() % item.min().y() % item.max().z() % ((float)v_colors[index][2] / 255.f) % ((float)v_colors[index][1] / 255.f) % ((float)v_colors[index][0] / 255.f)).str();

		f << (boost::format("f %s %s %s %s\n")% (index * 8 + 1 +0) % (index * 8 + 1 + 1) % (index * 8 + 1 + 2) % (index * 8 + 1 + 3)).str();
		f << (boost::format("f %s %s %s %s\n")% (index * 8 + 1 +4) % (index * 8 + 1 + 7) % (index * 8 + 1 + 6) % (index * 8 + 1 + 5)).str();
		f << (boost::format("f %s %s %s %s\n")% (index * 8 + 1 +0) % (index * 8 + 1 + 4) % (index * 8 + 1 + 5) % (index * 8 + 1 + 1)).str();
		f << (boost::format("f %s %s %s %s\n")% (index * 8 + 1 +3) % (index * 8 + 1 + 2) % (index * 8 + 1 + 6) % (index * 8 + 1 + 7)).str();
		f << (boost::format("f %s %s %s %s\n")% (index * 8 + 1 +0) % (index * 8 + 1 + 3) % (index * 8 + 1 + 7) % (index * 8 + 1 + 4)).str();
		f << (boost::format("f %s %s %s %s\n")% (index * 8 + 1 +1) % (index * 8 + 1 + 5) % (index * 8 + 1 + 6) % (index * 8 + 1 + 2)).str();
		index += 1;
	}
	f.close();
	return;
}