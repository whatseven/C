
#include "object_detection_tools.h"


void addTextNode(tx2::XMLDocument& vDoc,tx2::XMLElement* rootElement, std::string vName, int vVal) {
	using namespace tinyxml2;
	XMLElement* element = vDoc.NewElement(vName.c_str());
	XMLText* genderText = vDoc.NewText(std::to_string(vVal).c_str());
	element->InsertEndChild(genderText);
	rootElement->InsertEndChild(element);
}

void writeItem_1216(const std::string& vPath, const std::vector<CGAL::Bbox_2> building_bboxes, int vWidth) {
	using namespace tinyxml2;
	XMLDocument doc;
	XMLElement* root = doc.NewElement("annotation");

	XMLElement* sizeElement = doc.NewElement("size");
	addTextNode(doc, sizeElement, "depth", 3);
	addTextNode(doc, sizeElement, "width", vWidth);
	addTextNode(doc, sizeElement, "height", vWidth);
	root->InsertEndChild(sizeElement);

	int id = 0;
	for (auto item : building_bboxes) {
		XMLElement* objectElement = doc.NewElement("object");
		XMLElement* nameElement = doc.NewElement("name");
		XMLText* nameText = doc.NewText(std::to_string(id).c_str());
		nameElement->InsertEndChild(nameText);
		objectElement->InsertEndChild(nameElement);

		addTextNode(doc, objectElement, "difficult", 0);

		XMLElement* bndElement = doc.NewElement("bndbox");
		addTextNode(doc, bndElement, "xmin", item.xmin());
		addTextNode(doc, bndElement, "ymin", item.ymin());
		addTextNode(doc, bndElement, "xmax", item.xmax());
		addTextNode(doc, bndElement, "ymax", item.ymax());

		objectElement->InsertEndChild(bndElement);
		root->InsertEndChild(objectElement);
		id++;
	}
	doc.InsertEndChild(root);
	doc.SaveFile(vPath.c_str());
}

std::vector<cv::Rect> getBoundingBoxes(const std::vector<cv::Point>& pixel_points, const std::vector<std::vector<bool>>& is_same_color)
{
	cv::Rect now_rect = cv::boundingRect(pixel_points);
	const int img_width = now_rect.width;
	const int img_height = now_rect.height;
	std::vector<std::vector<int>> label(img_height, std::vector<int>(img_width, 0));
	std::vector<std::vector<int>> label_neighbors;
	std::vector<cv::Rect> output;
	int now_label = 1;
	for (int y = 0; y < label.size(); y++)
	{
		for (int x = 0; x < label[y].size(); x++)
		{
			// up down left right
			std::vector<int> neighbors(4, 0);
			if (!is_same_color[y][x])
			{
				label[y][x] = 0;
			}
			else
			{
				if (x != 0)
					neighbors[2] = label[y][x - 1];
				if (x != label[y].size() - 1)
					neighbors[3] = label[y][x + 1];
				if (y != 0)
					neighbors[0] = label[y - 1][x];
				if (y != label.size() - 1)
					neighbors[1] = label[y + 1][x];

				float min_label = 99999;
				// find min label or all zero
				{
					for (int i = 0; i < neighbors.size(); i++)
					{
						if (neighbors[i] != 0 && neighbors[i] < min_label)
							min_label = neighbors[i];
					}
				}

				if (min_label == 99999)
				{
					label[y][x] = now_label;
					label_neighbors.resize(now_label);
					now_label++;
				}
				else
				{
					label[y][x] = min_label;
					// Merge Neighbor relation
					for (int i = 0; i < neighbors.size(); i++)
					{
						if (neighbors[i] != 0)
						{
							auto neighbor_find_result = std::find(label_neighbors[neighbors[i] - 1].begin(), label_neighbors[neighbors[i] - 1].end(), label[y][x]);
							if (neighbor_find_result == label_neighbors[neighbors[i] - 1].end() && neighbors[i] > label[y][x])
							{
								label_neighbors[neighbors[i] - 1].push_back(label[y][x]);
							}
						}
					}
				}
			}
		}
	}

	std::vector<std::vector<cv::Point>> building_points;
	for (int y = 0; y < label.size(); y++)
	{
		for (int x = 0; x < label[y].size(); x++)
		{
			now_label = label[y][x];
			if (now_label == 0)
				continue;
			while (true)
			{
				if (label_neighbors[now_label - 1].size() == 0)
					break;
				else
					now_label = *std::min_element(label_neighbors[now_label - 1].begin(), label_neighbors[now_label - 1].end());
			}
			
			label[y][x] = now_label;
			if (now_label > building_points.size())
				building_points.resize(now_label);
			building_points[now_label - 1].push_back(cv::Point(now_rect.x + x, now_rect.y + y));
		}
	}

	for (int i = 0; i < building_points.size(); i++)
	{
		if (building_points[i].size() != 0)
			output.push_back(cv::boundingRect(building_points[i]));
	}
	return output;
}