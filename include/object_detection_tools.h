#pragma once

#include <string>
#include <vector>
#include <tinyxml2.h>
#include <CGAL/Bbox_2.h>
#include <opencv2/opencv.hpp>



namespace tx2 = tinyxml2;

void addTextNode(tx2::XMLDocument& vDoc, tx2::XMLElement* rootElement, std::string vName, int vVal);
void writeItem_1216(const std::string& vPath, const std::vector<CGAL::Bbox_2> building_bboxes, int vWidth);
std::vector<cv::Rect> getBoundingBoxes(const std::vector<cv::Point>& pixel_points, const std::vector<std::vector<bool>>& is_same_color);
