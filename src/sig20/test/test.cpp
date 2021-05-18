#include <cpr/cpr.h>
#include <opencv2/opencv.hpp>
#include "model_tools.h"
#include "cgal_tools.h"

//int main(int argc, char** argv){
//	cv::Mat img(200, 200, CV_8UC3, cv::Scalar(0,0,0));
//	for (int y = 100; y < 150; y++)
//		img.at<cv::Vec3b>(y, 100) = cv::Vec3b(0, 0, 255);
//
//	std::vector<uchar> data(img.ptr(), img.ptr() + img.size().width * img.size().height * img.channels());
//	std::string s(data.begin(), data.end());
//	
//	//auto r = cpr::Get(cpr::Url{ "http://127.0.0.1:5000/index" },
//	//	cpr::Parameters{ {"img",s} });
//	auto r = cpr::Post(cpr::Url{ "http://127.0.0.1:5000/index" },
//		cpr::Body{ s },
//		cpr::Header{ {"Content-Type", "text/plain"} });
//	std::cout << r.text << std::endl;
//
//	return 0;
//}

int main()
{
	Polygon_2 polygon;
	std::vector<Point_2> points{
		Point_2(0,0),
		Point_2(0,1),
		Point_2(1,1),
		Point_2(1,0)
	};
	polygon = Polygon_2(points.begin(), points.end());

	std::cout << (polygon.bounded_side(Point_2(0.5, 0.5)) == CGAL::Bounded_side::ON_BOUNDED_SIDE) << std::endl;
	
}
