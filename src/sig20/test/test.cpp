#include <cpr/cpr.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv){
	cv::Mat img(200, 200, CV_8UC3, cv::Scalar(0,0,0));
	for (int y = 100; y < 150; y++)
		img.at<cv::Vec3b>(y, 100) = cv::Vec3b(0, 0, 255);

	std::vector<uchar> data(img.ptr(), img.ptr() + img.size().width * img.size().height * img.channels());
	std::string s(data.begin(), data.end());
	
	//auto r = cpr::Get(cpr::Url{ "http://127.0.0.1:5000/index" },
	//	cpr::Parameters{ {"img",s} });
	auto r = cpr::Post(cpr::Url{ "http://127.0.0.1:5000/index" },
		cpr::Body{ s },
		cpr::Header{ {"Content-Type", "text/plain"} });
	std::cout << r.text << std::endl;

	return 0;
}
