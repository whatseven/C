#pragma once

#include "airsim_control.h"

std::map<std::string, cv::Mat> Airsim_tools::get_images()
{
    using namespace msr::airlib;
    typedef ImageCaptureBase::ImageRequest ImageRequest;
    typedef ImageCaptureBase::ImageResponse ImageResponse;
    typedef ImageCaptureBase::ImageType ImageType;

    vector<ImageRequest> request = {
        ImageRequest("0", ImageType::Scene, false, false),
        ImageRequest("0", ImageType::DepthPlanner, true,false),
        ImageRequest("0", ImageType::Segmentation, false,false),
        //ImageRequest("0", ImageType::DepthPerspective, true,false),
    };
    const vector<ImageResponse>& response = m_agent->simGetImages(request);

    std::map<std::string, cv::Mat> images;
    if (response.size() > 0) {
        for (const ImageResponse& image_info : response) {
            if (image_info.image_type == ImageType::Scene) {
                cv::Mat rgb = cv::Mat(image_info.height, image_info.width, CV_8UC3,
                    (unsigned*)image_info.image_data_uint8.data()).clone();
                if (rgb.dims == 0)
                    return std::map<std::string, cv::Mat>();
                //cv::cvtColor(rgb, rgb, CV_BGR2RGB);
                images.insert(std::pair<std::string, cv::Mat>("rgb", rgb));
            }
            else if (image_info.image_type == ImageType::DepthPlanner) {
                cv::Mat depth = cv::Mat(image_info.height, image_info.width, CV_32FC1,
                    (float*)image_info.image_data_float.data()).clone();
                if (depth.dims == 0)
                    return std::map<std::string, cv::Mat>();
                images.insert(std::pair<std::string, cv::Mat>("depth_planar", depth));

            }
            else if (image_info.image_type == ImageType::Segmentation) {
                cv::Mat seg = cv::Mat(image_info.height, image_info.width, CV_8UC3,
                    (unsigned*)image_info.image_data_uint8.data()).clone();
                if (seg.dims == 0)
                    return std::map<std::string, cv::Mat>();
                //cv::cvtColor(seg, seg, CV_BGR2RGB);
                images.insert(std::pair<std::string, cv::Mat>("segmentation", seg));
            }
            else if (image_info.image_type == ImageType::DepthPerspective) {
                cv::Mat depth = cv::Mat(image_info.height, image_info.width, CV_32FC1,
                    (float*)image_info.image_data_float.data()).clone();
                if (depth.dims == 0)
                    return std::map<std::string, cv::Mat>();
                images.insert(std::pair<std::string, cv::Mat>("depth_perspective", depth));

            }
        }
    }
    return images;
}

void Airsim_tools::adjust_pose(const Pos_Pack& v_pos_pack){
    Eigen::Quaternionf directionQuaternion;
    directionQuaternion= Eigen::AngleAxisf(-v_pos_pack.yaw, Eigen::Vector3f::UnitZ())* Eigen::AngleAxisf(
        -v_pos_pack.pitch, Eigen::Vector3f::UnitY());

    m_agent->simSetVehiclePose(msr::airlib::Pose(v_pos_pack.pos_airsim,
        directionQuaternion
    ),true);
    
    return;
}

void Airsim_tools::reset_color(const std::string& v_key_words) {
    int num = 1;
    for(const auto& item:m_agent->simListSceneObjects())
    {
    	if(v_key_words.size()>0&& item.find(v_key_words)!=item.npos)
            m_agent->simSetSegmentationObjectID(item, num++);
        else
			m_agent->simSetSegmentationObjectID(item, 0);
    }
    return;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> demo_move_to_next(msr::airlib::MultirotorRpcLibClient& v_agent,
	const Eigen::Vector3f& v_next_pos_airsim, const float v_speed, bool is_forward)
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> poses;
    Pose pose = v_agent.simGetVehiclePose();
    Eigen::Vector3f pos_cur = pose.position;
	
    while (true)
    {
        pose = v_agent.simGetVehiclePose();
        pos_cur = pose.position;
    	
        Eigen::Vector3f direction = v_next_pos_airsim - pos_cur;
        direction.normalize();
        direction = direction * v_speed;
    	if(is_forward)
			v_agent.moveByVelocityAsync(direction[0], direction[1], direction[2], 20,
				DrivetrainType::ForwardOnly, YawMode(false, 0));
        else
            v_agent.moveByVelocityAsync(direction[0], direction[1], direction[2], 20);

        Eigen::Vector3f pos_cur = pose.position;
    	if((pos_cur-v_next_pos_airsim).norm()<1)
            break;
        poses.push_back(std::make_pair(pos_cur, Eigen::Vector3f(0, 0, -1)));
        override_sleep(0.05);
    }
    return poses;
}
