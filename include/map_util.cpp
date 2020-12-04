/*
 * Convert coordinates in airsim, unreal, mesh, image
 */
#include "map_util.h"


MapConverter::MapConverter() {

}

void MapConverter::initDroneStart(const Eigen::Vector3f& vPos) {
	mDroneStart = vPos;
	initDroneDone = true;
}

Eigen::Vector3f MapConverter::convertUnrealToAirsim(const Eigen::Vector3f& vWorldPos) const {
	if (!initDroneDone)
		throw "Init is not done";
	Eigen::Vector3f result;
	result[0] = (vWorldPos[0] - mDroneStart[0]) / 100;
	result[1] = (vWorldPos[1] - mDroneStart[1]) / 100;
	result[2] = (-vWorldPos[2] + mDroneStart[2]) / 100;
	return result;
}

Eigen::Vector3f MapConverter::convertUnrealToMesh(const Eigen::Vector3f& vWorldPos) const {
	if (!initDroneDone)
		throw "Init is not done";
	Eigen::Vector3f result;
	result[0] = (vWorldPos[0] / 100);
	result[1] = -(vWorldPos[1] / 100);
	result[2] = vWorldPos[2] / 100;
	return result;
}

Eigen::Vector3f MapConverter::convertMeshToUnreal(const Eigen::Vector3f& vMeshPos) const {
	if (!initDroneDone)
		throw "Init is not done";
	Eigen::Vector3f result;
	result[0] = (vMeshPos[0] * 100);
	result[1] = -(vMeshPos[1] * 100);
	result[2] = vMeshPos[2] * 100;
	return result;
}

Eigen::Matrix3f MapConverter::convert_yaw_pitch_to_matrix_mesh(const float yaw,const float pitch)
{
	Eigen::Matrix3f result = (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(
		pitch, Eigen::Vector3f::UnitX())).toRotationMatrix();
	return result;
}

Eigen::Isometry3f MapConverter::get_camera_matrix(const float yaw, const float pitch,const Eigen::Vector3f& v_pos) {

	Eigen::Isometry3f result = Eigen::Isometry3f::Identity();

	result.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
	result.rotate(Eigen::AngleAxisf(-pitch, Eigen::Vector3f::UnitY()));

	result.rotate(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitX()));
	result.rotate(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitY()));

	result.pretranslate(v_pos);

	return result.inverse();
}

Eigen::Vector3f MapConverter::convert_yaw_pitch_to_direction_vector(const float yaw,const float pitch)
{
	Eigen::Vector3f direction = Eigen::Vector3f(1, 0, 0);
	direction = (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(-pitch, Eigen::Vector3f::UnitY())).toRotationMatrix() * direction;
	direction.normalize();
	return direction;
}

Pos_Pack MapConverter::get_pos_pack_from_unreal(const Eigen::Vector3f& v_pos_unreal,float yaw,float pitch)
{
	Pos_Pack pos_pack;
	pos_pack.yaw = yaw;
	pos_pack.pitch = pitch;
	pos_pack.pos_mesh = convertUnrealToMesh(v_pos_unreal);
	pos_pack.pos_airsim = convertUnrealToAirsim(v_pos_unreal);
	pos_pack.camera_matrix = get_camera_matrix(yaw, pitch, pos_pack.pos_mesh);
	
	return pos_pack;
}
