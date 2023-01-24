#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

template<typename T>
cv::Mat calc_dot(cv::Mat& normal_gt, cv::Mat& normal_est) {
	cv::Mat dot_map(normal_gt.size(), CV_64FC1);
	for (int y = 0; y < normal_gt.rows; ++y) {
		for (int x = 0; x < normal_gt.cols; ++x) {
			T v1 = normal_gt.at<T>(y, x);
			T v2 = normal_est.at<T>(y, x);
			double v_dot = static_cast<double>(v1.dot(v2));
			dot_map.at<double>(y, x) = v_dot;
		}
	}
	return dot_map;
}

inline cv::Mat calc_dotmap(cv::Mat& normal_gt, cv::Mat& normal_est) {
	CV_Assert(normal_gt.channels() == 3);
	auto type = normal_gt.type();
	if (type == CV_32FC3) return calc_dot<cv::Vec3f>(normal_gt, normal_est);
	if (type == CV_64FC3) return calc_dot<cv::Vec3d>(normal_gt, normal_est);
}

template<typename T>
void invert_normal(cv::Mat& dot_map, cv::Mat& normal_est) {
	for (int y = 0; y < normal_est.rows; ++y) {
		for (int x = 0; x < normal_est.cols; ++x) {
			T v2 = normal_est.at<T>(y, x);
			if (dot_map.at<double>(y, x) < DBL_EPSILON) {
				normal_est.at<T>(y, x) = -v2;
			}
		}
	}
}

inline void correct_normal_direction(cv::Mat& pc, cv::Mat& normal_est) {
	auto type = normal_est.type();
	cv::Mat pc_norm = -pc.clone();
	cv::Mat dot_map = calc_dotmap(pc_norm, normal_est);
	// Correction if only the direction of the normal is opposite to the vector to the origin.
	if (type == CV_32FC3) invert_normal<cv::Vec3f>(dot_map,normal_est);
	if (type == CV_64FC3) invert_normal<cv::Vec3d>(dot_map,normal_est);
}

template<typename T>
cv::Mat get_3D_EulerRmat(T alpha, T beta, T gamma)
{
	cv::Mat R_x =
		(cv::Mat_<T>(3, 3) <<
			1, 0, 0,
			0, std::cos(alpha), -std::sin(alpha),
			0, std::sin(alpha), std::cos(alpha));
	cv::Mat R_y =
		(cv::Mat_<T>(3, 3) <<
			std::cos(beta), 0, std::sin(beta),
			0, 1, 0,
			-std::sin(beta), 0, std::cos(beta));
	cv::Mat R_z =
		(cv::Mat_<T>(3, 3) <<
			std::cos(gamma), -std::sin(gamma), 0,
			std::sin(gamma), std::cos(gamma), 0,
			0, 0, 1);
	return R_z * R_y * R_x;
}

inline void rotate(const cv::Mat& R, cv::Mat& in) {
	for (int y = 0; y < in.rows; ++y) {
		for (int x = 0; x < in.cols; ++x) {
			cv::Mat1d v1 = (cv::Mat1d)in.at<cv::Vec3d>(y, x);
			cv::Mat1d v1_rot;
			v1_rot = R * v1;
			in.at<cv::Vec3d>(y, x) = v1_rot;
		}
	}
}

inline void normal_edge(const cv::Mat& normal, cv::Mat& normal_edge) {
	cv::Mat n_padd;//normal_padding
	cv::copyMakeBorder(normal, n_padd, 1, 1, 1, 1, CV_HAL_BORDER_REFLECT);
	normal_edge=cv::Mat::zeros(normal.size(), CV_64FC1);
	for (int y = 1; y < n_padd.rows - 1; ++y) {
		for (int x = 1; x < n_padd.cols - 1; ++x) {
			cv::Vec3d ni = n_padd.at<cv::Vec3d>(y, x);
			int valid = 0;
			double cos_similality_sum = 0.;
			for (int sx = -1; sx <= 1; ++sx) {
				for (int sy = -1; sy <= 1; ++sy) {
					int u = x + sx;
					int v = y + sy;
					cv::Vec3d nj = n_padd.at<cv::Vec3d>(v, u);
					double cos_similality = ni.dot(nj);
					if (!std::isnan(cos_similality)) {
						valid++;
						cos_similality_sum += cos_similality;
					}
				}
			}
			double mean_dot = cos_similality_sum / (double)valid;
			normal_edge.at<double>(y - 1, x - 1) = mean_dot;
		}
	}
}