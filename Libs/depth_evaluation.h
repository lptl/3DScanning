#pragma once
#include "normals.h"
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>


int count_non_NaN(const cv::Mat& in) {	
	CV_Assert(in.type() == CV_64FC1);
	int count = 0;
	for (int y = 0; y < in.rows; ++y) {
		for (int x = 0; x < in.cols; ++x) {
			if (std::isnan(in.at<double>(y, x))) count++;
		}
	}
	return in.total() - count;
}

void patch_nan_double(cv::Mat& in, double val = 0.0) {
	CV_Assert(in.type() == CV_64FC1);
	for (int y = 0; y < in.rows; ++y) {
		for (int x = 0; x < in.cols; ++x) {
			if (std::isnan(in.at<double>(y, x))) {
				in.at<double>(y, x) = val;
			}
		}
	}

}
// Calculate the median with histSize resolution up to the maximum value.
double medianMat(const cv::Mat& input, const int& histSize = 1000) {
	cv::Mat in;
#if 0
	// test data
	in = cv::Mat(1, 11, CV_32FC1);
	float val = 10.f;
	for (int y = 0; y < in.rows; ++y) {
		for (int x = 0; x < in.cols; ++x) {
			in.at<float>(y, x) = val;
			val *= 1.5;
		}
	}
#else
	input.convertTo(in, CV_32FC1);
#endif
	cv::Mat in_orig = in.clone();
	double minval, maxval;
	cv::minMaxLoc(in, &minval, &maxval, 0, 0);
	in /= maxval;

	double m = (in.rows * in.cols) / 2; 
	int bin = 0;

	float range[] = { 0.f, 1.f };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;
	cv::calcHist(&in, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	double med = -1.0;
	for (int i = 0; i < histSize && med < 0.0; ++i) {
		bin += cvRound(hist.at<float>(i));
		if (bin > m && med < 0.0)
			med = i;
	}
	auto offset = 1.f / ((float)histSize * 2.f);

	med = med / (float)histSize + offset;
	med *= maxval;

	// Pulling results from the side recently
	cv::Mat error = cv::abs(in_orig - med);
	cv::Point minloc;
	cv::minMaxLoc(error, 0, 0, &minloc, 0);

	float result = in_orig.at<float>(minloc.y, minloc.x);
	return static_cast<double>(result);
}

double calcNonNaNmean(const cv::Mat& in) {
	int N = count_non_NaN(in);
	cv::Mat in_patch=in.clone();
	patch_nan_double(in_patch, 0.);
	auto in_sums = cv::sum(in_patch);
	return in_sums[0] / (double)N;
}

double calcRootMeanValid(cv::Mat& input) {
	if (input.channels() > 1) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	// invalid is a zero assumption
	return cv::sqrt(calcNonNaNmean(input));
}


template<typename T>
cv::Mat calcDotmap(cv::Mat& mat_a, cv::Mat& mat_b) {
	cv::Mat dotmap(mat_a.size(), CV_MAKETYPE(mat_a.depth(), 1));
	const int H = mat_a.rows;
	const int W = mat_a.cols;
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			cv::Vec<T, 3> a = mat_a.at<cv::Vec<T, 3>>(y, x);
			cv::Vec<T, 3> b = mat_b.at<cv::Vec<T, 3>>(y, x);
			dotmap.at<T>(y, x) = a.dot(b);
		}
	}
	return dotmap;
}

cv::Mat calcDotmap(cv::Mat& mat_a, cv::Mat& mat_b) {
	assert(mat_a.size() == mat_b.size());

	cv::Mat dotmap;
	if (mat_a.type() == CV_32FC3) {
		dotmap = calcDotmap<float>(mat_a, mat_b);
	}
	else if(mat_a.type() == CV_64FC3) {
		dotmap = calcDotmap<double>(mat_a, mat_b);
	}
	return dotmap;
}


void calcNormal(const cv::Mat& depth, const cv::Mat& K, cv::Mat& cloud, cv::Mat& normal)
{
	cv::rgbd::RgbdNormals rgbdn(depth.rows, depth.cols, depth.depth(), K, 5);
	cv::rgbd::depthTo3d(depth, K, cloud);
	rgbdn(cloud, normal);
	correct_normal_direction(cloud, normal);
}



// https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html
// ![get-psnr]
double calcPSNR(const cv::Mat& I1, const cv::Mat& I2, double max_i = 255, double eps= 1e-10)
{
	cv::Mat s1;
	absdiff(I1, I2, s1);       
	s1.convertTo(s1, CV_64F);  
	s1 = s1.mul(s1);           
	cv::Scalar s = cv::sum(s1);
	double sse = s.val[0]; 

	if (sse <= eps) // for small values return zero
		return 0;
	else{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((max_i* max_i) / mse);
		return psnr;
	}
}