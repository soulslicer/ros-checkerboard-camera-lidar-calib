#ifndef EXTRINSIC_CALIBRATION_H
#define EXTIRNSIC_CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

using namespace std;
# define M_PI 3.14159265358979323846  /* pi */

class ExtrinsicCalibration
{
public:
    enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

    ExtrinsicCalibration(){}
    ExtrinsicCalibration(const std::vector<int>& boardSize, float squareSize);
    std::vector<Eigen::MatrixXd> calibrateExtrinsics(const cv::Mat& img, cv::Mat& drawImg, cv::Mat& extMat, bool draw);
    void checkCalibration(cv::Mat& img, cv::Mat extMat);
    void setBoardSize(cv::Size boardSize);
    void setSquareSize(float squareSize);
    void setCameraMatrix(cv::Mat cameraMatrix);
    void setDistCoeffs(cv::Mat distCoeffs);

private:

    cv::Mat cameraMatrix_, distCoeffs_;
    cv::Size boardSize_, imageSize_;
    float squareSize_;
    int pattern_;
};

#endif // EXTRINSIC_CALIBRATION_H
