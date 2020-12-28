#ifndef PARAMS_LIB_PYTHON_HPP
#define PARAMS_LIB_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <converters.hpp>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <chrono>
#include <sensor_msgs/image_encodings.h>
#include <ExtrinsicCalibration.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>

namespace py = pybind11;

struct RANSACData{
    std::vector<pcl::PointCloud<pcl::PointXYZ>> topSurfaceProjections;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> topSurfacePoints;
    std::vector<std::vector<float>> topSurfaceCoeffs;
    RANSACData() {}
};

RANSACData extractModel(const pcl::PointCloud<pcl::PointXYZ>& pcl, int model, float error, int minPoints=20, int maxCount=5){
    pcl::PointCloud<pcl::PointXYZ> topSurface;
    pcl::copyPointCloud(pcl, topSurface);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (model);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (error);

    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> topSurfacePtr = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(topSurface);
    RANSACData ransacData;
    int nr_points = (int) topSurface.size ();
    int counter = 0;
    while (topSurfacePtr->size () > 0.01 * nr_points)
    {
        counter++;
        if(counter > maxCount) break;

        // Segment the largest model component from the remaining cloud
        seg.setInputCloud (topSurfacePtr);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
          std::cout << "Could not estimate a model for the given dataset." << std::endl;
          break;
        }

        // Extract the model inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (topSurfacePtr);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the model
        pcl::PointCloud<pcl::PointXYZ> modelPoints;
        extract.filter (modelPoints);

        // Project
        pcl::PointCloud<pcl::PointXYZ> projPoints;
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType (model);
        proj.setIndices (inliers);
        proj.setInputCloud (topSurfacePtr);
        proj.setModelCoefficients (coefficients);
        proj.filter (projPoints);

        // Add
        if(modelPoints.size() > minPoints){
            ransacData.topSurfaceProjections.push_back(projPoints);
            ransacData.topSurfacePoints.push_back(modelPoints);
            ransacData.topSurfaceCoeffs.push_back(coefficients->values);
        }else{
            break;
        }

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*topSurfacePtr);
    }

    return ransacData;
}

Eigen::MatrixXf pclxyz2eigen(const pcl::PointCloud<pcl::PointXYZ>& pcl){
    Eigen::MatrixXf xyzCloud(pcl.size(), 4);
    for(int i=0; i<pcl.size(); i++){
        xyzCloud(i, 0) = pcl[i].x;
        xyzCloud(i, 1) = pcl[i].y;
        xyzCloud(i, 2) = pcl[i].z;
        xyzCloud(i, 3) = 1.;
    }
    return xyzCloud;
}

pcl::PointCloud<pcl::PointXYZ> eigen2pclxyz(const Eigen::MatrixXf& eigen){
    pcl::PointCloud<pcl::PointXYZ> xyzCloud;
    for(int i=0; i<eigen.rows(); i++){
        pcl::PointXYZ p(eigen(i,0), eigen(i,1), eigen(i,2));
        xyzCloud.push_back(p);
    }
    return xyzCloud;
}

Eigen::MatrixXf planeFit(const Eigen::MatrixXf& cloud, float error){
    pcl::PointCloud<pcl::PointXYZ> pclCloud = eigen2pclxyz(cloud);
    auto output = extractModel(pclCloud, pcl::SACMODEL_PLANE, error).topSurfaceProjections;
    if(output.size()){
        auto bestFit = output[0];
        return pclxyz2eigen(bestFit);
    }else{
        Eigen::MatrixXf empty;
        return empty;
    }
}

Eigen::MatrixXf concaveFit(const Eigen::MatrixXf& cloud){
    pcl::PointCloud<pcl::PointXYZ> pclCloud = eigen2pclxyz(cloud);
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pclCloudPtr = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(pclCloud);
    pcl::ConcaveHull<pcl::PointXYZ> concaveHull;
    pcl::PointCloud<pcl::PointXYZ> hullCloud;
    concaveHull.setInputCloud(pclCloudPtr);
    concaveHull.setAlpha (0.1);
    concaveHull.reconstruct(hullCloud);
    return pclxyz2eigen(hullCloud);
}

Eigen::MatrixXf convexFit(const Eigen::MatrixXf& cloud){
    pcl::PointCloud<pcl::PointXYZ> pclCloud = eigen2pclxyz(cloud);
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pclCloudPtr = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(pclCloud);
    pcl::ConvexHull<pcl::PointXYZ> convexHull;
    pcl::PointCloud<pcl::PointXYZ> hullCloud;
    convexHull.setInputCloud(pclCloudPtr);
    convexHull.reconstruct(hullCloud);
    return pclxyz2eigen(hullCloud);
}

pcl::PointXYZ lineIntersect(const std::vector<float>& P, const std::vector<float>& Q){
    Eigen::Vector3f DP(P[3], P[4], P[5]);
    Eigen::Vector3f DQ(Q[3], Q[4], Q[5]);
    Eigen::Vector3f P0(P[0], P[1], P[2]);
    Eigen::Vector3f Q0(Q[0], Q[1], Q[2]);
    Eigen::Vector3f PQ = Q0 - P0;

    float a = DP.dot(DP);
    float b = DP.dot(DQ);
    float c = DQ.dot(DQ);
    float d = DP.dot(PQ);
    float e = DQ.dot(PQ);

    float DD = a * c - b * b;
    float tt = (b * e - c * d) / DD;
    float uu = (a * e - b * d) / DD;

    Eigen::Vector3f Pi = P0 + tt * DP;
    Eigen::Vector3f Qi = Q0 + uu * DQ;
    float l2 = sqrt(pow(Pi(0) - Qi(0), 2) + pow(Pi(1) - Qi(1), 2) + pow(Pi(2) - Qi(2), 2));

    return pcl::PointXYZ(Pi(0), Pi(1), Pi(2));
}

std::vector<std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, std::vector<float>>> linesFit(const Eigen::MatrixXf& cloud, float error){
    pcl::PointCloud<pcl::PointXYZ> pclCloud = eigen2pclxyz(cloud);
    RANSACData ransacData = extractModel(pclCloud, pcl::SACMODEL_LINE, error, 4, 4);
    std::vector<std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, std::vector<float>>> projLines;
    if(ransacData.topSurfaceProjections.size() != 4) return projLines;
    for(int i=0; i<4; i++){
        std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, std::vector<float>> line;
        std::get<0>(line) = pclxyz2eigen(ransacData.topSurfaceProjections[i]);
        std::get<1>(line) = pclxyz2eigen(ransacData.topSurfacePoints[i]);
        std::get<2>(line) = ransacData.topSurfaceCoeffs[i];
        projLines.emplace_back(line);
    }
    return projLines;
}

Eigen::MatrixXf convertPointCloud2toXYZ(const sensor_msgs::PointCloud2& rosCloud){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(rosCloud,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
    pcl::fromPCLPointCloud2(pcl_pc2,temp_cloud);
    return pclxyz2eigen(temp_cloud);
}

sensor_msgs::PointCloud2 convertXYZtoPointCloud2(const Eigen::MatrixXf& cloud){
    pcl::PointCloud<pcl::PointXYZ> pclCloud = eigen2pclxyz(cloud);
    sensor_msgs::PointCloud2 rosCloud;
    pcl::toROSMsg(pclCloud, rosCloud);
    return rosCloud;
}

PYBIND11_MODULE(params_lib_python, m) {

    // LightCurtainWrapper
    py::class_<ExtrinsicCalibration, std::shared_ptr<ExtrinsicCalibration>>(m, "ExtrinsicCalibration")
            .def(py::init<std::vector<int>, float>(), py::arg("boardSize"), py::arg("squareSize"))
            .def("calibrateExtrinsics", &ExtrinsicCalibration::calibrateExtrinsics)
            .def("checkCalibration", &ExtrinsicCalibration::checkCalibration)
            .def("setCameraMatrix", &ExtrinsicCalibration::setCameraMatrix)
            .def("setDistCoeffs", &ExtrinsicCalibration::setDistCoeffs)
            .def("checkCalibration", &ExtrinsicCalibration::checkCalibration)
            ;

    m.def("convertXYZtoPointCloud2", &convertXYZtoPointCloud2, "convertXYZtoPointCloud2");
    m.def("convertPointCloud2toXYZ", &convertPointCloud2toXYZ, "convertPointCloud2toXYZ");
    m.def("planeFit", &planeFit, "planeFit");
    m.def("concaveFit", &concaveFit, "concaveFit");
    m.def("convexFit", &convexFit, "convexFit");
    m.def("linesFit", &linesFit, "linesFit");

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}

#endif