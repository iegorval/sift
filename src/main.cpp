#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "siftDetectorCPU.cuh"
#include "siftDetector.h"

int main(int argc, char* argv[]) {
    std::string inputPath(argv[1]);
    std::string savePath(argv[2]);
    std::string version(argv[3]);
    std::vector<cv::String> fn;
    cv::glob(inputPath + "*.jpg", fn, false);
    size_t count = fn.size(); 
    int duration, nKeypoints;
#ifdef DEBUG_LOG_TIME
    std::ofstream myfile("log.txt", std::ofstream::trunc);
#endif
    for (size_t i = 0; i < count; i++) {
        cv::Mat image = cv::imread(fn[i]);
        cv::Mat imageGray;
        cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
        if (image.empty()) {
            std::cout << "Unable to open the image file." << std::endl;
            std::cin.get();
            return -1;
        }
        try {
            auto start = std::chrono::high_resolution_clock::now();
            std::string fName = savePath + std::to_string(i) + ".bmp";
            if (version == "cv") {
                std::cout << "Using OpenCV CPU implementation..." << std::endl;
                auto siftCV = cv::SIFT::create();
                std::vector<cv::KeyPoint> keypoints;
                siftCV->detect(imageGray, keypoints);
                std::cout << "KEYPOINTS FOUND: " << keypoints.size() << std::endl;
                auto stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                std::cout << "Running time [ms]: " << duration << std::endl;
                cv::Mat kpImage;
                nKeypoints = keypoints.size();
                cv::drawKeypoints(imageGray, keypoints, kpImage, CV_RGB(255, 0, 0));
                cv::imwrite(fName, kpImage);
                //cv::imshow("Detected Keypoints", kpImage);
                //cv::waitKey(0);
                //cv::destroyWindow("Detected Keypoints");
            } else if (version == "gpu") {
                std::cout << "Using GPU implementation..." << std::endl;
                auto* siftCU = new SiftDetectorCU();
                siftCU->detectInterestPoints(imageGray);
                auto stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                std::cout << "Running time [ms]: " << duration << std::endl;
                nKeypoints = siftCU->getNumKeypoints();
                siftCU->drawKeypoints(imageGray, false, true, fName);
                delete siftCU;
            } else {
                std::cout << "Using CPU implementation..." << std::endl;
                auto* sift = new SiftDetector();
                sift->detectInterestPoints(imageGray);
                auto stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                std::cout << "Running time [ms]: " << duration << std::endl;
                nKeypoints = sift->getNumKeypoints();
                sift->drawKeypoints(imageGray, false, true, fName);
                delete sift;
            }
        }
        catch (cv::Exception& e) {
            std::cerr << e.msg << std::endl;
        }
#ifdef DEBUG_LOG_TIME
        myfile << duration << "," << nKeypoints << std::endl;
#endif
    }
#ifdef DEBUG_LOG_TIME
    myfile.close();
#endif
    return 0;
}