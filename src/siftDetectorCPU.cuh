#include "keyPoint.h"
#include "definitions.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <fstream>
#include <cstddef>
#include <math.h>

class SiftDetectorCU {
public:
	/**
	* Empty constructor.
	*/
	SiftDetectorCU() {};
	/**
	* Default destructor.
	*
	* Clears the keypoints.
	*/
	~SiftDetectorCU();
	/**
	* Detects interest points.
	*
	* Takes the input grayscale image. Constructs the Gaussian pyramid, then constructs the
	* DoG pyramid. Finally, locates the extrema of the DoG in scale-space, adjusts them,
	* filters, and reports the resulting keypoints.
	*
	* @param image - input grayscale image.
	*/
	void detectInterestPoints(const cv::Mat& image);
	/**
	* Plots the keypoints.
	*
	* Once the keypoints are detected, draws them on the input gray-scale image. The plotting
	* is heavily based on that from OpenCV, so that the results are visually comparable.
	*
	* @param image - input grayscale image.
	* @param show - whether to show the result after drawing the keypoints.
	* @param writeToFile - whether to write the result to the file.
	* @param fPath - path to the file where the result should be stored.
	*/
	void drawKeypoints(const cv::Mat& image, bool show, bool writeToFile, std::string fPath);
	/**
	* Gets the number of located keypoints.
	*
	* Once the keypoitns are computed, can be used to get the number of the detected keypoints.
	*
	* @return number of keypoints.
	*/
	int getNumKeypoints() { return _keypoints.size(); }
private:
	int _nOctaves; ///< Number of the octaves.
	std::vector<Keypoint> _keypoints; ///< Contains the detected keypoints.
	/**
	* Calculates all the sigmas.
	*
	* Fills the _sigmas array with all the sigmas for a single octave. All the
	* octaves use the same sequence of sigmas. Every sigma is computed relatively
	* to the previous sigma, so that the Gaussian blur is performed incrementally.
	* The first sigma is computed relatively to the assumed input blur of the image.
	* 
	* @param sigmas - vector containing all sigmas.
	*/
	void computeSigmas(std::vector<float>& sigmas);
	/**
	* Fills kernels for all filters.
	*
	* Takes all the sigmas and fills the corresponding Gaussian filter's kernels. 
	* The filter size is automatically determined from sigma of each layer. The
	* filter sizes are also recorded.
	*
	* @param kernels - vector containing all kernels of the Gaussian filter.
	* @param kRadius - vector containing all kernel sizes.
	* @param sigmas - vector containing all sigmas.
	*/
	void computeKernels(std::vector<float>& kernels, 
		std::vector<int>& kRadius, const std::vector<float>& sigmas);
	/**
	* Prepares the initial image.
	*
	* Computes the initial image by the optional upsampling and the initial Gaussian blur.
	*
	* @param devImage - input image on GPU.
	* @param devKernels - Gaussian filters on GPU.
	* @param width - width of the image.
	* @param height - height of the image.
	* @param kRad - radius of the current kernel.
	*/
	void prepareInitialImage(float* devImage, float* devKernels, int width, int height, int kRad);
	/**
	* Builds DoG pyramid.
	*
	* Computes the whole DoG pyramid by subsequently building one octave. Records the keypoints
	* after constructing each octave.
	*
	* @param devImage - input image on GPU.
	* @param devConvolved - intermediate image for convolution on GPU.
	* @param devDoG - DoG octave on GPU.
	* @param devKernels - Gaussian filters on GPU.
	* @param devKeypoints - keypoints on GPU.
	* @param devCounter - keypoints counter on GPU.
	* @param width - width of the image.
	* @param height - height of the image.
	* @param kRadius - vector containing all kernel sizes.
	*/
	void buildDoGPyramid(float* devImage, float* devConvolved, float* devDoG, float* devKernels, 
		Keypoint* devKeypoints, int* devCounter, int width, int height, std::vector<int> kRadius);
	/**
	* Builds one DoG octave.
	*
	* Computes one DoG octave sequentially for each layer. Each layer is then processed on
	* GPU in the parallel manner. 
	*
	* @param devImage - image at the start of the current octave on GPU.
	* @param devConvolved - horizontally blurred octave on GPU.
	* @param devDoG - DoG octave on GPU.
	* @param devKernels - Gaussian filters on GPU.
	* @param devKeypoints - keypoints on GPU.
	* @param devCounter = keypoints counter on GPU.
	* @param width - width of the image in the current octave.
	* @param height - height of the image in the current octave.
	* @param octave - current octave.
	* @param kRadius - vector containing all kernel sizes.
	*/
	void buildDoGOctave(float* devImage, float* devConvolved, float* devDoG, float* devKernels,
		Keypoint* devKeypoints, int* devCounter, int width, int height, int octave, std::vector<int> kRadius);
	/**
	* Locates the keypoint.
	*
	* 
	*
	* @param devDoG - DoG octave on GPU.
	* @param devKeypoints - keypoints on GPU.
	* @param devCounter - keypoints counter on GPU.
	* @param width - width of the image in the current octave.
	* @param height - height of the image in the current octave.
	* @param octave - current octave.
	*/
	void findKeypoints(
		float* devDoG, Keypoint* devKeypoints, int* devCounter, int width, int height, int octave);
};