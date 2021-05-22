#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "stdio.h"
#include "device_launch_parameters.h"
#include "keypoint.h"
#include "definitions.h"

#ifndef SIFT_GPU
#define SIFT_GPU
/**
* Horizontal Gaussian blur.
*
* GPU kernel that performs the 1D (horizontal) Gaussian blur on the input image
* by convolving the input image with the pre-computed Gaussian kernel. Every thread
* computes the convolution result for one pixel only.
*
* @param image - gray-scale input image.
* @param convolved - horizontally blurred image.
* @param convolvedDoG - current blurred octave.
* @param kernels - pre-computed Gaussian kernels.
* @param width - width of the image in the current octave.
* @param height - height of the image in the current octave.
* @param layer - current layer in the octave.
* @param kRadius - radius of the kernel in the current layer.
* @param returnDoG - true if this is part of DoG computation, false if it is part of initial image blurring.
*/
__global__ void gaussianHorizontal(float* image, float* convolved, float* convolvedDoG,
	float* kernels, int width, int height, int layer, int kRadius, bool returnDoG);

/**
* Vertical Gaussian blur and DoG computation.
*
* Takes the horizontally-blurred images in all levels of an octave and performs the vertical blur. Since the
* 2D Gaussian blur is separable, computing the vertical blur on the horizontally-blurred image corresponds
* to the image convolved with the 2D Gaussian filter. Difference-of-Gaussian between the neighboring levels
* of an octave is computed.
*
* @param image - gray-scale image that will be used for the downsampling to the next octave.
* @param concolved - horizontally blurred image.
* @param convolvedDoG - vertically blurred octave if !returnDoG, DoG octave otherwise.
* @param kernels - pre-computed Gaussian kernels.
* @param width - width of the image in the current octave.
* @param height - height of the image in the current octave.
* @param layer - current layer in the octave.
* @param kRadius - radius of the kernel in the current layer.
* @param returnDoG - true if this is part of DoG computation, false if it is part of initial image blurring.
*/
__global__ void gaussianVerticalDoG(float* image, float* convolved, float* DoG,
	float* kernels, int width, int height, int layer, int kRadius, bool returnDoG);

/**
* Downsamples the image by factor of 2.
*
* Reduces the size of the image by factor of 2 to move to the next octave in the scale-space.
* Each pixel in the new image is derived from a 2x2 tile in the old image (a simple average
* is taken as the new pixel value).
*
* @param image - gray-scale image to be downsampled.
* @param width - width of the image in the previous octave.
* @param height - height of the image in the previous octave.
*/
__global__ void downsampleImage(float* image, int width, int height);

/**
* Upsamples the image by factor of 2.
*
* Increases the size of the image by factor of 2 to prepare the initial image with upsampling.
* Each pixel in the new image is derived from a 2x2 tile in the old image. The weights are
* calculated based on the linear interpolation.
*
* @param image - gray-scale image to be upsampled.
* @param upsampledImage - gray-scale upsampled image.
* @param width - original width of the image.
* @param height - original height of the image.
* @param rRatio - ratio for calculating weights between different rows.
* @param cRatio - ratio for calculating weights between different columns.
*/
__global__ void upsampleImage(float* image, float* upsampledImage,
	int width, int height, const float rRatio, const float cRatio);

/**
* Locates the keypoints in DoG.
* 
* Takes the 27-neighborhood of a pixel and checks if the current pixel is a local minima or
* maxima. If it is the case, the keypoint at this pixel is created. The keypoint is then
* adjusted by the quadratic interpolation to better correspond to the true position of the
* extrema. The keypoint is then filtered by the constrast and by edgeness. If all the filters
* are passed, the keypoint is added to the total list of keypoints (unless the list is full).
* 
* @param convolvedDoG - DoG pyramid for the current octave.
* @param keypoints - array where the keypoints are stored.
* @param counter - index in the array of keypoints.
* @param width - width of the image in the current octave.
* @param height - height of the image in the current octave.
* @oaram octave - current octave.
*/
__global__ void locateExtrema(float* convolvedDoG, Keypoint* keypoints,
	int* counter, int width, int height, int octave);

/**
* Gets the 27-neighborhood of a pixel from cache
* 
* Fills the cache with the 26 neighbors of a pixel, including the neighbors on the current,
* upper and the bottom image level. 
* 
* @param neighbors - array where the neighborhood is stored.
* @param cache - cache where the sub-image for current thread block is stored.
* @param tIdy - row of the active thread in the thread block.
* @param tIdx - column of the active thread in the thread block.
* @param layer - layer of the current octave in DoG.
*/
__device__ void getPointNeighborhood(float* neighbors, float* cache, int tIdy, int tIdx, int layer);

/**
* Get the 27-neighborhood of a pixel
*
* Fills the cache with the 26 neighbors of a pixel, including the neighbors on the current,
* upper and the bottom image level.
*
* @param neighbors - array where the neighborhood is stored.
* @param convolvedDoG - DoG pyramid for the current octave.
* @param row - image's row filled by the current thread.
* @param col - image's column filled by the current thread.
* @param layer - layer of the current octave in DoG.
* @param width - width of the image in the current octave.
*/
__device__ void getPointNeighborhoodFromImage(
	float* neighbors, float* convolvedDoG, int row, int col, int layer, int width);


/**
* Checks for maximum.
* 
* Goes through the 27-neighborhood of a pixel to determine if it is a local maximum.
* 
* @param neighbors - array where the neighborhood is stored.
* @return true if the center of the 27-neighborhood is a local maximum, false otherwise.
*/
__device__ bool isMaximum(float* neighbors);

/**
* Checks for minimum.
*
* Goes through the 27-neighborhood of a pixel to determine if it is a local minimum.
*
* @param neighbors - array where the neighborhood is stored.
* @return true if the center of the 27-neighborhood is a local minimum, false otherwise.
*/
__device__ bool isMinimum(float* neighbors);

/**
* Gets values at updated extremum.
* 
* Performs quadratic interpolation to adjust the current position of the extremum. Fills
* the alpha array, containing the values for extremum udpates. Returns the value of the
* function at the udpated extremum point.
* 
* @param neighbors - array where the neighborhood is stored.
* @param alpha - array where the extremum updates are stored.
* @return DoG function value at the updated extremum.
*/
__device__ float getValueAtExtremum(float* neighbors, float* alpha);

/**
* Adjust the keypoint.
* 
* A fit to nearby data is performed to adjust the position of the keypoint in the scale-space. 
* It is done by fitting a 2D quadratic function and determining the interpolated position of 
* the maximum. The keypoints with low contrast are rejected.
* 
* @param kp - current keypoint.
* @param neighbors - array where the neighborhood is stored.
* @param convolvedDoG - DoG pyramid for the current octave.
* @param width - width of the image in the current octave.
* @param height - height of the image in the current octave.
* @return true if a keypoint should be kept, false otherwise.
*/
__device__ bool adjustKeypoint(Keypoint* kp, 
	float* neighbors, float* convolvedDoG, int width, int height);

/**
* Filter the keypoint.
* 
* 
* @param kp - current keypoint.
* @param neighbors - array where the neighborhood is stored.
* @param convolvedDoG - DoG octave on GPU.
* @param width - width of the image in the current octave.
* @return true if a keypoint should be kept, false otherwise.
*/
__device__ bool keepKeypoint(Keypoint* kp, float* neighbors, float* convolvedDoG, int width);

#endif