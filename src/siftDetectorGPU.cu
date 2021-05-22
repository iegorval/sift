#include "siftDetectorGPU.cuh"

__global__ void gaussianHorizontal(float* image, float* convolved, float* convolvedDoG,
	float* kernels, int width, int height, int layer, int kRadius, bool returnDoG)
{
	__shared__ float cache[TILE_SIDE + 2 * MAX_KERNEL_RADIUS];
	int tIdx = threadIdx.x;
	int col = blockIdx.x * TILE_SIDE + tIdx;
	int row = blockIdx.y;
	int numLayers = (returnDoG) ? (SIZE_OCTAVES + 3) : 1;
	int offset = row * width;
	int cacheLeftOffset = MAX_KERNEL_RADIUS - kRadius;

	if (col < width) {
		// The first layer of an octave already has the correct blur from initial blur / downsampling
		if (layer == 0 && returnDoG) {
			convolved[offset + col] = image[offset + col];
		} else {
			int layerOffset = offset * numLayers;
			float* source = (returnDoG) ? convolvedDoG + (layer - 1) * width : image;
			// Copy the part of the image row to the shared memory: each thread takes care of 1 pixel
			cache[tIdx + MAX_KERNEL_RADIUS] = source[layerOffset + col];

			// If a thread is located at the corners of the tread block, it also takes care of the 
			// neighboring pixels required for the 1D convolution.
			if (col == 0) {
				// Mirrow the pixels at the left border of the image
				for (int i = 0; i < kRadius; i++)
					cache[cacheLeftOffset + i] = source[layerOffset + kRadius - i - 1];
			}
			else if (tIdx == 0) {
				for (int i = 0; i < kRadius; i++) {
					int colIdx = (int)fmaxf(0.f, col - kRadius + i);
					cache[cacheLeftOffset + i] = source[layerOffset + colIdx];
				}
			}
			if (col == width - 1) {
				// Mirrow pixels at the right border of the image
				for (int i = 0; i < kRadius; i++)
					cache[MAX_KERNEL_RADIUS + tIdx + i + 1] = source[layerOffset + col - i - 1];
			}
			else if (tIdx == TILE_SIDE - 1) {
				for (int i = 0; i < kRadius; i++) {
					int colIdx = (int)fminf((float)width - 1.f, col + i + 1);
					cache[MAX_KERNEL_RADIUS + tIdx + i + 1] = source[layerOffset + colIdx];
				}
			}
			__syncthreads();

			float* data = cache + cacheLeftOffset + tIdx;
			// Compute 1D convolution for the current pixel, corresponding to the horizontal blur		
			float* kernel = kernels + layer * (MAX_KERNEL_RADIUS + 1);
			float convResult = kernel[0] * data[kRadius];
			for (int i = 1; i <= kRadius; i++)
				convResult += kernel[i] * (data[kRadius - i] + data[kRadius + i]);
			// Write the result of the convolution for the current pixel
			convolved[offset + col] = convResult;
		}
	}
}

__global__ void gaussianVerticalDoG(float* image, float* convolved, float* convolvedDoG,
	float* kernels, int width, int height, int layer, int kRadius, bool returnDoG)
{
	__shared__ float cache[(TILE_SIDE + 2 * MAX_KERNEL_RADIUS) * TILE_SIDE];
	int tIdx = threadIdx.x; 
	int tIdy = threadIdx.y;
	int col = blockIdx.x * TILE_SIDE + tIdx;
	int row = blockIdx.y * TILE_SIDE + tIdy;
	int numLayers = (returnDoG) ? (SIZE_OCTAVES + 3) : 1;
	int cacheTopOffset = MAX_KERNEL_RADIUS - kRadius;

	if (row < height && col < width) {
		int curIdx = row * width * numLayers + layer * width + col;
		// The first layer already has the correct blur
		if (layer == 0 && returnDoG) {
			convolvedDoG[curIdx] = image[row * width + col];
		} else {
			// Copy the part of the image row to the shared memory: each thread takes care of 1 pixel
			int rowOffsetCache = (tIdy + MAX_KERNEL_RADIUS) * TILE_SIDE;
			cache[rowOffsetCache + tIdx] = convolved[row * width + col];

			if (row == 0) {
				// Mirrow the pixels at the upper border of the image
				for (int i = 0; i < kRadius; i++) {
					int convolvedIdx = (kRadius - i - 1) * width + col;
					cache[(cacheTopOffset + i) * TILE_SIDE + tIdx] = convolved[convolvedIdx];
				}
			}
			else if (tIdy == 0) {
				for (int i = 0; i < kRadius; i++) {
					int rowIdx = (int)fmaxf(0.f, row - kRadius + i);
					int convolvedIdx = rowIdx * width + col;
					cache[(cacheTopOffset + i) * TILE_SIDE + tIdx] = convolved[convolvedIdx];
				}
			}
			if (row == height - 1) {
				// Mirrow the pixels at the bottom border of the image
				for (int i = 0; i < kRadius; i++) {
					int cacheIdx = (MAX_KERNEL_RADIUS + tIdy + i + 1) * TILE_SIDE + tIdx;
					int convolvedIdx = (row - i - 1) * width + col;
					cache[cacheIdx] = convolved[convolvedIdx];
				}
			}
			else if (tIdy == TILE_SIDE - 1) {
				for (int i = 0; i < kRadius; i++) {
					int rowIdx = (int)fminf((float)height - 1.f, row + i + 1);
					int cacheIdx = (MAX_KERNEL_RADIUS + tIdy + i + 1) * TILE_SIDE + tIdx;
					int convolvedIdx = rowIdx * width + col;
					cache[cacheIdx] = convolved[convolvedIdx];
				}
			}
			__syncthreads();

			// Compute 1D convolution for the current pixel, corresponding to the vertical blur
			int cacheOffset = (tIdy + MAX_KERNEL_RADIUS) * TILE_SIDE + tIdx;
			float* kernel = kernels + layer * (MAX_KERNEL_RADIUS + 1);
			// Compute 1D convolution for the current pixel, corresponding to the vertical blur		
			float convResult = kernel[0] * cache[cacheOffset];
			for (int i = 1; i <= kRadius; i++) {
				convResult += kernel[i] * (
					cache[cacheOffset - i * TILE_SIDE] + cache[cacheOffset + i * TILE_SIDE]);
			}
			// Write the result of the convolution for the current pixel
			convolvedDoG[curIdx] = convResult;

			if (returnDoG) {
				// Directly construct the Difference-of-Gaussian for the neighboring layers of an octave
				convolvedDoG[curIdx - width] = convolvedDoG[curIdx] - convolvedDoG[curIdx - width];
			}
			__syncthreads();

			// Save the image that will be used for downsampling 
			if (layer == SIZE_OCTAVES && returnDoG) {
				image[row * width + col] = convolvedDoG[curIdx];
			}
		}
	}
}

__global__ void downsampleImage(float* image, int width, int height) 
{
	int col = blockIdx.x * TILE_SIDE_FIXED + threadIdx.x;
	int row = blockIdx.y * TILE_SIDE_FIXED + threadIdx.y;
 	if (row < height / 2 && col < width / 2) {
		float result = image[row * 2 * width + col * 2];
		result += image[(row * 2 + 1) * width + col * 2]; 
		result += image[row * 2 * width + col * 2 + 1];
		result += image[(row * 2 + 1) * width + col * 2 + 1];
		image[row * (width / 2) + col] = 0.25f * result;
	}
}

__global__ void upsampleImage(float* image, float* upsampledImage,
	int width, int height, const float rRatio, const float cRatio)
{
	int col = blockIdx.x * TILE_SIDE_FIXED + threadIdx.x;
	int row = blockIdx.y * TILE_SIDE_FIXED + threadIdx.y;
	if (row < 2 * height && col < 2 * width) {
		int cLow = (int)floorf(cRatio * col), rLow = (int)floorf(rRatio * row);
		int cHigh = (int)ceilf(cRatio * col), rHigh = (int)ceilf(rRatio * row);
		float cWeight = (cRatio * col) - (float)cLow, rWeight = (rRatio * row) - (float)rLow;
		float result = image[rLow * width * 2 + cLow] * (1 - cWeight) * (1 - rWeight);
		result += image[rLow * width * 2 + cHigh] * cWeight * (1 - rWeight);
		result += image[rHigh * width * 2 + cLow] * (1 - cWeight) * rWeight;
		result += image[rHigh * width * 2 + cHigh] * cWeight * rWeight;
		upsampledImage[row * width * 2 + col] = result;
	}
}

__global__ void locateExtrema(
	float* convolvedDoG, Keypoint* keypoints, int* counter, int width, int height, int octave) 
{
	__shared__ float cache[(TILE_SIDE_FIXED + 2) * (TILE_SIDE_FIXED + 2) * (SIZE_OCTAVES + 2)];
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	int col = blockIdx.x * TILE_SIDE_FIXED + tIdx;
	int row = blockIdx.y * TILE_SIDE_FIXED + tIdy;
	
	// Get the corresponding block of pixels to the shared memory
	if (row < height && col < width) {
		int cacheOffset = (TILE_SIDE_FIXED + 2) * (SIZE_OCTAVES + 2);
		int imageOffset = width * (SIZE_OCTAVES + 3);
		for (int layer = 0; layer < SIZE_OCTAVES + 2; layer++) {
			int cacheCol = layer * (TILE_SIDE_FIXED + 2) + tIdx + 1;
			int imageIdx = row * imageOffset + layer * width + col;
			cache[(tIdy + 1) * cacheOffset + cacheCol] = convolvedDoG[imageIdx];
			if (tIdx == 0 && col != 0)
				cache[(tIdy + 1) * cacheOffset + cacheCol - 1] = convolvedDoG[imageIdx - 1];
			if (tIdx == TILE_SIDE_FIXED - 1 && col != width - 1)
				cache[(tIdy + 1) * cacheOffset + cacheCol + 1] = convolvedDoG[imageIdx + 1];
			if (tIdy == 0 && row != 0)
				cache[cacheCol] = convolvedDoG[imageIdx - imageOffset];
			if (tIdy == TILE_SIDE_FIXED - 1 && row != height - 1)
				cache[(tIdy + 2) * cacheOffset + cacheCol] = convolvedDoG[imageIdx + imageOffset];
		}
	}
	__syncthreads();

	// Look for the keypoints in the DoG pyramid
	const int N_NEIGHBORS = 27;
	float neighbors[N_NEIGHBORS];
	if (row > SIFT_BORDER && row < height - SIFT_BORDER && col > SIFT_BORDER && col < width - SIFT_BORDER) {
		for (int layer = 1; layer < SIZE_OCTAVES + 1; layer++) {
			getPointNeighborhood(neighbors, cache, tIdy, tIdx, layer);
			if (isMaximum(neighbors) || isMinimum(neighbors)) {
				Keypoint* kp = new Keypoint((float)row, (float)col, layer, octave, layer);
				bool adjusted = adjustKeypoint(kp, neighbors, convolvedDoG, width, height);
				bool keep = keepKeypoint(kp, neighbors, convolvedDoG, width);
				if (adjusted && keep) {
					int oldCounter = atomicAdd(&counter[0], 1);
					if (oldCounter < MAX_N_KEYPOINTS) {
						keypoints[oldCounter].x = kp->x;
						keypoints[oldCounter].y = kp->y;
						keypoints[oldCounter].xAdj = kp->xAdj;
						keypoints[oldCounter].yAdj = kp->yAdj;
						keypoints[oldCounter].sigma = kp->sigma;
						keypoints[oldCounter].octave = kp->octave;
						keypoints[oldCounter].layer = kp->layer;
					}
				}
				delete kp;
			}
		}
	}
}

__device__ void getPointNeighborhood(float* neighbors, float* cache, int tIdy, int tIdx, int layer)
{
	int neighIdx = 0;
	int cacheIdx = (tIdy + 1) * (TILE_SIDE_FIXED + 2) * (SIZE_OCTAVES + 2) +
				   (layer - 1) * (TILE_SIDE_FIXED + 2) + tIdx + 1;
	int cacheOffset = (TILE_SIDE_FIXED + 2) * (SIZE_OCTAVES + 2);
	for (int i = 0; i < 3; i++) {
		neighbors[neighIdx++] = cache[cacheIdx - cacheOffset - 1]; // up-left  
		neighbors[neighIdx++] = cache[cacheIdx - cacheOffset]; // up
		neighbors[neighIdx++] = cache[cacheIdx - cacheOffset + 1]; // up-right
		neighbors[neighIdx++] = cache[cacheIdx - 1]; // left
		neighbors[neighIdx++] = cache[cacheIdx]; // center
		neighbors[neighIdx++] = cache[cacheIdx + 1]; // right
		neighbors[neighIdx++] = cache[cacheIdx + cacheOffset - 1]; // down-left
		neighbors[neighIdx++] = cache[cacheIdx + cacheOffset]; // down
		neighbors[neighIdx++] = cache[cacheIdx + cacheOffset + 1]; // down-right
		cacheIdx += TILE_SIDE_FIXED + 2;
	}
}

__device__ void getPointNeighborhoodFromImage(float* neighbors,
	float* convolvedDoG, int row, int col, int layer, int width)
{
	int neighIdx = 0;
	int imageIdx = row * width * (SIZE_OCTAVES + 3) + (layer - 1) * width + col;
	int imageOffset = width * (SIZE_OCTAVES + 3);
	for (int i = 0; i < 3; i++) {
		neighbors[neighIdx++] = convolvedDoG[imageIdx - imageOffset - 1]; // up-left
		neighbors[neighIdx++] = convolvedDoG[imageIdx - imageOffset]; // up
		neighbors[neighIdx++] = convolvedDoG[imageIdx - imageOffset + 1]; // up-right
		neighbors[neighIdx++] = convolvedDoG[imageIdx - 1]; // left
		neighbors[neighIdx++] = convolvedDoG[imageIdx]; // center
		neighbors[neighIdx++] = convolvedDoG[imageIdx + 1]; // right
		neighbors[neighIdx++] = convolvedDoG[imageIdx + imageOffset - 1]; // down-left
		neighbors[neighIdx++] = convolvedDoG[imageIdx + imageOffset]; // down
		neighbors[neighIdx++] = convolvedDoG[imageIdx + imageOffset + 1]; // down-right
		imageIdx += width;
	}
}

__device__ bool isMaximum(float* neighbors)
{
	const int CENTER = 13;
	for (int i = 0; i < 27; i++) {
		if (i == CENTER) continue;
		if (neighbors[CENTER] <= neighbors[i]) return false;
	}
	return true;
}

__device__ bool isMinimum(float* neighbors)
{
	const int CENTER = 13;
	for (int i = 0; i < 27; i++) {
		if (i == CENTER) continue;
		if (neighbors[CENTER] >= neighbors[i]) return false;
	}
	return true;
}

__device__ float getValueAtExtremum(float* neighbors, float* alpha)
{
	const int CENTER = 13;
	// Perform quadratic interpolation
	float g[3] = {
		0.5f * (neighbors[22] - neighbors[4]),
		0.5f * (neighbors[16] - neighbors[10]),
		0.5f * (neighbors[14] - neighbors[12]) };
	// Calculate entries of 3x3 Hessian matrix
	float h11 = neighbors[22] + neighbors[4] - 2 * neighbors[CENTER];
	float h22 = neighbors[16] + neighbors[10] - 2 * neighbors[CENTER];
	float h33 = neighbors[14] + neighbors[12] - 2 * neighbors[CENTER];
	float h12 = 0.25f * (neighbors[25] - neighbors[19] - neighbors[7] + neighbors[1]);
	float h13 = 0.25f * (neighbors[23] - neighbors[21] - neighbors[5] + neighbors[3]);
	float h23 = 0.25f * (neighbors[17] - neighbors[15] - neighbors[11] + neighbors[9]);
	float h21 = h12, h31 = h13, h32 = h23;
	float det = h11 * (h22 * h33 - h32 * h23) - h12 * (h21 * h33 - h23 * h31) + h13 * (h21 * h32 - h22 * h31);
	// Calculate the inverse of 3x3 Hessian matrix
	float invDet = fabsf(1.f / det);
	float h11Inv = (h22 * h33 - h32 * h23) * invDet;
	float h12Inv = (h13 * h32 - h12 * h33) * invDet;
	float h13Inv = (h12 * h23 - h13 * h22) * invDet;
	float h21Inv = (h23 * h31 - h21 * h33) * invDet;
	float h22Inv = (h11 * h33 - h13 * h31) * invDet;
	float h23Inv = (h21 * h13 - h11 * h23) * invDet;
	float h31Inv = (h21 * h32 - h31 * h22) * invDet;
	float h32Inv = (h31 * h12 - h11 * h32) * invDet;
	float h33Inv = (h11 * h22 - h21 * h12) * invDet;
	float invHessian[9] = { h11Inv, h12Inv, h13Inv, h21Inv, h22Inv, h23Inv, h31Inv, h32Inv, h33Inv };
	float extremumVal = neighbors[CENTER];
	for (int i = 0; i < 3; i++) {
		alpha[i] = 0.f;
		for (int j = 0; j < 3; j++) {
			alpha[i] += -invHessian[3 * i + j] * g[j];
		}
		extremumVal += 0.5f * alpha[i] * g[i];
	}
	return extremumVal;
}

__device__ bool adjustKeypoint(Keypoint* kp, float* neighbors, float* convolvedDoG, int width, int height)
{
	float l = (float)kp->layer, x = kp->x, y = kp->y;
	float maxAlpha = 1.f;
	const int CENTER = 13;

	// Conservative test for low contrast
	if (fabsf(neighbors[CENTER]) < roundf(0.5f * CONTRAST_TH / (float)SIZE_OCTAVES)) return false;
	
	float extremumVal, alpha[3];
	// Iterate until the keypoint is adjusted or maximum of iterations is exceeded
	for (int i = 0; i < MAX_ADJ_ITER; i++) {
		// Calculate adjusted coordinates of the keypoint in the scale-space
		extremumVal = getValueAtExtremum(neighbors, alpha);

		// Check for the successful keypoint adjustment
		maxAlpha = fmaxf(fmaxf(fabsf(alpha[0]), fabsf(alpha[1])), fabsf(alpha[2]));
		if (maxAlpha <= 0.5f) break;

		// Update interpolating position
		l += roundf(alpha[0]);
		x += roundf(alpha[1]);
		y += roundf(alpha[2]);
		if ((l < 1 || l > SIZE_OCTAVES) || (x < SIFT_BORDER) || (y < SIFT_BORDER) ||
			(x > height - SIFT_BORDER) || (y > width - SIFT_BORDER)) return false;

		// Update the neighborhood for the new extrema position
		getPointNeighborhoodFromImage(neighbors, convolvedDoG, (int)x, (int)y, (int)l, width);
	}
	if (maxAlpha <= 0.5f) {
		// The second test for low constrast
		if (fabsf(extremumVal) < (CONTRAST_TH / SIZE_OCTAVES)) return false;
		// Calculate adjusted coordinates of the keypoint in the scale-space
		int octaveMult = (1 << kp->octave);
		float sigma_0 = sqrt(SIGMA_INIT * SIGMA_INIT - IMAGE_SIGMA * IMAGE_SIGMA);
		kp->sigma = 2 * sigma_0 * octaveMult * powf(2.f, (alpha[0] + l) / SIZE_OCTAVES);
		kp->xAdj = MIN_SAMPLING_DIST * octaveMult * (alpha[1] + x);
		kp->yAdj = MIN_SAMPLING_DIST * octaveMult * (alpha[2] + y);
		kp->x = x; kp->y = y; kp->layer = l;
		return true;
	}
	return false;
}

__device__ bool keepKeypoint(Keypoint* kp, float* neighbors, float* convolvedDoG, int width)
{
	const int CENTER = 13;
	float l = kp->layer, x = kp->x, y = kp->y;
	if ((int)kp->sigma != kp->layer) {
		getPointNeighborhoodFromImage(neighbors, convolvedDoG, (int)x, (int)y, (int)l, width);
	}
	int octaveOffset = kp->octave * (SIZE_OCTAVES + 2);
	// Calculate entries of 2x2 Hessian matrix
	float h11 = neighbors[16] + neighbors[10] - 2 * neighbors[CENTER];
	float h22 = neighbors[14] + neighbors[12] - 2 * neighbors[CENTER];
	float h12 = 0.25f * (neighbors[17] - neighbors[15] - neighbors[11] + neighbors[9]);
	float trace = h11 + h12;
	float det = h11 * h22 - h12 * h12;
	if (det <= 0.f) return false;
	// Filter based on the edgeness of the keypoint
	float edgeness = (trace * trace) / det;
	if (edgeness >= powf(EDGENESS_TH + 1.f, 2.f) / EDGENESS_TH) return false;
	return true;
}