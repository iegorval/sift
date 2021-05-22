#define _USE_MATH_DEFINES

#include "siftDetectorCPU.cuh"
#include "siftDetectorGPU.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void SiftDetectorCU::computeSigmas(std::vector<float>& sigmas) 
{
	float k = pow(2.f, 1.f / SIZE_OCTAVES);
	std::vector<float> sigmasTotal; 
	sigmasTotal.resize(SIZE_OCTAVES + 3);
	sigmas[0] = std::sqrtf(SIGMA_INIT * SIGMA_INIT - IMAGE_SIGMA * IMAGE_SIGMA);
	sigmasTotal[0] = SIGMA_INIT;
	for (int i = 1; i < SIZE_OCTAVES + 3; i++) {
		sigmasTotal[i] = sigmasTotal[i - 1] * k;
		sigmas[i] = std::sqrtf(sigmasTotal[i] * sigmasTotal[i] - sigmasTotal[i - 1] * sigmasTotal[i - 1]);
	}
}

void SiftDetectorCU::computeKernels(
	std::vector<float>& kernels, std::vector<int>& kRadius, const std::vector<float>& sigmas)
{
	for (int layer = 0; layer < SIZE_OCTAVES + 3; layer++) {
		float sigma = sigmas[layer];
		int kernelSize = int(2 * std::roundf(sigma * 3.f) + 1);
		int kernelRadius = (kernelSize - 1) / 2;
		kRadius[layer] = kernelRadius;
		float normConst = 1.f / (float)(sqrt(2 * M_PI) * sigma);
		float expConst = -1.f / (2 * sigma * sigma);
		for (int i = kernelRadius; i < kernelSize; i++) {
			float x = (float)i - (float)(kernelSize - 1) / 2.f;
			kernels[layer * (MAX_KERNEL_RADIUS + 1) + (i - kernelRadius)] = normConst * exp(expConst * x * x);
		}
	}
}

void SiftDetectorCU::detectInterestPoints(const cv::Mat& image) 
{
	// OpenCV image matrix to the float array, assuming that the matrix is not neccesserily continuous
	int height = image.rows, width = image.cols;
	cv::Mat imageFloat;
	image.convertTo(imageFloat, CV_32F, 1.0 / 255, 0);
	_nOctaves = N_OCTAVES + (int)UPSAMPLING;
	height += (int)UPSAMPLING * height;
	width += (int)UPSAMPLING * width;
	int imageSize = height * width;

	float* hostImage = new float[imageSize];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (i < image.rows && j < image.cols) {
				hostImage[i * width + j] = imageFloat.at<float>(i, j);
			} else {
				hostImage[i * width + j] = 0.f;
			}
		}
	}

	// Prepare values for the Gaussian blur at all levels of an octave
	std::vector<float> sigmas; sigmas.resize(SIZE_OCTAVES + 3);
	computeSigmas(sigmas);
	std::vector<float> kernels; kernels.resize((MAX_KERNEL_RADIUS + 1) * (SIZE_OCTAVES + 3));
	std::vector<int> kRadius; kRadius.resize(SIZE_OCTAVES + 3);
	computeKernels(kernels, kRadius, sigmas);

	// Prepare host variables
	int* hostCounter = new int[1]; 
	hostCounter[0] = 0;
	float* hostKernels = &kernels[0];
	Keypoint hostKeypoints[MAX_N_KEYPOINTS];

	// Prepare device variables
	float* devImage, * devKernels, * devConvolved, * devDoG;
	Keypoint* devKeypoints;
	int* devCounter;

	// Memory allocation
	cudaMalloc((void**)&devImage, imageSize * sizeof(float));
	cudaMalloc((void**)&devConvolved, imageSize * sizeof(float));
	cudaMalloc((void**)&devCounter, sizeof(int));
	cudaMalloc((void**)&devKernels, (MAX_KERNEL_RADIUS + 1) * (SIZE_OCTAVES + 3) * sizeof(float));
	cudaMalloc((void**)&devKeypoints, MAX_N_KEYPOINTS * sizeof(Keypoint));
	cudaMalloc((void**)&devDoG, imageSize * sizeof(float) * (SIZE_OCTAVES + 3));
		
	// Transfer data to the device
	cudaMemcpy(devImage, hostImage, imageSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCounter, hostCounter, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devKernels, hostKernels, 
		(MAX_KERNEL_RADIUS + 1) * (SIZE_OCTAVES + 3) * sizeof(float), cudaMemcpyHostToDevice);

	// Get the initial image based on the assumed input image's blur
	prepareInitialImage(devImage, devKernels, width, height, kRadius[0]);

	// Build DoG pyramid and locate the keypoints
	buildDoGPyramid(devImage, devConvolved, devDoG, devKernels, devKeypoints, devCounter, width, height, kRadius);

	// Get the keypoints into a vector
	cudaMemcpy(hostCounter, devCounter, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hostKeypoints, devKeypoints, MAX_N_KEYPOINTS * sizeof(Keypoint), cudaMemcpyDeviceToHost);
	hostCounter[0] = std::min(hostCounter[0], MAX_N_KEYPOINTS);
	for (int i = 0; i < hostCounter[0]; i++) _keypoints.emplace_back(hostKeypoints[i]);
	std::cout << "KEYPOINTS FOUND: " << hostCounter[0] << std::endl;
	
	// Free the memory on GPU
	cudaFree(devImage);
	cudaFree(devCounter);
	cudaFree(devKernels);
	cudaFree(devKeypoints);
	cudaFree(devDoG);
	cudaFree(devConvolved);
}

void SiftDetectorCU::buildDoGPyramid(float* devImage, float* devConvolved, float* devDoG, float* devKernels, 
	Keypoint* devKeypoints, int *devCounter, int width, int height, std::vector<int> kRadius)
{
	int imageSize = height * width;
	// Prepare the DoG from scale-space and locate all the suitable keypoints
	for (int octave = 0; octave < _nOctaves; octave++) { 
		// Downsample the image at the end of each octave
		if (octave > 0) {
			int newWidth = width / 2;
			int newHeight = height / 2;
			dim3 gridSize = dim3(newWidth / TILE_SIDE_FIXED + (int)((newWidth % TILE_SIDE_FIXED) > 0),
								 newHeight / TILE_SIDE_FIXED + (int)((newHeight % TILE_SIDE_FIXED) > 0));
			dim3 blockSize = dim3(TILE_SIDE_FIXED, TILE_SIDE_FIXED);
			downsampleImage <<< gridSize, blockSize >>> (devImage, width, height);
			width = newWidth; height = newHeight;
			imageSize = height * width;
		}
		// Fill the DoG octave
		buildDoGOctave(devImage, devConvolved, devDoG, devKernels, devKeypoints, devCounter, width, height, octave, kRadius);
	}
}

void SiftDetectorCU::prepareInitialImage(float* devImage, float* devKernels, int width, int height, int kRad)
{
	int imageSize = height * width;
	// Allocate space for the intermediate image
	float* devImageInter;
	cudaMalloc((void**)&devImageInter, imageSize * sizeof(float));

#ifdef UPSAMPLING
	int originalWidth = width / 2, originalHeight = height / 2;
	const float rRatio = (originalHeight - 1.f) / (height - 1.f);
	const float cRatio = (originalWidth - 1.f) / (width - 1.f);
	dim3 gridSizeUp = dim3(width / TILE_SIDE_FIXED + (int)((width % TILE_SIDE_FIXED) > 0),
						   height / TILE_SIDE_FIXED + (int)((height % TILE_SIDE_FIXED) > 0));
	dim3 blockSizeUp = dim3(TILE_SIDE_FIXED, TILE_SIDE_FIXED);
	upsampleImage <<< gridSizeUp, blockSizeUp >>> (
		devImage, devImageInter, originalWidth, originalHeight, rRatio, cRatio);
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(devImage, devImageInter, imageSize * sizeof(float), cudaMemcpyDeviceToDevice);
#endif

	int cacheSize = (TILE_SIDE + 2 * MAX_KERNEL_RADIUS) * sizeof(float);
	dim3 gridSize = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0), height);
	dim3 blockSize = dim3(TILE_SIDE, 1);
	gaussianHorizontal <<< gridSize, blockSize, cacheSize >>> (
		devImage, devImageInter, nullptr, devKernels, width, height, 0, kRad, false);
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

	cacheSize = TILE_SIDE * (TILE_SIDE + 2 * MAX_KERNEL_RADIUS) * sizeof(float);
	gridSize = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0),
	                height / TILE_SIDE + (int)((height % TILE_SIDE) > 0));
	blockSize = dim3(TILE_SIDE, TILE_SIDE);
	gaussianVerticalDoG <<< gridSize, blockSize, cacheSize >>> (
		nullptr, devImageInter, devImage, devKernels, width, height, 0, kRad, false);
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize())

	cudaFree(devImageInter);
}

void SiftDetectorCU::buildDoGOctave(float* devImage, float* devConvolved, float* devDoG, float* devKernels, 
	Keypoint* devKeypoints, int* devCounter, int width, int height, int octave, std::vector<int> kRadius)
{
	int cacheSizeH = (TILE_SIDE + 2 * MAX_KERNEL_RADIUS) * sizeof(float);
	dim3 gridSizeH = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0), height);
	dim3 blockSizeH = dim3(TILE_SIDE, 1);

	int cacheSizeV = TILE_SIDE * (TILE_SIDE + 2 * MAX_KERNEL_RADIUS) * sizeof(float);
	dim3 gridSizeV = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0),
						  height / TILE_SIDE + (int)((height % TILE_SIDE) > 0));
	dim3 blockSizeV = dim3(TILE_SIDE, TILE_SIDE);
	for (int layer = 0; layer < SIZE_OCTAVES + 3; layer++) {
		gaussianHorizontal <<< gridSizeH, blockSizeH, cacheSizeH >>> (
			devImage, devConvolved, devDoG, devKernels, width, height, layer, kRadius[layer], true);
		gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

		gaussianVerticalDoG <<< gridSizeV, blockSizeV, cacheSizeV >>> (
			devImage, devConvolved, devDoG, devKernels, width, height, layer, kRadius[layer], true);
		gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
	}

#ifdef DEBUG_SHOW_DOGdow
	float* tmpFloat = new float[width * height * (SIZE_OCTAVES + 3)];
	cudaMemcpy(tmpFloat, devDoG, width * height * sizeof(float) * (SIZE_OCTAVES + 3), cudaMemcpyDeviceToHost);
	for (int layer = 0; layer < SIZE_OCTAVES + 3; layer++) {
		std::vector<float> tmp;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				float val = tmpFloat[i * width * (SIZE_OCTAVES + 3) + layer * width + j];
				tmp.emplace_back(val);
			}	
		cv::Mat test = cv::Mat(height, width, CV_32FC1);
		memcpy(test.data, tmp.data(), width * height * sizeof(float));
		std::string fullWindowName = "Layer " + std::to_string(layer);
		cv::imshow(fullWindowName, test);
		cv::waitKey(0);
		cv::destroyWindow(fullWindowName);
	}
#endif
	// Find, adjust and filter the keypoints obtained from the DoG pyramid
	findKeypoints(devDoG, devKeypoints, devCounter, width, height, octave);
}

void SiftDetectorCU::findKeypoints(float* devDoG, Keypoint* devKeypoints,
	int* devCounter, int width, int height, int octave) 
{
	int cacheSize = (TILE_SIDE_FIXED + 2) * (TILE_SIDE_FIXED + 2) * (SIZE_OCTAVES + 2) * sizeof(float);
	dim3 gridSize = dim3(width / TILE_SIDE_FIXED + (int)((width % TILE_SIDE_FIXED) > 0),
			             height / TILE_SIDE_FIXED + (int)((height % TILE_SIDE_FIXED) > 0));
	dim3 blockSize = dim3(TILE_SIDE_FIXED, TILE_SIDE_FIXED);
	
	locateExtrema << <gridSize, blockSize, cacheSize >> > (
		devDoG, devKeypoints, devCounter, width, height, octave);
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
}

void SiftDetectorCU::drawKeypoints(const cv::Mat& image, bool show, bool writeToFile, std::string fPath)
{
	if (_keypoints.size() == 0) throw "No keypoints available";
	const int draw_shift_bits = 4;
	const int draw_multiplier = 1 << draw_shift_bits;
	cv::Mat kpImage(image.size(), CV_8UC3);
	cv::cvtColor(image, kpImage, cv::COLOR_GRAY2RGB);
	std::string fullWindowName = "Detected Keypoints";
	for (auto kp : _keypoints) {
		int x = ROUND(kp.xAdj * draw_multiplier);
		int y = ROUND(kp.yAdj * draw_multiplier);
		int radius = ROUND(kp.sigma / 2 * draw_multiplier); 
		if (radius > 0.f)
			cv::circle(kpImage, cv::Point(y, x), radius, CV_RGB(255, 0, 0), 1, cv::LINE_AA, draw_shift_bits);
	}
	if (show) {
		cv::imshow(fullWindowName, kpImage);
		cv::waitKey(0);
		cv::destroyWindow(fullWindowName);
	}
	if (writeToFile) {
		cv::imwrite(fPath, kpImage);
	}
#ifdef DEBUG_FILE_OUTPUT
	std::ofstream myfile("test_gpu.txt", std::ofstream::trunc);
	for (auto kp : _keypoints) {
		if (myfile.is_open()) {
			myfile << kp.octave << "," << kp.layer << "," << kp.sigma << std::endl;
		}
	}
	myfile.close();
#endif
}

SiftDetectorCU::~SiftDetectorCU()
{
	_keypoints.clear();
}