#define _USE_MATH_DEFINES
#include "siftDetector.h"
#include <math.h>

void SiftDetector::detectInterestPoints(const cv::Mat& image)
{
    _nOctaves = N_OCTAVES + (int)UPSAMPLING;
    buildGaussianPyr(image);
    buildDoGPyr();
    locateExtrema();
}

cv::Mat SiftDetector::convolveKernel2D(const std::vector<float>& kernel, const cv::Mat& image) 
{
    int kernelRadius = kernel.size() - 1;
    cv::Mat paddedImageH(image.rows, image.cols + 2 * kernelRadius, image.type());
    cv::Mat convolved1D(image.rows, image.cols, image.type());
    cv::copyMakeBorder(image, paddedImageH, 0, 0, kernelRadius, kernelRadius, cv::BORDER_REFLECT);
    
    for (int i = 0; i < convolved1D.rows; i++) {
        for (int j = 0; j < convolved1D.cols; j++) {
            float convResult = kernel[0] * paddedImageH.at<float>(i, j + kernelRadius);
            for (int k = 1; k <= kernelRadius; k++) {
                convResult += kernel[k] * (paddedImageH.at<float>(i, j + kernelRadius - k) +
                                           paddedImageH.at<float>(i, j + kernelRadius + k));
            }
            convolved1D.at<float>(i, j) = convResult;
        }
    }
    cv::Mat paddedImageV(image.rows + 2 * kernelRadius, image.cols, image.type());
    cv::Mat convolved2D(image.rows, image.cols, image.type());
    cv::copyMakeBorder(convolved1D, paddedImageV, kernelRadius, kernelRadius, 0, 0, cv::BORDER_REFLECT);
    for (int i = 0; i < convolved1D.rows; i++) {
        for (int j = 0; j < convolved1D.cols; j++) {
            float convResult = kernel[0] * paddedImageV.at<float>(i + kernelRadius, j);
            for (int k = 1; k < kernel.size(); k++) {
                convResult += kernel[k] * (paddedImageV.at<float>(i + kernelRadius - k, j) +
                                           paddedImageV.at<float>(i + kernelRadius + k, j));
            }
            convolved2D.at<float>(i, j) = convResult;
        }
    }
    return convolved2D;
}

cv::Mat SiftDetector::downscaleImage(const cv::Mat& image) 
{
    cv::Mat downsampledImage(image.rows / 2, image.cols / 2, image.type());
    for (int i = 0; i < downsampledImage.rows; i++) {
        for (int j = 0; j < downsampledImage.cols; j++) {
            downsampledImage.at<float>(i, j) = 0.25f * (
                image.at<float>(i * 2, j * 2) + image.at<float>(i * 2 + 1, j * 2) +
                image.at<float>(i * 2, j * 2 + 1) + image.at<float>(i * 2 + 1, j * 2 + 1));
        }
    }
    return downsampledImage;
}

cv::Mat SiftDetector::upscaleImage(const cv::Mat& image)
{
    cv::Mat upsampledImage(image.rows * 2, image.cols * 2, image.type());
    const float rRatio = (image.rows - 1.f) / (image.rows * 2.f - 1.f);
    const float cRatio = (image.cols - 1.f) / (image.cols * 2.f - 1.f);
    for (int i = 0; i < upsampledImage.rows; i++) {
        for (int j = 0; j < upsampledImage.cols; j++) {
            int cLow = (int)std::floorf(cRatio * j), rLow = (int)std::floorf(rRatio * i);
            int cHigh = (int)std::ceilf(cRatio * j), rHigh = (int)std::ceilf(rRatio * i);
            float cWeight = (cRatio * j) - (float)cLow, rWeight = (rRatio * i) - (float)rLow;
            upsampledImage.at<float>(i, j) = image.at<float>(rLow, cLow) * (1 - cWeight) * (1 - rWeight) +
                                             image.at<float>(rLow, cHigh) * cWeight * (1 - rWeight) +
                                             image.at<float>(rHigh, cLow) * (1 - cWeight) * rWeight +
                                             image.at<float>(rHigh, cHigh) * cWeight * rWeight;
        }
    }
    return upsampledImage;
}

void SiftDetector::computeSigmas() 
{
    float k = pow(2.f, 1.f / SIZE_OCTAVES);
    std::vector<float> sigmasTotal; 
    sigmasTotal.resize(SIZE_OCTAVES + 3);
    _sigmas[0] = std::max(std::sqrtf(SIGMA_INIT * SIGMA_INIT - IMAGE_SIGMA * IMAGE_SIGMA), 0.01f);
    sigmasTotal[0] = SIGMA_INIT;
    for (int i = 1; i < SIZE_OCTAVES + 3; i++) {
        sigmasTotal[i] = sigmasTotal[i - 1] * k;
        _sigmas[i] = std::sqrtf(sigmasTotal[i] * sigmasTotal[i] - sigmasTotal[i - 1] * sigmasTotal[i - 1]);
    }
}

void SiftDetector::createKernel(std::vector<float>& kernel, float sigma)
{
    int kernelSize = int(2 * std::roundf(sigma * 3.f) + 1);
    int kernelRadius = (kernelSize - 1) / 2;
    float normConst = 1.f / (float)(sqrt(2 * M_PI) * sigma);
    float expConst = -1.f / (2 * sigma * sigma);
    for (int i = kernelRadius; i < kernelSize; i++) {
        float x = (float)i - (float)(kernelSize - 1) / 2.f;
        kernel.emplace_back(normConst * exp(expConst * x * x));
    }
}

void SiftDetector::buildGaussianPyr(const cv::Mat& image) 
{
    _sigmas.resize(SIZE_OCTAVES + 3);
    computeSigmas();
    std::vector<std::vector<float>> kernels;
    for (int layer = 0; layer < SIZE_OCTAVES + 3; layer++) {
        std::vector<float> kernel;
        createKernel(kernel, _sigmas[layer]);
        kernels.emplace_back(kernel);
    }
    cv::Mat imageForPyramid;
    image.convertTo(imageForPyramid, CV_32F, 1.0 / 255, 0);
    _pyrGaussian.resize(_nOctaves * (SIZE_OCTAVES + 3));
#if UPSAMPLING
    imageForPyramid = upscaleImage(imageForPyramid);
#endif
    imageForPyramid = convolveKernel2D(kernels[0], imageForPyramid);
    int pyrIdx = 0;
    for (int octave = 0; octave < _nOctaves; octave++) {
        for (int layer = 0; layer < SIZE_OCTAVES + 3; layer++) {
            if (octave == 0 && layer == 0) {
                // If it is the level in the first octave, prepare the initial image by Gaussian blur
                _pyrGaussian[pyrIdx++] = imageForPyramid.clone();
            } else if (octave != 0 && layer == 0) {
                // If it is the first level in any octave (except for the first octave), downsample the image
                int baseIdx = (octave - 1) * (SIZE_OCTAVES + 3) + SIZE_OCTAVES;
                imageForPyramid = downscaleImage(_pyrGaussian[baseIdx]);
                _pyrGaussian[pyrIdx++] = imageForPyramid.clone();
            } else {
                // Otherwise, convolve with the Gaussian filter of corresponding sigma
                imageForPyramid = convolveKernel2D(kernels[layer], _pyrGaussian[pyrIdx - 1]);
                _pyrGaussian[pyrIdx++] = imageForPyramid.clone();
            }
        }
    }
#ifdef DEBUG_SHOW_GAUSS
    pyrIdx = 0;
    for (int octave = 0; octave < _nOctaves; octave++) {
        int octaveOffset = octave * (SIZE_OCTAVES + 3);
        for (int layer = 0; layer < SIZE_OCTAVES + 3; layer++) {
            std::string fullWindowName = "Layer " + std::to_string(layer);
            cv::imshow(fullWindowName, _pyrGaussian[pyrIdx++]);
            cv::waitKey(0);
            cv::destroyWindow(fullWindowName);
        }
    }
#endif
}

void SiftDetector::buildDoGPyr() 
{
    int pyrIdx = 0;
    for (int octave = 0; octave < _nOctaves; octave++) {
        int octaveOffset = octave * (SIZE_OCTAVES + 3);
        for (int layer = 0; layer < SIZE_OCTAVES + 2; layer++) {
            cv::Mat imageTop = _pyrGaussian[octaveOffset + layer + 1].clone();
            cv::Mat imageBot = _pyrGaussian[octaveOffset + layer].clone();
            cv::Mat imageDoG = imageTop - imageBot;
            _pyrDoG.push_back(imageDoG);
#ifdef DEBUG_SHOW_DOG
            std::string fullWindowName = "Layer " + std::to_string(layer);
            cv::imshow(fullWindowName, imageDoG);
            cv::waitKey(0);
            cv::destroyWindow(fullWindowName);
#endif
        }
    }
}

void SiftDetector::locateExtrema() 
{
    for (int octave = 0; octave < _nOctaves; octave++) {
        int octaveOffset = octave * (SIZE_OCTAVES + 2);
        for (int layer = 1; layer < SIZE_OCTAVES + 1; layer++) {
            cv::Mat imageBot = _pyrDoG[octaveOffset + layer - 1];
            cv::Mat imageCur = _pyrDoG[octaveOffset + layer];
            cv::Mat imageTop = _pyrDoG[octaveOffset + layer + 1];
            for (int x = SIFT_BORDER; x < imageCur.rows - SIFT_BORDER; x++) {
                for (int y = SIFT_BORDER; y < imageCur.cols - SIFT_BORDER; y++) {
                    float curValue = imageCur.at<float>(x, y);
                    std::vector<float> neighborhood = getPointNeighborhood(x, y, imageCur, imageBot, imageTop);
                    if ((std::all_of(neighborhood.begin(), neighborhood.end(), [&](float u) { return u < curValue; })) ||
                        (std::all_of(neighborhood.begin(), neighborhood.end(), [&](float u) { return u > curValue; }))) {
                        Keypoint* kp = new Keypoint ((float)x, (float)y, _sigmas[layer], octave, layer);
                        if (adjustKeypoint(kp) && keepKeypoint(kp)) _keypoints.emplace_back(kp);
                    }
                }
            }
        } 
    }
    std::cout << "KEYPOINTS FOUND: " << _keypoints.size() << std::endl;
    
}

std::vector<float> SiftDetector::getPointNeighborhood(
    int x, int y, cv::Mat& imageCur, cv::Mat& imageBot, cv::Mat& imageTop) 
{
    std::vector<float> neighborhood {
        imageCur.at<float>(x - 1, y), imageCur.at<float>(x + 1, y), imageCur.at<float>(x - 1, y - 1),
        imageCur.at<float>(x + 1, y - 1), imageCur.at<float>(x - 1, y + 1), imageCur.at<float>(x + 1, y + 1),
        imageCur.at<float>(x, y + 1), imageCur.at<float>(x, y - 1),
        imageBot.at<float>(x, y), imageBot.at<float>(x - 1, y), imageBot.at<float>(x + 1, y),
        imageBot.at<float>(x - 1, y - 1), imageBot.at<float>(x + 1, y - 1), imageBot.at<float>(x - 1, y + 1),
        imageBot.at<float>(x + 1, y + 1), imageBot.at<float>(x, y + 1), imageBot.at<float>(x, y - 1),
        imageTop.at<float>(x, y), imageTop.at<float>(x - 1, y), imageTop.at<float>(x + 1, y),
        imageTop.at<float>(x - 1, y - 1), imageTop.at<float>(x + 1, y - 1), imageTop.at<float>(x - 1, y + 1),
        imageTop.at<float>(x + 1, y + 1), imageTop.at<float>(x, y + 1), imageTop.at<float>(x, y - 1)
    };
    return neighborhood;
}

std::vector<float> SiftDetector::getHessian(
    int x, int y, cv::Mat& imageCur, cv::Mat& imageBot, cv::Mat& imageTop) 
{
    float h11 = imageTop.at<float>(x, y) + imageBot.at<float>(x, y) - 2.f * imageCur.at<float>(x, y);
    float h22 = imageCur.at<float>(x + 1, y) + imageCur.at<float>(x - 1, y) - 2.f * imageCur.at<float>(x, y);
    float h33 = imageCur.at<float>(x, y + 1) + imageCur.at<float>(x, y - 1) - 2.f * imageCur.at<float>(x, y);
    float h12 = 0.25f * (imageTop.at<float>(x + 1, y) - imageTop.at<float>(x - 1, y) - 
        imageBot.at<float>(x + 1, y) + imageBot.at<float>(x - 1, y));
    float h13 = 0.25f * (imageTop.at<float>(x, y + 1) - imageTop.at<float>(x, y - 1) -
        imageBot.at<float>(x, y + 1) + imageBot.at<float>(x, y - 1));
    float h23 = 0.25f * (imageCur.at<float>(x + 1, y + 1) - imageCur.at<float>(x + 1, y - 1) -
        imageCur.at<float>(x - 1, y + 1) + imageCur.at<float>(x - 1, y - 1));
    std::vector<float> hessian{ h11, h12, h13, h12, h22, h23, h13, h23, h33 };
    return hessian;
}

bool SiftDetector::adjustKeypoint(Keypoint* kp) 
{
    float l = (float)kp->layer, x = kp->x, y = kp->y;
    float maxAlpha = 1.f;
    int octaveOffset = kp->octave * (SIZE_OCTAVES + 2);

    // Conservative test for low contrast
    cv::Mat imageCur = _pyrDoG[octaveOffset + l];
    if (std::abs(imageCur.at<float>(x, y)) < roundf(0.5f * CONTRAST_TH / (float)SIZE_OCTAVES))  return false; 

    cv::Mat alpha; 
    float extremumVal;
    // Iterate until the keypoint is adjusted or maximum of iterations is exceeded
    for (int i = 0; i < MAX_ADJ_ITER; i++) {
        cv::Mat imageBot = _pyrDoG[octaveOffset + l - 1];
        cv::Mat imageCur = _pyrDoG[octaveOffset + l];
        cv::Mat imageTop = _pyrDoG[octaveOffset + l + 1];

        // Perform quadratic interpolation
        cv::Mat gMat = (cv::Mat_<float>(3, 1) << 
            0.5f * (imageTop.at<float>(x, y) - imageBot.at<float>(x, y)),
            0.5f * (imageCur.at<float>(x + 1, y) - imageCur.at<float>(x - 1, y)),
            0.5f * (imageCur.at<float>(x, y + 1) - imageCur.at<float>(x, y - 1)));
        auto hessian = getHessian((int)x, (int)y, imageCur, imageBot, imageTop);

        cv::Mat hessianMat = cv::Mat(3, 3, CV_32F, hessian.data());
        alpha = -(hessianMat.inv()) * gMat;
        cv::Mat wMat = imageCur.at<float>(x, y) + 0.5f * gMat.t() * alpha;
        extremumVal = wMat.at<float>(0, 0);

        maxAlpha = std::max({ 
            std::abs(alpha.at<float>(0)), std::abs(alpha.at<float>(1)), std::abs(alpha.at<float>(2)) });
        if (maxAlpha <= 0.5f) break;

        // Update interpolating position
        l += std::roundf(alpha.at<float>(0)); 
        x += std::roundf(alpha.at<float>(1)); 
        y += std::roundf(alpha.at<float>(2)); 
        if ((l < 1 || l > SIZE_OCTAVES) || (x < SIFT_BORDER) || (y < SIFT_BORDER) ||
            (x > imageCur.rows - SIFT_BORDER) ||( y > imageCur.cols - SIFT_BORDER)) {
            return false;
        }
    }
    // The keypoint will be ignored if the adjustment is unsuccessful
    if (maxAlpha <= 0.5f) {
        // The second test for low constrast
        if (std::abs(extremumVal) < (CONTRAST_TH / SIZE_OCTAVES)) return false; 
        // Calculate adjusted coordinates of the keypoint in the scale-space
        int octaveMult = (1 << kp->octave);
        kp->sigma = 2 * _sigmas[0] * octaveMult * std::powf(2.f, (alpha.at<float>(0) + l) / SIZE_OCTAVES);
        kp->xAdj = MIN_SAMPLING_DIST * octaveMult * (alpha.at<float>(1) + x);
        kp->yAdj = MIN_SAMPLING_DIST * octaveMult * (alpha.at<float>(2) + y);
        kp->x = x; kp->y = y; kp->layer = l;
        return true;
    }
    return false;
}

bool SiftDetector::keepKeypoint(Keypoint* kp) 
{
    float x = kp->x; float y = kp->y;
    cv::Mat imageCur = _pyrDoG[kp->octave * (SIZE_OCTAVES + 2) + kp->layer];
    float h11 = (imageCur.at<float>(x + 1, y) + imageCur.at<float>(x - 1, y) - 2 * imageCur.at<float>(x, y));
    float h22 = (imageCur.at<float>(x, y + 1) + imageCur.at<float>(x, y - 1) - 2 * imageCur.at<float>(x, y));
    float h12 = 0.25f * (imageCur.at<float>(x + 1, y + 1) - imageCur.at<float>(x + 1, y - 1) -
        imageCur.at<float>(x - 1, y + 1) + imageCur.at<float>(x - 1, y - 1));
    float det = h11 * h22 - h12 * h12;
    if (det <= 0.f) return false;
    float trace = h11 + h22;
    float edgeness = (trace * trace) / det;
    if (edgeness >= std::powf(EDGENESS_TH + 1, 2.f) / (float)EDGENESS_TH) return false;
    return true;
}

void SiftDetector::drawKeypoints(const cv::Mat& image, bool show, bool writeToFile, std::string fPath)
{
    if (_keypoints.size() == 0) throw "No keypoints available";
    const int draw_shift_bits = 4;
    const int draw_multiplier = 1 << draw_shift_bits;
    cv::Mat kpImage(image.size(), CV_8UC3);
    cv::cvtColor(image, kpImage, cv::COLOR_GRAY2RGB);
    std::string fullWindowName = "Detected Keypoints";
    for (auto kp : _keypoints) {
        int x = ROUND(kp->xAdj * draw_multiplier);
        int y = ROUND(kp->yAdj * draw_multiplier);
        int radius = ROUND(kp->sigma / 2 * draw_multiplier);
        cv::circle(kpImage, cv::Point(y, x), radius, CV_RGB(255, 0, 0), 1, cv::LINE_AA, draw_shift_bits);
    }
    if (show) {
        cv::imshow(fullWindowName, kpImage);
        cv::waitKey(0);
        cv::destroyWindow(fullWindowName);
    }
    if (writeToFile) {
        cv:imwrite(fPath, kpImage);
    }
#ifdef DEBUG_FILE_OUTPUT
    std::ofstream myfile("test_cpu.txt", std::ofstream::trunc);
    for (auto kp : _keypoints) {
        if (myfile.is_open()) {
            myfile << kp->octave << "," << kp->layer << "," << kp->sigma << std::endl;
        }
    }
    myfile.close();
#endif
}

SiftDetector::~SiftDetector() 
{
    for (auto kp : _keypoints) delete kp;
    _keypoints.clear();
}