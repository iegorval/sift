#include <opencv2/opencv.hpp>
#include <fstream>
#include "keyPoint.h"
#include "definitions.h"

class SiftDetector {
public:
    /**
    * Empty constructor.
    */
    SiftDetector() {};
    /**
    * Default destructor.
    * 
    * Clears the keypoints.
    */
    ~SiftDetector();
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
    std::vector<float> _sigmas; ///< Contains the values of sigma for each level.
    std::vector<cv::Mat> _pyrGaussian; ///< Contains the Gaussian scale-space pyramid.
    std::vector<cv::Mat> _pyrDoG; ///< Contains the DoG scale-space pyramid.
    std::vector<Keypoint*> _keypoints; ///< Contains the detected keypoints.

    /**
    * Fills the filter's kernel.
    *
    * Takes the current sigma value and fills the Gaussian filter's kernel with the
    * corresponding values. The filter size is automatically determined from sigma.
    *
    * @param kernel - kernel of the Gaussian filter.
    * @param sigma - sigma value for the kernel.
    */
    void createKernel(std::vector<float>& kernel, float sigma);
    /**
    * Calculates all the sigmas.
    *
    * Fills the _sigmas array with all the sigmas for a single octave. All the
    * octaves use the same sequence of sigmas. Every sigma is computed relatively
    * to the previous sigma, so that the Gaussian blur is performed incrementally.
    * The first sigma is computed relatively to the assumed input blur of the image.
    */
    void computeSigmas();
    /**
    * Applies 2D Gaussian blur.
    *
    * Performs the 2D Gaussian blur by sequentially applying a horizontal and
    * vertical Gaussian kernels with pre-computed values.
    *
    * @param kernel - pre-computed kernel for the current layer.
    * @param image - input image to the blur.
    * 
    * @return image convolved with kernel.
    */
    cv::Mat convolveKernel2D(const std::vector<float>& kernel, const cv::Mat& image);
    /**
    * Downscales by 2.
    *
    * Performs the downscaling by 2. The value of the downscaled pixel is determined
    * by taking the average of the corresponding 2x2 block of the original image.
    *
    * @param image - input image to the downscaling.
    * 
    * @return downscaled image.
    */
    cv::Mat downscaleImage(const cv::Mat& image);
    /**
    * Upscales by 2.
    *
    * Performs the upscaling by 2. Each pixel in the upscaled image corresponds to 4
    * values in the original image, with the weights determined by linear interpolation.
    *
    * @param image - input image to the upscaling.
    * 
    * @return upscaled image.
    */
    cv::Mat upscaleImage(const cv::Mat& image);
    /**
    * Builds the Gaussian pyramid.
    *
    * Builds the Gaussian scale-space pyramid. The first layer is taken as the input
    * image after initial blur and possible upsampling. Then, the Gaussian blur is
    * iteratively applied within each octave, and the downsampling is applied at the
    * end of each octave. The process is continued until all the octaves are filled.
    *
    * @param image - input grayscale image.
    */
    void buildGaussianPyr(const cv::Mat& image);
    /**
    * Build the DoG pyramid.
    *
    * Builds the DoG scale-space pyramid by computing the difference of each two
    * subsequent layers in the Gaussian pyramid withint each octave.
    */
    void buildDoGPyr();
    /**
    * Locates the keypoints.
    *
    * Takes all the pixels in the scale-space (except for the first and last layer
    * of each DoG octave) and checks them for being a local extremum (i.e. minimum
    * or a maximum) in the 26-neighborhood. If the pixel is an extremum, it is listed
    * as a possible keypoint. The keypoint is then subsequently adjusted and filtered.
    * Only the keypoints passing all the tests are recorded to the final list of
    * keypoints.
    */
    void locateExtrema();
    /**
    * Computes 3x3 Hessian matrix.
    *
    * Computes the 3x3 Hessian matrix by calculating the derivatives as the differences
    * between the neighboring pixels in the 26-neighborhood of the scale-space.
    *
    * @param x - x-coordinate of a pixel.
    * @param y - y-coordinate of a pixel.
    * @param imageCur - DoG image in the current scale-space layer.
    * @param imageBot - DoG image in the previous scale-space layer.
    * @param imageTop - DoG image in the next scale-space layer.
    */
    std::vector<float> getHessian(
        int x, int y, cv::Mat& imageCur, cv::Mat& imageBot, cv::Mat& imageTop);
    /**
    * Adjusts the keypoints & filters by contrast.
    *
    * Iteratively adjusts the location of the detected keypoints by the quadratic
    * interpolation. If this process is successful within the pre-selected number of
    * iterations, the keypoint is kept. The keypoint is additionally checked against
    * the contrast threshold. The keypoints with the low contrast are discarded.
    *
    * @param kp - keypoints to be adjusted.
    */
    bool adjustKeypoint(Keypoint* kp);
    /**
    * Filters the keypoint by edgeness.
    *
    * Filters the keypoint by the edgeness response. It depends on the ratio of the
    * eigenvalues of the corresponding 2x2 Hessian matrix. This ratio is calculated
    * from the trace and teteminant of the Hessian instead of the direct computation
    * of the eigenvalues. The keypoints with the low edgeness value are discarded.
    *
    * @param
    */
    bool keepKeypoint(Keypoint* kp);
    /**
    * Gets the 26-neighborhood of a point.
    *
    * Gets a vector of pixel values in the 26-neighborhood of a pixel in scale-space.
    *
    * @param x - x-coordinate of a pixel.
    * @param y - y-coordinate of a pixel.
    * @param imageCur - DoG image in the current scale-space layer.
    * @param imageBot - DoG image in the previous scale-space layer.
    * @param imageTop - DoG image in the next scale-space layer.
    */
    std::vector<float> getPointNeighborhood(
        int x, int y, cv::Mat& imageCur, cv::Mat& imageBot, cv::Mat& imageTop);
};