#pragma once

// Macros
#define CEIL(X) ( (X - (int)X)==0 ? (int)X : (int)X+1 )
#define ROUND(X) (int)( X + (X >= 0 ? 0.5f : -0.5f) );

// SIFT parameters
#define UPSAMPLING true
#define N_OCTAVES 4     // excluding the upsampled octave
#define SIZE_OCTAVES 3
#define SIGMA_INIT 1.6f
#define MAX_ADJ_ITER 5
#define CONTRAST_TH 0.04
#define EDGENESS_TH 10
#define MAX_KERNEL_SIZE 20 
#define MAX_KERNEL_RADIUS int((MAX_KERNEL_SIZE - 1) / 2)
#define MAX_N_KEYPOINTS 10000
#define SIFT_BORDER 10
#ifdef UPSAMPLING
#define MIN_SAMPLING_DIST 0.5f
#define IMAGE_SIGMA 1.f
#else
#define MIN_SAMPLING_DIST 1.0f
#define IMAGE_SIGMA 0.5f
#endif

// CUDA parameters
#define TILE_SIDE MAX_KERNEL_SIZE 
#define TILE_SIDE_FIXED 8

// Debugging parameters
//#define DEBUG_SHOW_GAUSS  // available only for CPU version
//#define DEBUG_SHOW_DOG
//#define DEBUG_FILE_OUTPUT
#define DEBUG_LOG_TIME
