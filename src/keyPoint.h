#ifndef KEYPOINT_H
#define KEYPOINT_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

struct Keypoint {
	CUDA_CALLABLE_MEMBER Keypoint() : x(0.f), y(0.f), octave(0), layer(-1) {};
	CUDA_CALLABLE_MEMBER Keypoint(float x, float y, float s, int octave, int layer)
		: x(x), y(y), sigma(s), octave(octave), layer(layer) {};
	float sigma, x, y, xAdj, yAdj;
	int octave, layer;
};

#endif