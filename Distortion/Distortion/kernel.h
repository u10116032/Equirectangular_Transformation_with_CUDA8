#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef unsigned char uchar;

#define gpuErrorCheck(error) { gpuAssert((error), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t error, const char *file, int line, bool abort = false);

__global__ void ConvertToBgra(uchar *src, size_t srcPitch, uchar4 *dst, int width, int height);
void convertToBgra(uchar *deviceSrc, size_t srcPitch, uchar4 *image, int width, int height);

__global__ void Undistortion(uchar4 *result, size_t resultPitch, int panoramaWidth, int panoramaHeight, int srcWidth, int srcHeight);
void undistortion(cudaArray *textureArray, uchar4 *image, cudaChannelFormatDesc *channelDesc, uchar4 *result, size_t resultPitch, int panoramaWidth, int panoramaHeight, int srcWidth, int srcHeight);
