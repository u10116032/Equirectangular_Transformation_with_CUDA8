#include "kernel.h"

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> textureImage;

inline void gpuAssert(cudaError_t error, const char *file, int line, bool abort)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
		if (abort)
			exit(error);
	}
}

__global__ void ConvertToBgra(uchar *src, size_t srcPitch, uchar4 *dst, int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height) {
		int srcIndex = y * srcPitch + x * 3;
		int dstIndex = y * width + x;

		dst[dstIndex].x = src[srcIndex];
		dst[dstIndex].y = src[srcIndex + 1];
		dst[dstIndex].z = src[srcIndex + 2];
		dst[dstIndex].w = 255;
	}
}

void convertToBgra(uchar *deviceSrc, size_t srcPitch, uchar4 *image, int width, int height)
{
	dim3 threadsPecrBlock(32, 32, 1);
	dim3 blocksPerGrid((width / threadsPecrBlock.x) + 1, (height / threadsPecrBlock.y) + 1, 1);
	ConvertToBgra << <blocksPerGrid, threadsPecrBlock, 1 >> > (deviceSrc, srcPitch, image, width, height);

	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaDeviceSynchronize());
}

__global__ void Undistortion(uchar4 *result, size_t resultPitch, int panoramaWidth, int panoramaHeight, int srcWidth, int srcHeight)
{
	float PI = 3.1415926535897931;

	float fx = 3331.585269009607;
	float fy = 3333.875783213165;
	float cx = 3015.591157338786;
	float cy = 2105.588990694724;

	float k1 = -0.6044104008292924;
	float k2 = 0.1613746793368791;
	float p1 = -0.006228015039654566;
	float p2 = -0.0002828451886688114;

	float xi = 1.197982220299675;

	float rotationMatrix[9]{ 0.0 };
	float angle = 6.0 * PI / 180.0;
	rotationMatrix[0] = 1.0;
	rotationMatrix[4] = cos(angle);
	rotationMatrix[5] = -sin(angle);
	rotationMatrix[7] = sin(angle);
	rotationMatrix[8] = cos(angle);

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < panoramaWidth && y < panoramaHeight) {
		float phi = ((float)x / (float)(panoramaWidth - 1)) * 2.0 * PI;
		float theta = ((float)y / (float)(panoramaHeight - 1))  * PI;

		float temp[3];
		temp[0] = sin(theta) * cos(phi);
		temp[1] = sin(theta) * sin(phi);
		temp[2] = cos(theta);

		float worldPoint[3];
		worldPoint[0] = rotationMatrix[0] * temp[0] + rotationMatrix[1] * temp[1] + rotationMatrix[2] * temp[2];
		worldPoint[1] = rotationMatrix[3] * temp[0] + rotationMatrix[4] * temp[1] + rotationMatrix[5] * temp[2];
		worldPoint[2] = rotationMatrix[6] * temp[0] + rotationMatrix[7] * temp[1] + rotationMatrix[8] * temp[2];


		if (worldPoint[2] < 0)
			return;
		worldPoint[2] += xi;

		float inverseZ = 1.0 / worldPoint[2];

		float normalizedX = worldPoint[0] * inverseZ;
		float normalizedY = worldPoint[1] * inverseZ;

		float radius = sqrt(normalizedX * normalizedX + normalizedY * normalizedY);
		float radius2 = radius * radius;
		float radius4 = radius2 * radius2;

		float radialX = normalizedX * (k1 * radius2 + k2 * radius4);
		float radialY = normalizedY * (k1 * radius2 + k2 * radius4);


		float tangentialX = 2.0 * p1 * normalizedX * normalizedY + p2 * (radius2 + 2.0 * normalizedX * normalizedX);
		float tangentialY = p1 * (radius2 + 2.0 * normalizedY * normalizedY) + 2.0 * p2 * normalizedX * normalizedY;

		float imageX = (normalizedX + radialX + tangentialX) * fx + cx;
		float imageY = (normalizedY + radialY + tangentialY) * fy + cy;

		if (imageX >= 0 && imageX < srcWidth && imageY >= 0 && imageY < srcHeight) {

			result[y * resultPitch + x].x = (uchar)(tex2D(textureImage, imageX, imageY).x * 255.0f);
			result[y * resultPitch + x].y = (uchar)(tex2D(textureImage, imageX, imageY).y * 255.0f);
			result[y * resultPitch + x].z = (uchar)(tex2D(textureImage, imageX, imageY).z * 255.0f);
			result[y * resultPitch + x].w = (uchar)(tex2D(textureImage, imageX, imageY).w * 255.0f);

		}
	}
}

void undistortion(cudaArray *textureArray, uchar4 *image, cudaChannelFormatDesc *channelDesc, uchar4 *result, size_t resultPitch, int panoramaWidth, int panoramaHeight, int srcWidth, int srcHeight)
{
	cudaMemcpyToArray(textureArray, 0, 0, image, srcWidth * srcHeight * 4, cudaMemcpyDeviceToDevice);
	textureImage.addressMode[0] = cudaAddressModeClamp;
	textureImage.addressMode[1] = cudaAddressModeClamp;
	textureImage.filterMode = cudaFilterModeLinear;
	textureImage.normalized = false;
	cudaBindTextureToArray(textureImage, textureArray);

	dim3 threadsPecrBlock(32, 32, 1);
	dim3 blocksPerGrid((panoramaWidth / threadsPecrBlock.x) + 1, (panoramaHeight / threadsPecrBlock.y) + 1, 1);
	Undistortion << <blocksPerGrid, threadsPecrBlock >> > (result, resultPitch / 4, panoramaWidth, panoramaHeight, srcWidth, srcHeight);

	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaDeviceSynchronize());
}
