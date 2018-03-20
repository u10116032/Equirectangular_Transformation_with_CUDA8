#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

#include "kernel.h"

int main()
{
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
	std::chrono::duration<float, std::ratio<1, 1000>> duration;

	const double PI = 3.1415926535897931;

	const int srcWidth = 6000;
	const int srcHeight = 4000;

	cudaSetDevice(0);

	uchar *deviceSrc = 0;
	size_t srcPitch;
	gpuErrorCheck(cudaMallocPitch((void**)&deviceSrc, &srcPitch, srcWidth * 3, srcHeight));

	uchar4 *image = 0;
	cudaMalloc((void**)&image, srcWidth * srcHeight * 4);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaArray *textureArray;
	cudaMallocArray(&textureArray, &channelDesc, srcWidth, srcHeight);

	const int panoramaWidth = 4096;
	const int panoramaHeight = 2048;

	uchar4 *undistortedImage = 0;
	size_t undistortedImagePitch;
	gpuErrorCheck(cudaMallocPitch((void**)&undistortedImage, &undistortedImagePitch, panoramaWidth * 4, panoramaHeight));

	cv::Mat resultMat = cv::Mat(panoramaHeight, panoramaWidth, CV_8UC4);

	for (int i = 2698; i <= 2698; ++i) {
		cv::Mat src = cv::imread("./DSC_" + std::to_string(i) + ".jpg");

		start = std::chrono::high_resolution_clock::now();

		gpuErrorCheck(cudaMemcpy2D(deviceSrc, srcPitch, src.data, srcWidth * 3, srcWidth * 3, srcHeight, cudaMemcpyHostToDevice));

		convertToBgra(deviceSrc, srcPitch, image, srcWidth, srcHeight);

		undistortion(textureArray, image, &channelDesc, undistortedImage, undistortedImagePitch, panoramaWidth, panoramaHeight, srcWidth, srcHeight);

		gpuErrorCheck(cudaMemcpy2D(resultMat.data, panoramaWidth * 4, undistortedImage, undistortedImagePitch, panoramaWidth * 4, panoramaHeight, cudaMemcpyDeviceToHost));

		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1000>>>(end - start);
		std::cout << duration.count() << std::endl;

		/*cv::resize(resultMat, resultMat, cv::Size(panoramaWidth / 5, panoramaHeight / 5));
		cv::imshow("result", resultMat);*/
		//cv::waitKey(0);

		bool success = cv::imwrite("./result.png", resultMat);

		if (success)
		std::cout << std::to_string(i) << " success" << std::endl;
		else
		std::cout << std::to_string(i) << " failed" << std::endl;

	}
	getchar();
	cudaFree(deviceSrc);
	cudaFree(image);
	cudaFree(undistortedImage);

}

