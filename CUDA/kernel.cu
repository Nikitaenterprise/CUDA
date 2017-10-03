
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>
#include <string>

#include "Hash.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
__global__ void multKernel(int *c, const int *a, const int *b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n) c[i] = a[i] * b[((i * 1) % n + n) % n];
}

//int main(int argc, char * argv[])
//{
//	int t = GetTickCount();
//	int  n = 10000000;
//	int *a = new int[n];
//	int *b = new int[n];
//	int *c = new int[n];
//	int *d = new int[n];
//	int *_a;
//	int *_b;
//	int *_c;
//
//	for (int i = 0; i<n; i++)
//	{
//		a[i] = rand();
//		b[i] = rand();
//	}
//	int startTime = clock();
//	for (int i = 0; i < n; i++)
//	{
//		c[i] = a[i] * b[((i * 1) % n + n )%n];
//	}
//	int endTime = clock();
//	std::cout << "Run time on CPU = " << endTime - startTime << std::endl;
//
//	if (cudaMalloc((void**)&_a, n * sizeof(int)) != cudaSuccess) std::cout << "Error CudaMalloc1";
//	if (cudaMalloc((void**)&_b, n * sizeof(int)) != cudaSuccess) std::cout << "Error CudaMalloc2";
//	if (cudaMalloc((void**)&_c, n * sizeof(int)) != cudaSuccess) std::cout << "Error CudaMalloc3";
//
//	if (cudaMemcpy(_a, a, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Error CudaMemcpy4" << std::endl;
//	if (cudaMemcpy(_b, b, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Error CudaMemcpy5" << std::endl;
//
//	startTime = clock();
//	multKernel << <n / 512 + 1, 512 >> >(_c, _a, _b, n);
//	if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Error CudaDeviceSynchronize7";
//	endTime = clock();
//	std::cout << "Run time on GPU = " << endTime - startTime << std::endl;
//	
//	if (cudaMemcpy(d, _c, n * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "Error  CudaMemcpy6" << std::endl;
//	for (int i = 0; i < n; i++)
//	{
//		if (c[i] != d[i]) std::cout << "Arrays are not equal : " << c[i] << " != " << d[i] << std::endl;
//	}
//	std::system("pause");
//}

int main(int argc, char *argv[])
{
	hash::Hash hash;
	
	std::string str = "Appolon13";
	/*std::cout << "Type string you want to hash" << std::endl;
	std::cin >> str;
	std::cout << "Type length of hash string" << std::endl;
	unsigned int length;
	std::cin >> length;*/
	str = hash.GetHash(str, 5);
	std::cout << str << std::endl;

	system("PAUSE");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
