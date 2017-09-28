
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>

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


//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}

//int main(int argc, char *  argv[])
//{
//	int n = 10000000;
//	int *a = new int[n];
//	int *b = new int[n];
//	int *c = new int[n];
//	int t = GetTickCount();
//	for (int i = 0; i<n; i++)
//	{
//		a[i] = rand();
//		b[i] = rand();
//	}
//	for (int i = 0; i<n; i++)
//	{
//		c[i] = a[i] * b[i];
//		// printf("c = %i,	a = %i,	b = %i\n", c[i], a[i], b[i]); 
//	}
//
//	std::cout << "Time = " << t << std::endl;
//	system("pause");
//}

int main(int argc, char * argv[])
{
	int t = GetTickCount();
	int  n = 10000000;
	int *a = new int[n];
	int *b = new int[n];
	int *c = new int[n];
	int *d = new int[n];
	int *_a;
	int *_b;
	int *_c;

	for (int i = 0; i<n; i++)
	{
		a[i] = rand();
		b[i] = rand();
	}
	int startTime = clock();
	for (int i = 0; i < n; i++)
	{
		c[i] = a[i] * b[((i * 1) % n + n )%n];
	}
	int endTime = clock();
	std::cout << "Run time on CPU = " << endTime - startTime << std::endl;

	if (cudaMalloc((void**)&_a, n * sizeof(int)) != cudaSuccess) std::cout << "Error CudaMalloc1";
	if (cudaMalloc((void**)&_b, n * sizeof(int)) != cudaSuccess) std::cout << "Error CudaMalloc2";
	if (cudaMalloc((void**)&_c, n * sizeof(int)) != cudaSuccess) std::cout << "Error CudaMalloc3";

	if (cudaMemcpy(_a, a, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Error CudaMemcpy4" << std::endl;
	if (cudaMemcpy(_b, b, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Error CudaMemcpy5" << std::endl;

	startTime = clock();
	multKernel << <n / 512 + 1, 512 >> >(_c, _a, _b, n);
	if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Error CudaDeviceSynchronize7";
	endTime = clock();
	std::cout << "Run time on GPU = " << endTime - startTime << std::endl;
	
	if (cudaMemcpy(d, _c, n * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "Error  CudaMemcpy6" << std::endl;
	for (int i = 0; i < n; i++)
	{
		if (c[i] != d[i]) std::cout << "Arrays are not equal : " << c[i] << " != " << d[i] << std::endl;
	}
	std::system("pause");
}


//int main(int argc, char *  argv[]) {
//	int		deviceCount;
//	cudaDeviceProp	devProp;
//	cudaGetDeviceCount(&deviceCount);
//	printf("Found %d devices\n", deviceCount);
//	for (int device = 0; device < deviceCount; device++) {
//		cudaGetDeviceProperties(&devProp, device);
//		printf("Device %d\n", device);
//		printf("Compute capability     : %d.%d\n", devProp.major, devProp.minor);
//		printf("Name                   : %s\n", devProp.name);
//		printf("Total Global Memory    : %d\n", devProp.totalGlobalMem);
//		printf("Shared memory per block: %d\n", devProp.sharedMemPerBlock);
//		printf("Registers per block    : %d\n", devProp.regsPerBlock);
//		printf("Warp size              : %d\n", devProp.warpSize);
//		printf("Max threads per block  : %d\n", devProp.maxThreadsPerBlock);
//		printf("Total constant memory  : %d\n", devProp.totalConstMem);
//	}
//	return 0;
//}

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
