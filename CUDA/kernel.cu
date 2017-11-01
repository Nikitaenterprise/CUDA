
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>
#include <string>
#include <random>

#include "Hash.h"

#define BLOCK_SIZE 512

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
__global__ void multMatrix(float *c, const float *a, const float *b, int m)
{
	int   bx = blockIdx.x;
	int   by = blockIdx.y;
	int   tx = threadIdx.x;
	int   ty = threadIdx.y;
	float sum = 0.0f;
	int   ia = m * 16 * by + m * ty;
	int   ib = 16 * bx + tx;
	int   ic = m * 16 * by + 16 * bx;

	for (int k = 0; k < m; k++)
		sum += a[ia + k] * b[ib + k*m];

	c[ic + m * ty + tx] = sum;
}

__global__ void reduce1(int * inData, int * outData)
{
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i]; 	// load into shared memory
	__syncthreads();
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) 	// heavy branching !!!
			data[tid] += data[tid + s];
		__syncthreads();
	}
	if (tid == 0) 		// write result of block reduction
		outData[blockIdx.x] = data[0];
}

__global__ void reduce2(int * inData, int * outData)
{
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	data[tid] = inData[i]; 	// load into shared memory
	__syncthreads();
	for (int s = 1; s < blockDim.x; s <<= 1)
	{
		int index = 2 * s * tid;	// better replace with >>
		if (index < blockDim.x)
			data[index] += data[index + s];
		__syncthreads();
	}
	if (tid == 0) 		// write result of block reduction
		outData[blockIdx.x] = data[0];
}

__global__ void reduce3(int * inData, int * outData)
{
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i];
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			data[tid] += data[tid + s];
		__syncthreads();
	}
	if (tid == 0)
		outData[blockIdx.x] = data[0];
}

__global__ void reduce4(int * inData, int * outData)
{
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i] + inData[i + blockDim.x]; // sum
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			data[tid] += data[tid + s];
		__syncthreads();
	}
	if (tid == 0)
		outData[blockIdx.x] = data[0];
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
//	multKernel <<<n / 512 + 1, 512 >>> (_c, _a, _b, n);
//	if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Error CudaDeviceSynchronize7";
//	endTime = clock();
//	std::cout << "Run time on GPU = " << endTime - startTime << std::endl;
//	
//	if (cudaMemcpy(d, _c, n * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "Error  CudaMemcpy6" << std::endl;
//	for (int i = 0; i < n; i++)
//	{
//		if (c[i] != d[i]) std::cout << "Arrays are not equal : " << c[i] << " != " << d[i] << std::endl;
//	}
//
//	float *X, *Y, *result, *result1;
//	int m = 512;
//	X = new float[m*m];
//	Y = new float[m*m];
//	result = new float[m*m];
//	result1 = new float[m*m];
//	float *_X, *_Y, *_result;
//	srand(235806);
//	for (int i = 0; i < m; i++)
//	{
//		for (int j = 0; j < m; j++)
//		{
//			X[m*i+j] = rand()%500;
//			Y[m*i+j] = rand()%500;
//		}
//	}
//	startTime = clock();
//	for (int i = 0; i < m; i++)
//	{
//		for (int j = 0; j < m; j++)
//		{
//			result[m*i + j] = 0;
//			for (int k = 0; k < m; k++)
//			{
//				result[m*i + j] += X[i*m + k] * Y[k*m + j];
//			}
//		}
//	}
//	endTime = clock();
//	std::cout << "Calculating on CPU = " << endTime - startTime << std::endl;
//	if (cudaMalloc((void**)&_X, m*m * sizeof(float)) != cudaSuccess) std::cout << "Error in first malloc" << std::endl;
//	if (cudaMalloc((void**)&_Y, m*m * sizeof(float)) != cudaSuccess) std::cout << "Error in second malloc" << std::endl;
//	if (cudaMalloc((void**)&_result, m*m * sizeof(float)) != cudaSuccess) std::cout << "Error in third malloc" << std::endl;
//	if (cudaMemcpy(_X, X, m*m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Error in first copy" << std::endl;
//	if (cudaMemcpy(_Y, Y, m*m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Error in second copy" << std::endl;
//	dim3 threads(16, 16);
//	dim3 blocks(m / threads.x, m / threads.y);
//	startTime = clock();
//	multMatrix << <blocks, threads >> > (_result, _X, _Y, m);
//	cudaThreadSynchronize();
//	if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Error in syncronization" << std::endl;
//	endTime = clock();
//	std::cout << "Calculating on GPU = " << endTime - startTime << std::endl;
//	if (cudaMemcpy(result1, _result, m*m * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "Error in third copy" << std::endl;
//	for (int i = 0; i < m; i++)
//	{
//		for (int j = 0; j < m; j++)
//		{
//			if (abs(result[m*i + j] - result1[m*i + j]) > 0.00001)
//			{
//				std::cout << "n*i+j = " << m*i + j << "\t" << "result[n*i + j] = " << result[m*i + j] << "\t" << "result1[n*i + j] = " << result1[m*i + j] << std::endl;
//			}
//		}
//	}
//
//	delete[] a, b, c, d, _a, _b, _c;
//	
//	a = new int[n];
//	b = new int[n];
//
//	startTime = clock();
//	for (int i = 0; i < n; i++)
//	{
//		a[i] = rand();
//	}
//	long int A = 0;
//	for (int i = 0; i < n; i++)
//	{
//		A += a[i];
//	}
//	endTime = clock();
//	std::cout << "Calculating on CPU = " << endTime - startTime << std::endl;
//	
//	for (int i = 0; i < 4; i++)
//	{
//		long int _A = 0;
//		startTime = clock();
//		if (cudaMalloc((void**)&_a, n * sizeof(int)) != cudaSuccess) std::cout << "Bad alloc in run " << i << std::endl;
//		if (cudaMalloc((void**)&_b, n * sizeof(int)) != cudaSuccess) std::cout << "Bad alloc in run " << i <<std::endl;
//		if (cudaMemcpy(_a, a, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) std::cout << "Bad copy to device in run " << i <<std::endl;
//		switch (i)
//		{
//		case 0:
//			reduce1 << <n / 512, 512 >> > (_a, _b);
//			break;
//		case 1:
//			reduce2 << <n / 512, 512 >> > (_a, _b);
//			break;
//		case 2:
//			reduce3 << <n / 512, 512 >> > (_a, _b);
//			break;
//		case 3:
//			reduce4 << <n / 512, 512 >> > (_a, _b);
//			break;
//		}
//		cudaThreadSynchronize();
//		if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Error in syncronization in run " << i <<std::endl;
//		if (cudaMemcpy(b, _b, n * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "Bad copy to host in run " << i << std::endl;
//		endTime = clock();
//		std::cout << "Calculating on GPU = " << endTime - startTime << " in run " << i << std::endl;
//		for (int j = 0; j < n / 512; j++) { _A += b[j]; }
//		result = { 0 };
//		std::cout << "A = " << A << " _A = " << _A << " A - _A = " << A-_A << std::endl;
//	}
//
//	std::system("pause");
//}

int main(int argc, char *argv[])
{
	hash::Hash hash;
	hash::Hash firstHash;
	std::string password = "Pa8D";
	std::cout << "password = " << password << std::endl; 
	std::string str = firstHash.GetHash(password, password.length());
	std::cout << "str = " << str << std::endl;
	std::string ans;
	std::string tr(str);
	std::cout << "tr = " << tr << std::endl;
	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_int_distribution<> distribution1(65, 90); //A-Z
	std::uniform_int_distribution<> distribution2(97, 122); //a-z
	std::uniform_int_distribution<> distribution3(48, 57); //0-9
	std::vector<std::string> coincidences;
	//AVictor2007@yandex.ru
	std::cout << "str.length() = " << str.length() << std::endl;
	int innerCounter = 0, outerCounter = 0;
	while (tr != password)
	{
		while (ans != str)
		{
			for (int i = 0; i < str.length(); i++)
			{
				int random = rand() % 3;
				switch (random)
				{
				case 0:
					tr[i] = distribution1(generator);
					break;
				case 1:
					tr[i] = distribution2(generator);
					break;
				case 2:
					tr[i] = distribution3(generator);
					break;
				}
			}
			ans = hash.GetHash(tr, str.length());
			coincidences.push_back(ans);
			hash.Clear();
			innerCounter++;
		}
		/*std::cout << "We are looking for this string : " << password << std::endl;
		std::cout << "\tWith this hash : " << ans << std::endl;
		std::cout << "We`ve got this string : " << tr << std::endl;
		std::cout << "\tWith this hash : " << str << std::endl;	*/
		ans.empty();
		tr.empty();
		outerCounter++;
	}
	std::cout << "Total number of steps = " << innerCounter << std::endl;
	std::cout << "Nuber of coincidences = " << outerCounter << std::endl;
	std::cout << "Chance for coincidences = " << static_cast <long double> (outerCounter) / static_cast <long double> (innerCounter) * 100 << "%" << std::endl;
	std::cout << "Do you wanna see all coincidences?\n Y/N" << std::endl;
	char *des;
	switch (*des)
	{
	case'Y':
		for (int i = 0; i < coincidences.size(); i++) std::cout << coincidences[i] << std::endl;
		break;
	case'N':
		break;
	}
	hash::Hash checkHash;
	std::string check;
	for (;;)
	{
		std::cout << "Type string : ";
		std::cin >> check;
		std::cout << "your hash is : " << checkHash.GetHash(check, check.length()) << std::endl;
		check.clear();
		checkHash.Clear();
	}
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
