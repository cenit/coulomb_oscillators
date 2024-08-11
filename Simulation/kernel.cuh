//  Some basic code for CPU and GPU
//  Copyright (C) 2021 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef KERNEL_CUDA_H
#define KERNEL_CUDA_H

#include "constants.cuh"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

//#ifndef __CUDACC_RTC__
//#define __CUDACC_RTC__
//#endif

#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "helper_math.h"

// IMPORTANT: set WDDM TDR Delay to 60 sec (default is 2) from NVIDIA Nsight Monitor Options
// otherwise, GPU functions (kernels) that take more than 2 seconds will fail

//#include <device_functions.h>
//#include <cuda_fp16.h> // half precision floating point

#include <cmath>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#define VEC_PASTE(U, n) U##n
#define VEC_T(U, n) VEC_PASTE(U, n) /* vector type */
#define VEC VEC_T(SCAL, DIM)
#define IVEC VEC_T(int, DIM)

#define ALIGNED 0

#if DIM == 3 && ALIGNED == 1
#define ALIGNED_VEC VEC_T(SCAL, 3)
#else
#define ALIGNED_VEC VEC
#endif

#include "mymath.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPUassert: " << cudaGetErrorString(code) << ' ' << file << ' ' << line << std::endl;
		if (abort)
		{
			cudaDeviceReset();
			exit(code);
		}
	}
}

inline __device__ __host__ VEC aligned_load(const ALIGNED_VEC& v)
{
#if DIM == 3 && ALIGNED == 1
	ALIGNED_VEC t(v);
	return {t.x, t.y, t.z};
#else
	return v;
#endif
}

inline __device__ __host__ ALIGNED_VEC aligned_store(const VEC& v)
{
#if DIM == 3 && ALIGNED == 1
	return {v.x, v.y, v.z};
#else
	return v;
#endif
}

struct ParticleSystem { ALIGNED_VEC *__restrict__ pos, *__restrict__ vel, *__restrict__ acc; };

std::ostream& operator<<(std::ostream& os, VEC_T(SCAL, 2) a)
{
	os << a.x << ", " << a.y;
	return os;
}
std::ostream& operator<<(std::ostream& os, VEC_T(SCAL, 3) a)
{
	os << a.x << ", " << a.y << ", " << a.z;
	return os;
}
std::ostream& operator<<(std::ostream& os, VEC_T(SCAL, 4) a)
{
	os << a.x << ", " << a.y << ", " << a.z << ", " << a.w;
	return os;
}

__global__ void step_krnl(ALIGNED_VEC *__restrict__ b, const ALIGNED_VEC *__restrict__ a, SCAL ds, int n)
// multiply-addition kernel
// b += a * ds
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < n)
	{
		VEC t = aligned_load(a[i]);
		VEC s = aligned_load(b[i]);

		b[i] = aligned_store(fma(ds, t, s));
		i += blockDim.x * gridDim.x;
	}
}

void step(ALIGNED_VEC *b, const ALIGNED_VEC *a, SCAL ds, int n)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	step_krnl <<< nBlocks, BLOCK_SIZE >>> (b, a, ds, n);
}

void step_cpu(ALIGNED_VEC *__restrict__ b, const ALIGNED_VEC *__restrict__ a, SCAL ds, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				b[j] += a[j] * ds;
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void add_elastic_krnl(const ALIGNED_VEC *__restrict__ p, ALIGNED_VEC *__restrict__ a, int n,
                                 const ALIGNED_VEC *__restrict__ param)
// elastic force computation kernel with elastic costants defined in "param" pointer
{
	VEC k = -aligned_load(param[0]);
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		VEC t = aligned_load(p[i]);
		VEC s = aligned_load(a[i]);

		a[i] = aligned_store(fma(k, t, s));
	}
}

__global__ void add_elastic_krnl(const ALIGNED_VEC *__restrict__ p, ALIGNED_VEC *__restrict__ a, int n)
// elastic force computation kernel
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		a[i] = aligned_store(aligned_load(a[i]) - aligned_load(p[i]));
	}
}

void add_elastic(ALIGNED_VEC *p, ALIGNED_VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (param != nullptr)
		add_elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, (const ALIGNED_VEC*)param);
	else
		add_elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n);
}

void add_elastic_cpu(ALIGNED_VEC *__restrict__ p, ALIGNED_VEC *__restrict__ a, int n, const SCAL* param)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	if (param != nullptr)
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				const ALIGNED_VEC *k = (const ALIGNED_VEC*)param;
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] -= p[j] * k[0];
			});
	else
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] -= p[j];
			});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void elastic_krnl(const ALIGNED_VEC *__restrict__ p, ALIGNED_VEC *__restrict__ a, int n,
                             const ALIGNED_VEC *__restrict__ param)
// elastic force computation kernel with elastic costants defined in "param" pointer
{
	VEC k = -aligned_load(param[0]);
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		a[i] = aligned_store(k*aligned_load(p[i]));
	}
}

__global__ void elastic_krnl(const ALIGNED_VEC *__restrict__ p, ALIGNED_VEC *__restrict__ a, int n)
// elastic force computation kernel
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		a[i] = aligned_store(-aligned_load(p[i]));
	}
}

void elastic(ALIGNED_VEC *p, ALIGNED_VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (param != nullptr)
		elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, (const ALIGNED_VEC*)param);
	else
		elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n);
}

void elastic_cpu(ALIGNED_VEC *__restrict__ p, ALIGNED_VEC *__restrict__ a, int n, const SCAL* param)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	if (param != nullptr)
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				const ALIGNED_VEC *k = (const ALIGNED_VEC*)param;
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] = p[j] * -k[0];
			});
	else
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] = -p[j];
			});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

template<typename T>
__global__ void gather_krnl(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
// dst array is built from src array through a permutation map pointer
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[i] = src[map[i]];
	}
}

template<typename T>
void gather_cpu(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				dst[j] = src[map[j]];
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

template<typename T>
__global__ void gather_inverse_krnl(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
// dst array is built from src array through the inverse permutation of map pointer
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[map[i]] = src[i];
	}
}

template<typename T>
void gather_inverse_cpu(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				dst[map[j]] = src[j];
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

template<typename T>
__global__ void copy_krnl(T *__restrict__ dst, const T *__restrict__ src, int n)
// copy content from src to dst
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[i] = src[i];
	}
}

template<typename T>
void copy_gpu(T *dst, const T *src, int n)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (dst, src, n);
}

template<typename T>
void copy_cpu(T *__restrict__ dst, const T *__restrict__ src, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				dst[j] = src[j];
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

#endif // !KERNEL_CUDA_H