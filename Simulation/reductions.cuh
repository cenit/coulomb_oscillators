//  CUDA reductions, based on code by NVIDIA Corporation
//  Copyright (C) 2021-24 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

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

#ifndef REDUCTIONS_CUDA_H
#define REDUCTIONS_CUDA_H

#include "kernel.cuh"

#include <cub/cub.cuh> // cub::DeviceReduce::Reduce

#if DIM == 2
#define ONES_VEC VEC{1,1}
#elif DIM == 3
#define ONES_VEC VEC{1,1,1}
#elif DIM == 4
#define ONES_VEC VEC{1,1,1,1}
#endif

__forceinline__ __host__ __device__ 
inline bool isPow2(unsigned int x)
{
    return (x&(x-1))==0;
}

__device__ __host__
inline SCAL rel_diff1(VEC x, VEC ref)
{
	VEC d = x - ref;
	SCAL dist2 = dot(d,d), ref2 = dot(ref,ref) + SCAL(1.e-18);
	return sqrt(max(dist2/ref2, SCAL(0)));
}

__device__ __host__
inline SCAL rel_diff2(VEC x, VEC ref)
{
	VEC d = x - ref;
	VEC s = x + ref;
	SCAL dist2 = dot(d,d), div2 = dot(s,s) + SCAL(1.e-18);
	return 2*sqrt(dist2/div2);
}

struct VecMin
{
	__device__ __host__ __forceinline__
	ALIGNED_VEC operator()(const ALIGNED_VEC& a, const ALIGNED_VEC& b) const
	{
		return aligned_store(fmin(aligned_load(a), aligned_load(b)));
	}
};
struct VecMax
{
	__device__ __host__ __forceinline__
	ALIGNED_VEC operator()(const ALIGNED_VEC& a, const ALIGNED_VEC& b) const
	{
		return aligned_store(fmax(aligned_load(a), aligned_load(b)));
	}
};

void minmaxReduce2(ALIGNED_VEC *minmax, const ALIGNED_VEC *src, unsigned int n, void *& d_tmp_stor, size_t& stor_bytes)
{
	size_t new_stor_bytes = 0;
	gpuErrchk(cub::DeviceReduce::Reduce(nullptr, new_stor_bytes, src, minmax, n, VecMin(), aligned_store(ONES_VEC)*FLT_MAX));
	if (new_stor_bytes > stor_bytes)
	{
		if (stor_bytes > 0)
			gpuErrchk(cudaFree(d_tmp_stor));
		stor_bytes = new_stor_bytes;
		gpuErrchk(cudaMalloc(&d_tmp_stor, stor_bytes));
	}
	gpuErrchk(cub::DeviceReduce::Reduce(d_tmp_stor, stor_bytes, src, minmax, n, VecMin(), aligned_store(ONES_VEC)*FLT_MAX));
	gpuErrchk(cub::DeviceReduce::Reduce(d_tmp_stor, stor_bytes, src, minmax+1, n, VecMax(), aligned_store(-ONES_VEC)*FLT_MAX));
}

template <int blockSize>
__global__
void relerrReduce2_krnl(SCAL *relerr, const ALIGNED_VEC *__restrict__ x, const ALIGNED_VEC *__restrict__ xref, int n)
{
	using BlockReduceT = cub::BlockReduce<SCAL, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	SCAL result;
	if (gid < n)
		result = BlockReduceT(temp_storage).Sum(rel_diff1(aligned_load(x[gid]), aligned_load(xref[gid]))/n);

	if (tid == 0)
		myAtomicAdd(relerr, result);
}

void relerrReduce2(SCAL *relerr, const ALIGNED_VEC *x, const ALIGNED_VEC *xref, unsigned int n)
{
	int nBlocks = (n-1)/1024 + 1;
	cudaMemset(relerr, 0, sizeof(SCAL));
	relerrReduce2_krnl<1024> <<< nBlocks, 1024 >>> (relerr, x, xref, n);
}

template <int blockSize>
__global__
void relerrReduce3Num_krnl(SCAL *relerr, const ALIGNED_VEC *__restrict__ x, const ALIGNED_VEC *__restrict__ xref, int n)
{
	using BlockReduceT = cub::BlockReduce<SCAL, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	SCAL result;
	if (gid < n)
	{
		VEC d = aligned_load(x[gid]) - aligned_load(xref[gid]);
		result = BlockReduceT(temp_storage).Sum(dot(d, d));
	}

	if (tid == 0)
		myAtomicAdd(relerr, result);
}
template <int blockSize>
__global__
void relerrReduce3Den_krnl(SCAL *relerr, const ALIGNED_VEC *xref, int n)
{
	using BlockReduceT = cub::BlockReduce<SCAL, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	SCAL result;
	if (gid < n)
	{
		VEC xgid = aligned_load(xref[gid]);
		result = BlockReduceT(temp_storage).Sum(dot(xgid, xgid));
	}

	if (tid == 0)
		myAtomicAdd(relerr+1, result);
}
__global__ void relerrReduce3Res_krnl(SCAL *relerr)
{
	relerr[0] = sqrt(relerr[0] / relerr[1]);
}

void relerrReduce3(SCAL *relerr, const ALIGNED_VEC *x, const ALIGNED_VEC *xref, unsigned int n)
{
	int nBlocks = (n-1)/1024 + 1;
	cudaMemset(relerr, 0, 2*sizeof(SCAL));
	relerrReduce3Num_krnl<1024> <<< nBlocks, 1024 >>> (relerr, x, xref, n);
	relerrReduce3Den_krnl<1024> <<< nBlocks, 1024 >>> (relerr, xref, n);
	relerrReduce3Res_krnl <<< 1, 1 >>> (relerr);
}

template <int blockSize, bool nIsPow2>
__global__
void minmaxReduce_krnl(ALIGNED_VEC *minmax_, const ALIGNED_VEC *x, unsigned int n)
{
	extern __shared__ ALIGNED_VEC sminmax[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	unsigned int sid = 2*threadIdx.x;

	if (i >= n)
	{
		if (tid == 0)
		{
			minmax_[blockIdx.x*2] = aligned_store(99999999999 * ONES_VEC);
			minmax_[blockIdx.x*2+1] = aligned_store(-99999999999 * ONES_VEC);
		}
		return;
	}

	VEC val_min(aligned_load(x[i])), val_max(aligned_load(x[i]));

	if (nIsPow2 || i + blockSize < n)
	{
		VEC xi = aligned_load(x[i+blockSize]);
		val_min = fmin(val_min, xi);
		val_max = fmax(val_max, xi);
	}

	i += gridSize;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		VEC xi = aligned_load(x[i]);
		val_min = fmin(val_min, xi);
		val_max = fmax(val_max, xi);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
		{
			xi = aligned_load(x[i+blockSize]);
			val_min = fmin(val_min, xi);
			val_max = fmax(val_max, xi);
		}

		i += gridSize;
    }

	// each thread puts its local reduction into shared memory
	sminmax[sid] = aligned_store(val_min);
	sminmax[sid+1] = aligned_store(val_max);
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 1024) && (tid < 512))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 1024])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 1025])));
	}
	__syncthreads();
	
	if ((blockSize >= 512) && (tid < 256))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 512])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 513])));
	}
	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 256])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 257])));
	}
    __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 128])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 129])));
	}
    __syncthreads();

	if ((blockSize >= 64) && (tid <  32))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 64])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 65])));
	}
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((blockSize >= 32) && (tid <  16))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 32])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 33])));
	}
	__syncthreads();

	if ((blockSize >= 16) && (tid <   8))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 16])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 17])));
	}
	__syncthreads();

	if ((blockSize >=  8) && (tid <   4))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 8])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 9])));
	}
	__syncthreads();

	if ((blockSize >=  4) && (tid <   2))
	{
		sminmax[sid] = aligned_store(fmin(aligned_load(sminmax[sid]), aligned_load(sminmax[sid + 4])));
		sminmax[sid+1] = aligned_store(fmax(aligned_load(sminmax[sid+1]), aligned_load(sminmax[sid + 5])));
	}
	__syncthreads();

	if ((blockSize >=  2) && (tid == 0))
	{
		sminmax[0] = aligned_store(fmin(aligned_load(sminmax[0]), aligned_load(sminmax[2])));
		sminmax[1] = aligned_store(fmax(aligned_load(sminmax[1]), aligned_load(sminmax[3])));
	}
	__syncthreads();

    // write result for this block to global mem
	if (tid == 0)
	{
		minmax_[blockIdx.x*2] = sminmax[0];
		minmax_[blockIdx.x*2+1] = sminmax[1];
	}
}

void minmaxReduce(ALIGNED_VEC *minmax, const ALIGNED_VEC *src, unsigned int n, int nBlocksRed = 1)
{
	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(ALIGNED_VEC)*2 : BLOCK_SIZE * sizeof(ALIGNED_VEC)*2;
	if (isPow2(n))
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				minmaxReduce_krnl<1, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 32:
				minmaxReduce_krnl<32, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 64:
				minmaxReduce_krnl<64, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 128:
				minmaxReduce_krnl<128, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 256:
				minmaxReduce_krnl<256, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 512:
				minmaxReduce_krnl<512, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 1024:
				minmaxReduce_krnl<1024, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
	else
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				minmaxReduce_krnl<1, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 32:
				minmaxReduce_krnl<32, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 64:
				minmaxReduce_krnl<64, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 128:
				minmaxReduce_krnl<128, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 256:
				minmaxReduce_krnl<256, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 512:
				minmaxReduce_krnl<512, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 1024:
				minmaxReduce_krnl<1024, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
}

template <int blockSize, bool nIsPow2>
__global__
void relerrReduce_krnl(SCAL *relerr, const ALIGNED_VEC *x, const ALIGNED_VEC *xref, unsigned int n)
{
	extern __shared__ SCAL srelerr[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	if (i >= n)
	{
		if (tid == 0)
			relerr[blockIdx.x] = (SCAL)0;
		return;
	}

	SCAL val{};

	while (i < n)
	{
		val += rel_diff1(aligned_load(x[i]), aligned_load(xref[i]));

		if (nIsPow2 || i + blockSize < n)
			val += rel_diff1(aligned_load(x[i+blockSize]), aligned_load(xref[i+blockSize]));

		i += gridSize;
    }

	srelerr[tid] = val;
	__syncthreads();
	
	if ((blockSize >= 1024) && (tid < 512))
		srelerr[tid] += srelerr[tid + 512];
	__syncthreads();

	if ((blockSize >= 512) && (tid < 256))
		srelerr[tid] += srelerr[tid + 256];
	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
		srelerr[tid] += srelerr[tid + 128];
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
		srelerr[tid] += srelerr[tid +  64];
    __syncthreads();

	if ((blockSize >= 64) && (tid <  32))
		srelerr[tid] += srelerr[tid + 32];
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((blockSize >= 32) && (tid <  16))
		srelerr[tid] += srelerr[tid + 16];
	__syncthreads();

	if ((blockSize >= 16) && (tid <   8))
		srelerr[tid] += srelerr[tid +  8];
	__syncthreads();

	if ((blockSize >=  8) && (tid <   4))
		srelerr[tid] += srelerr[tid +  4];
	__syncthreads();

	if ((blockSize >=  4) && (tid <   2))
		srelerr[tid] += srelerr[tid +  2];
	__syncthreads();

	if ((blockSize >=  2) && (tid == 0))
		srelerr[tid] += srelerr[tid +  1];
	__syncthreads();

    // write result for this block to global mem
	if (tid == 0)
		relerr[blockIdx.x] = srelerr[0];
}

void relerrReduce(SCAL *relerr, const ALIGNED_VEC *x, const ALIGNED_VEC *xref, unsigned int n, int nBlocksRed = 1)
{
	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(SCAL) : BLOCK_SIZE * sizeof(SCAL);
	if (isPow2(n))
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				relerrReduce_krnl<1, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 32:
				relerrReduce_krnl<32, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 64:
				relerrReduce_krnl<64, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 128:
				relerrReduce_krnl<128, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 256:
				relerrReduce_krnl<256, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 512:
				relerrReduce_krnl<512, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 1024:
				relerrReduce_krnl<1024, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
	else
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				relerrReduce_krnl<1, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 32:
				relerrReduce_krnl<32, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 64:
				relerrReduce_krnl<64, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 128:
				relerrReduce_krnl<128, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 256:
				relerrReduce_krnl<256, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 512:
				relerrReduce_krnl<512, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 1024:
				relerrReduce_krnl<1024, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
}

__host__ __device__
inline VEC binarypow(VEC x, int n)
{
// calculates x^n with O(log(n)) multiplications
// assumes n >= 1
	VEC y = ONES_VEC;
	while (n > 1)
	{
		y *= (n & 1) ? x : ONES_VEC;
		x *= x;
		n /= 2;
	}
	return x * y;
}

template <int blockSize, bool nIsPow2>
__global__
void powReduce_krnl(ALIGNED_VEC *power, const ALIGNED_VEC *x, int expo, unsigned int n)
{
// sum the powers of vectors x:
// power = sum_i x_i ^ expo
	extern __shared__ ALIGNED_VEC spow[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	if (i >= n)
	{
		if (tid == 0)
			power[blockIdx.x] = ALIGNED_VEC{};
		return;
	}

	VEC v{};

	while (i < n)
	{
		v += binarypow(aligned_load(x[i]), expo);

		if (nIsPow2 || i + blockSize < n)
			v += binarypow(aligned_load(x[i+blockSize]), expo);

		i += gridSize;
    }

	spow[tid] = aligned_store(v);
	__syncthreads();
	
	if ((blockSize >= 1024) && (tid < 512))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid + 512]));
	__syncthreads();

	if ((blockSize >= 512) && (tid < 256))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid + 256]));
	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid + 128]));
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid +  64]));
    __syncthreads();

	if ((blockSize >= 64) && (tid <  32))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid + 32]));
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((blockSize >= 32) && (tid <  16))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid + 16]));
	__syncthreads();

	if ((blockSize >= 16) && (tid <   8))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid +  8]));
	__syncthreads();

	if ((blockSize >=  8) && (tid <   4))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid +  4]));
	__syncthreads();

	if ((blockSize >=  4) && (tid <   2))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid +  2]));
	__syncthreads();

	if ((blockSize >=  2) && (tid == 0))
		spow[tid] = aligned_store(aligned_load(spow[tid]) + aligned_load(spow[tid +  1]));
	__syncthreads();

    // write result for this block to global mem
	if (tid == 0)
		power[blockIdx.x] = spow[0];
}

void powReduce(ALIGNED_VEC *power, const ALIGNED_VEC *x, int expo, unsigned int n, int nBlocksRed = 1)
{
	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(ALIGNED_VEC) : BLOCK_SIZE * sizeof(ALIGNED_VEC);
	if (isPow2(n))
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				powReduce_krnl<1, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 32:
				powReduce_krnl<32, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 64:
				powReduce_krnl<64, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 128:
				powReduce_krnl<128, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 256:
				powReduce_krnl<256, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 512:
				powReduce_krnl<512, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 1024:
				powReduce_krnl<1024, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
	else
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				powReduce_krnl<1, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 32:
				powReduce_krnl<32, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 64:
				powReduce_krnl<64, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 128:
				powReduce_krnl<128, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 256:
				powReduce_krnl<256, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 512:
				powReduce_krnl<512, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 1024:
				powReduce_krnl<1024, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
}

#endif // !REDUCTIONS_CUDA_H