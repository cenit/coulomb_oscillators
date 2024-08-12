//  Constants
//  Copyright (C) 2024 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

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

#ifndef CONSTANTS_CUDA_H

#include <bit> // std::countl_zero, std::bit_ceil

// Important defines
#ifndef SCAL
#define SCAL float // scalar 
#endif

#ifndef DIM
#define DIM 3 // dimensions
#endif
// note: most functions do not depend on the number of dimensions, but others
// will need to be rewritten for DIM != 2

#if (DIM < 2) || (DIM > 4)
#error "DIM cannot be greater than 4 or smaller than 2"
#endif

int BLOCK_SIZE = 128; // number of threads in a GPU block
int MAX_GRID_SIZE = 10; // number of blocks in a GPU grid
int CACHE_LINE_SIZE = 64; // CPU cache line size (in bytes)
SCAL EPS2 = (SCAL)1.e-18; // softening parameter squared

int CPU_THREADS = 8; // number of concurrent threads in CPU
int fmm_order = 3; // fast multipole method order
SCAL tree_radius = 1;
int tree_L = 0;
int tree_steps = 8;

int h_mlt_max;
__managed__ int m_fmm_order, m_mlt_max;

bool coll = true, b_unsort = true;

SCAL dens_inhom = 1;

template <typename T>
__forceinline__ __device__
inline T myAtomicAdd(T* address, T val)
{
	return atomicAdd(address, val);
}

#if __CUDA_ARCH__ < 600

__device__
inline double myAtomicAdd(double* address, double val)
{
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif // __CUDA_ARCH__ < 600

template <typename T>
__forceinline__ __device__
inline T myAtomicMin(T* address, T val)
{
	return atomicMin(address, val);
}
template <typename T>
__forceinline__ __device__
inline T myAtomicMax(T* address, T val)
{
	return atomicMax(address, val);
}

__device__
inline float myAtomicMin(float* address, float val)
{
	return !signbit(val) ? __int_as_float(atomicMin((int*)address, __float_as_int(val)))
		: __uint_as_float(atomicMax((unsigned*)address, __float_as_uint(val)));
}
__device__
inline float myAtomicMax(float* address, float val)
{
	return !signbit(val) ? __int_as_float(atomicMax((int*)address, __float_as_int(val)))
		: __uint_as_float(atomicMin((unsigned*)address, __float_as_uint(val)));
}

__device__
inline double myAtomicMin(double* address, double val)
{
	return !signbit(val) ? __longlong_as_double(atomicMin((long long*)address, __double_as_longlong(val)))
		: __longlong_as_double(atomicMax((unsigned long long*)address, (unsigned long long)__double_as_longlong(val)));
}
__device__
inline double myAtomicMax(double* address, double val)
{
	return !signbit(val) ? __longlong_as_double(atomicMax((long long*)address, __double_as_longlong(val)))
		: __longlong_as_double(atomicMin((unsigned long long*)address, (unsigned long long)__double_as_longlong(val)));
}

__forceinline__ __host__ __device__
inline uint32_t clz(uint32_t val)
{
// count leading zeros
#ifdef __CUDA_ARCH__
	return __clz(val);
#else
	return std::countl_zero(val);
#endif
}

__forceinline__ __host__ __device__
inline uint32_t bitceil(uint32_t val)
{
// find the smallest power of two that is equal to or greater than `val`
// used in `fmm_p2p3_kdtree` and in `fmm_p2p3_kdtree_coalesced` inside `fmm_cart3_kdtree.cuh`
#ifdef __CUDA_ARCH__
	if (val <= 1)
		return 1;
	return 1 << (32 - __clz(val-1));
#else
	return std::bit_ceil(val);
#endif
}

__forceinline__ __host__ __device__
inline float reciprocal_sqrt(float x)
{
	return rsqrtf(x);
}
__forceinline__ __host__ __device__
inline double reciprocal_sqrt(double x)
{
	return rsqrt(x);
}

// magic unsigned division by multiplication by constant
// algorithm by A. Wesley, Hacker's Delight
struct mu
{
	unsigned M; // magic number and divisor
	int a, s; // "add" indicator and shift amount
};

__host__ __device__
inline mu magicu(unsigned d)
{
// prepare magic number and other constants for unsigned division by 'd'
// must have 1 <= d <= 2^32 - 1
	int p;
	unsigned nc, delta, q1, r1, q2, r2;
	mu magu;

	magu.a = 0; // initialize "add" indicator
	nc = unsigned(-1) - unsigned(-d)%d; // unsigned arithmetic here
	p = 31; // init p
	q1 = 0x80000000/nc; // init q1 = 2^p/nc
	r1 = 0x80000000 - q1*nc; // remainder
	q2 = 0x7FFFFFFF/d; // init q2 = (2^p-1)/d
	r2 = 0x7FFFFFFF - q2*d; // init
	do
	{
		++p;
		if (r1 >= nc-r1)
		{
			q1 = 2*q1 + 1;
			r1 = 2*r1 - nc;
		}
		else
		{
			q1 *= 2;
			r1 *= 2;
		}
		if (r2+1 >= d-r2)
		{
			if (q2 >= 0x7FFFFFFF)
				magu.a = 1;
			q2 = 2*q2 + 1;
			r2 = 2*r2 + 1 - d;
		}
		else
		{
			if (q2 >= 0x80000000)
				magu.a = 1;
			q2 *= 2;
			r2 = 2*r2 + 1;
		}
		delta = d - 1 - r2;
	} while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
	magu.M = q2 + 1;
	magu.s = p - 32;
	return magu; // (magu.a was set above)
}

__forceinline__ __host__ __device__
inline unsigned magicdivu(unsigned x, mu magic)
{
// magic unsigned division
#ifdef __CUDA_ARCH__
	return (__umulhi(x, magic.M) + x*magic.a) >> magic.s;
#else
	return (unsigned(((unsigned long long)x*magic.M) >> 32) + x*magic.a) >> magic.s;
#endif
}

__forceinline__ __host__ __device__
inline unsigned magicremu(unsigned x, unsigned d, mu magic)
{
// magic unsigned remainder
// d is the divisor, the same as the one used to construct the magic number
	return x - d*magicdivu(x, magic);
}

__managed__ mu m_magichi, m_magiclo{};

#endif // !CONSTANTS_CUDA_H














