//  N-body coulomb oscillators
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

/*

Compilation:
nvcc main3.cu -o nbco3 -O3 -ftz=true -arch=sm_75 --expt-relaxed-constexpr <std=c++20>
nvcc main3.cu -o nbco3 -O3 -use_fast_math -arch=sm_75 --expt-relaxed-constexpr <std=c++20>

Use `--std c++20` for Windows (Visual Studio 2022) or `-std=c++20` for Linux (GCC 10-12)
`-ftz=true` (flushing to zero) will ignore denormal numbers and treat them as zero

For profiling:
nvprof nbco3 -test
ncu -f -o profile nbco3 -test

ncu --section SchedulerStats --section WarpStateStats --section SourceCounters -o profile nbco3 -test -n 100000 -r 2 -p 4
ncu --set full -o profile nbco3 -test -n 100000 -r 2 -p 4

For debugging:
nvcc -lineinfo main3.cu -o nbco3 --expt-relaxed-constexpr <std=c++20>
compute-sanitizer --print-limit 1 nbco3 -test

*/

#include "direct.cuh"
#include "integrator.cuh"
#include "fmm_cart3_kdtree.cuh"
#include "reductions.cuh"

#include <fstream> // std::ifstream, std::ofstream
#include <limits> // std::numeric_limits
#include <random> // std::mt19937_64, std::uniform_real_distribution, std::normal_distribution
#include <chrono> // std::chrono::steady_clock, std::chrono::duration_cast
#include <vector> // std::vector

using namespace std::chrono;

void coulombOscillatorDirect(ALIGNED_VEC *p, ALIGNED_VEC *a, int n, const SCAL* param)
{
	direct3(p, a, n, param);
	add_elastic(p, a, n, param+4); // shift pointer
}

void coulombOscillatorDirect_cpu(ALIGNED_VEC *p, ALIGNED_VEC *a, int n, const SCAL* param)
{
	direct3_cpu(p, a, n, param);
	add_elastic_cpu(p, a, n, param+4); // shift pointer
}

void coulombOscillatorFMMKD3(ALIGNED_VEC *p, ALIGNED_VEC *a, int n, const SCAL* param)
{
	fmm_cart3_kdtree(p, a, n, param);
	add_elastic(p, a, n, param+4); // shift pointer
}

void coulombOscillatorFMMKD3_cpu(ALIGNED_VEC *p, ALIGNED_VEC *a, int n, const SCAL* param)
{
	fmm_cart3_kdtree_cpu(p, a, n, param);
	add_elastic_cpu(p, a, n, param+4); // shift pointer
}

void centerDist(ALIGNED_VEC *data, int n)
{
// center the sampled distribution
	VEC d{};
	for (int i = 0; i < n; ++i)
		d += aligned_load(data[i]);
	d /= (SCAL)n;
	for (int i = 0; i < n; ++i)
		data[i] = aligned_store(aligned_load(data[i]) - d);
}

void adjustRMS(ALIGNED_VEC *data, int n, VEC adj)
{
// adjust the RMS of the sampled distribution such that it's equal to adj
	VEC d{}, datai;
	for (int i = 0; i < n; ++i)
	{
		datai = aligned_load(data[i]);
		d += datai*datai;
	}
	d /= (SCAL)n;
	d = sqrt(d);
	for (int i = 0; i < n; ++i)
		data[i] = aligned_store(aligned_load(data[i]) * adj / d);
}

void initU(ALIGNED_VEC *data, int n, VEC a, VEC b, std::mt19937_64 &gen)
{
// Uniform distribution over a rectangular cuboid
// velocities remain uninitialized
// a_j < x_ij < b_j for each particle i and coordinate j = 1,2,3
	int nBodies = n/2;
	std::uniform_real_distribution<SCAL> distx(a.x, b.x);
	std::uniform_real_distribution<SCAL> disty(a.y, b.y);
	std::uniform_real_distribution<SCAL> distz(a.z, b.z);

	VEC d;
	for (int i = 0; i < nBodies; ++i)
	{
		d.x = distx(gen);
		d.y = disty(gen);
		d.z = distz(gen);
		data[i] = aligned_store(d);
	}

	centerDist(data, nBodies);
}

void initGA(ALIGNED_VEC *data, int n, VEC x, VEC u, std::mt19937_64 &gen)
{
// Gaussian distribution
// x is the position std.dev.
// u is the velocity std.dev.
	std::normal_distribution<SCAL> dist((SCAL)0, (SCAL)1); // Marsaglia method?
									// mean = 0, std.dev. = 1
	int nBodies = n/2;
	VEC d;
	for (int i = 0; i < nBodies; ++i)
	{
		d.x = dist(gen)*x.x;
		d.y = dist(gen)*x.y;
		d.z = dist(gen)*x.z;
		data[i] = aligned_store(d);
	}
	for (int i = nBodies; i < 2*nBodies; ++i)
	{
		d.x = dist(gen)*u.x;
		d.y = dist(gen)*u.y;
		d.z = dist(gen)*u.z;
		data[i] = aligned_store(d);
	}

	centerDist(data, nBodies);
	adjustRMS(data, nBodies, x);

	data += nBodies;

	centerDist(data, nBodies);
	adjustRMS(data, nBodies, u);
}

SCAL test_accuracy(void(*test)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), void(*ref)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*),
				   SCAL *d_buf, int n, const SCAL* param, bool b_update = false)
{
// test the accuracy of "test" function with respect to the reference "ref" function
// print the mean relative error on console window
	static int n_max = 0;
	static SCAL *d_relerr = nullptr;
	static SCAL relerr;
	static ALIGNED_VEC *d_tmp = nullptr;
	if (n > n_max)
	{
		if (n_max > 0)
		{
			gpuErrchk(cudaFree(d_tmp));
			gpuErrchk(cudaFree(d_relerr));
		}
		gpuErrchk(cudaMalloc((void**)&d_tmp, sizeof(ALIGNED_VEC)*n));
		gpuErrchk(cudaMalloc((void**)&d_relerr, sizeof(SCAL)));
	}
	if (::b_unsort)
	{
		if (b_update || n != n_max)
		{
			compute_force(ref, d_buf, n, param);
			copy_gpu(d_tmp, (ALIGNED_VEC*)d_buf + 2 * n, n);
		}
		compute_force(test, d_buf, n, param);
		relerrReduce2(d_relerr, (ALIGNED_VEC*)d_buf + 2 * n, d_tmp, n);
	}
	else
	{
		compute_force(test, d_buf, n, param);
		copy_gpu(d_tmp, (ALIGNED_VEC*)d_buf + 2 * n, n);
		compute_force(ref, d_buf, n, param);
		relerrReduce2(d_relerr, d_tmp, (ALIGNED_VEC*)d_buf + 2 * n, n);
	}

	gpuErrchk(cudaMemcpy(&relerr, d_relerr, sizeof(SCAL), cudaMemcpyDeviceToHost));

	if (n > n_max)
		n_max = n;

	return relerr;
}

SCAL test_accuracy_cpu(void(*test)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), void(*ref)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*),
					   SCAL *buf, int n, const SCAL* param)
{
// test the accuracy of "test" function with respect to the reference "ref" function
// print the mean relative error on console window
// both functions must run on CPU
	static int n_max = 0;
	std::vector<SCAL> relerr(CPU_THREADS);
	static ALIGNED_VEC *tmp = nullptr;
	if (n > n_max)
	{
		if (n_max > 0)
			delete[] tmp;
		tmp = new ALIGNED_VEC[n];
	}
	compute_force(test, buf, n, param);
	copy_cpu(tmp, (ALIGNED_VEC*)buf + 2 * n, n);
	compute_force(ref, buf, n, param);

	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=, &relerr] {
			SCAL err(0);
			ALIGNED_VEC *bufacc = (ALIGNED_VEC*)buf + 2*n;
			for (int j = niter*i; j < std::min(n, niter*(i+1)); ++j)
				err += rel_diff1(aligned_load(tmp[j]), aligned_load(bufacc[j]));
			relerr[i] = err;
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();

	for (int i = 1; i < CPU_THREADS; ++i)
		relerr[0] += relerr[i];

	if (n > n_max)
		n_max = n;

	return relerr[0] / (SCAL)n;
}

int main(const int argc, const char** argv)
{
	std::cout << "N-body coulomb oscillators, Copyright (C) 2021-24 Alessandro Lo Cuoco\n\n"
				 "Type 'nbco -h' for a brief documentation.\n\n";
	
	int nBodies = 30001;
	SCAL dt = (SCAL)5.e-4; // time step
	int nIters = 30001;  // simulation iterations
	int nSteps = 200;  // number of steps for every "snapshot" saved to file
	std::string strout("out"), strin;
	bool in = false, cpu = false, test = false, test2 = false, b_accuracy = false;
	SCAL accuracy = 0.001;
	
	auto symp_integ = leapfrog;
	
	SCAL xi(2.e-6);
	VEC omega0{1.095, 1.0, 1.0}, x, u;

	// GA default parameters
	x = VEC{0.003, 0.001, 0.01};
	u = omega0 * x;
	
	for (int i = 1; i < argc; ++i)
	{
		if (argv[i][0] == '-')
		{
			if ((argv[i][1] == 'h' && argv[i][2] == '\0') ||
				(argv[i][1] == 'h' && argv[i][2] == 'e' && argv[i][3] == 'l' && argv[i][4] == 'p' && argv[i][5] == '\0'))
			{
				std::cout << "This program comes with ABSOLUTELY NO WARRANTY.\n"
							 "This is free software, and you can redistribute it under certain conditions.\n"
							 "See the LICENSE file for more details.\n\n"
							 "Usage: nbco [options] [input]\n\n"
							 "  [input] is the path to a file that contains a state of the system. The format\n"
							 "  is that of a binary file that contains the positions of all particles and\n"
							 "  then their respective velocities in the same order. This state is used for\n"
							 "  initialising the system to be simulated. If not specified, the system will be\n"
							 "  initalised by sampling from a gaussian distribution.\n\n"
							 "Other options:\n"
							 "  -h or -help       Display this documentation.\n"
							 "  -o <output>       Specify the folder where the output will be written.\n"
							 "                    Default is './out'. This folder must already exist. The\n"
							 "                    format of output files is the same as the input file format\n"
							 "                    described before.\n"
							 "  -n <npart>        Number of particles to simulate. Default is 30001. Will be\n"
							 "                    ignored if [input] is specified.\n"
							 "  -ds <v>           Time step. Default is 5e-4.\n"
							 "  -iters <n>        Number of total simulation iterations. Default is 30000.\n"
							 "  -steps <n>        Number of steps to simulate before saving to file. Default\n"
							 "                    is 200.\n"
							 "  -integ <name>     Set the symplectic integrator to be used instead of the\n"
							 "                    default one (2nd order). <name> must be chosen from the\n"
							 "                    list {eu, fr, pefrl}, respectively the semi-implicit Euler\n"
							 "                    (1st order), Forest-Ruth (4th order) and PEFRL (4th order).\n"
							 "  -p <order>        FMM expansion order. Default is 2. Will be ignored if\n"
							 "                    -accuracy is specified.\n"
							 "  -r <radius>       Interaction radius. Must be 1 or greater. Default is 1.\n"
							 "                    Will be ignored if -accuracy is specified.\n"
							 "  -eps <v>          Smoothing factor. Must be greater than 0. Default is 1e-9.\n"
							 "  -i <v>            A factor so that max FMM level is round(log(n*i/p^2)).\n"
							 "                    Default is 1. Will be ignored if -maxlevel is specified.\n"
							 "  -maxlevel <n>     Set the maximum level for kd-tree construction. By default,\n"
							 "                    it is calculated depending on the number of particles and\n"
							 "                    expansion order.\n"
							 "  -ncoll            P2P pass will not be calculated. Will be ignored\n"
							 "                    if -accuracy is specified.\n"
							 "  -accuracy <v>     Set minimum accuracy for the simulation. The program will\n"
							 "                    search optimized parameters that satisfy this condition.\n"
							 "                    This ignores -p, -r, and -ncoll options.\n"
							 "  -cpu              Use CPU with multithreading (default is GPU).\n"
							 "  -cpu-threads <n>  Number of CPU threads. Must be 1 or greater (default is 8).\n"
							 "                    Implies -cpu.\n"
							 "  -cacheline <n>    CPU cache line size in bytes. Defaut is 64. Will be ignored\n"
							 "                    if -cpu is NOT specified.\n"
							 "  -test             Show relative errors and execution times of a single\n"
							 "                    iteration instead of doing the simulation.\n"
							 "  -test2            Another test that will be documented soon.\n"
							 "  -x <vx> <vy> <vz> Set the std.dev. of positions. Will be ignored if [input]\n"
							 "                    is specified.\n"
							 "  -u <vx> <vy> <vz> Set the std.dev. of velocities. Will be ignored if [input]\n"
							 "                    is specified.\n";
				
				return 0;
			}
			else if (argv[i][1] == 'o' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-o'\n";
					return -1;
				}
				strout = argv[i+1];
				++i;
			}
			else if (argv[i][1] == 'n' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-n'\n";
					return -1;
				}
				nBodies = atoi(argv[i+1]);
				if (nBodies <= 0)
				{
					std::cerr << "Error: invalid argument to '-n': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'd' && argv[i][2] == 's' && argv[i][3] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-ds'\n";
					return -1;
				}
				dt = atof(argv[i+1]);
				if (dt <= 0)
				{
					std::cerr << "Error: invalid argument to '-ds': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'i' && argv[i][2] == 't' && argv[i][3] == 'e' && argv[i][4] == 'r' && argv[i][5] == 's'
				  && argv[i][6] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-iters'\n";
					return -1;
				}
				nIters = atoi(argv[i+1])+1;
				if (nIters <= 0)
				{
					std::cerr << "Error: invalid argument to '-iters': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 's' && argv[i][2] == 't' && argv[i][3] == 'e' && argv[i][4] == 'p' && argv[i][5] == 's'
				  && argv[i][6] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-steps'\n";
					return -1;
				}
				nSteps = atoi(argv[i+1]);
				if (nSteps <= 0)
				{
					std::cerr << "Error: invalid argument to '-steps': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'i' && argv[i][2] == 'n' && argv[i][3] == 't' && argv[i][4] == 'e' && argv[i][5] == 'g'
				  && argv[i][6] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-integ'\n";
					return -1;
				}
				if (argv[i+1][1] == 'e' && argv[i+1][2] == 'u' && argv[i+1][3] == '\0')
					symp_integ = symplectic_euler;
				else if (argv[i+1][1] == 'f' && argv[i+1][2] == 'r' && argv[i+1][3] == '\0')
					symp_integ = forestruth;
				else if (argv[i+1][1] == 'p' && argv[i+1][2] == 'e' && argv[i+1][3] == 'f' && argv[i+1][4] == 'r' && argv[i+1][5] == 'l'
					  && argv[i+1][6] == '\0')
					symp_integ = pefrl;
				else
				{
					std::cerr << "Error: invalid argument to '-integ': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'p' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-p'\n";
					return -1;
				}
				::fmm_order = atoi(argv[i+1]);
				if (fmm_order <= 0)
				{
					std::cerr << "Error: invalid argument to '-p': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'r' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-r'\n";
					return -1;
				}
				::tree_radius = atof(argv[i+1]);
				if (tree_radius <= 0)
				{
					std::cerr << "Error: invalid argument to '-r': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'e' && argv[i][2] == 'p' && argv[i][3] == 's' && argv[i][4] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-eps'\n";
					return -1;
				}
				::EPS2 = atof(argv[i+1]);
				if (::EPS2 <= 0)
				{
					std::cerr << "Error: invalid argument to '-eps': " << argv[i+1] << '\n';
					return -1;
				}
				::EPS2 *= ::EPS2;
				if (::EPS2 == 0) // underflow
				{
					std::cerr << "Error: too small argument to '-eps': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'i' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-i'\n";
					return -1;
				}
				::dens_inhom = atof(argv[i+1]);
				if (::dens_inhom <= 0)
				{
					std::cerr << "Error: invalid argument to '-i': " << argv[i+1] << " (should be greater than 0)\n";
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'm' && argv[i][2] == 'a' && argv[i][3] == 'x' && argv[i][4] == 'l' && argv[i][5] == 'e'
			      && argv[i][6] == 'v' && argv[i][7] == 'e' && argv[i][8] == 'l' && argv[i][9] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-maxlevel'\n";
					return -1;
				}
				::tree_L = atoi(argv[i+1]);
				if (tree_L <= 0)
				{
					std::cerr << "Error: invalid argument to '-maxlevel': " << argv[i+1] << " (should be greater than 0)\n";
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'n' && argv[i][2] == 'c' && argv[i][3] == 'o' && argv[i][4] == 'l' && argv[i][5] == 'l'
				  && argv[i][6] == '\0')
				coll = false;
			else if (argv[i][1] == 'a' && argv[i][2] == 'c' && argv[i][3] == 'c' && argv[i][4] == 'u' && argv[i][5] == 'r'
				  && argv[i][6] == 'a' && argv[i][7] == 'c' && argv[i][8] == 'y' && argv[i][9] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-accuracy'\n";
					return -1;
				}
				b_accuracy = true;
				accuracy = atof(argv[i+1]);
				if (accuracy <= 0)
				{
					std::cerr << "Error: invalid argument to '-accuracy': " << argv[i+1] << " (should be greater than 0)\n";
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'c' && argv[i][2] == 'p' && argv[i][3] == 'u' && argv[i][4] == '\0')
				cpu = true;
			else if (argv[i][1] == 'c' && argv[i][2] == 'p' && argv[i][3] == 'u' && argv[i][4] == '-' && argv[i][5] == 't'
				  && argv[i][6] == 'h' && argv[i][7] == 'r' && argv[i][8] == 'e' && argv[i][9] == 'a' && argv[i][10] == 'd'
				  && argv[i][11] == 's' && argv[i][12] == '\0')
			{
				cpu = true;
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-cpu-threads'\n";
					return -1;
				}
				CPU_THREADS = atoi(argv[i+1]);
				if (CPU_THREADS <= 0)
				{
					std::cerr << "Error: invalid argument to '-cpu-threads': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'c' && argv[i][2] == 'a' && argv[i][3] == 'c' && argv[i][4] == 'h' && argv[i][5] == 'e'
				  && argv[i][6] == 'l' && argv[i][7] == 'i' && argv[i][8] == 'n' && argv[i][9] == 'e' && argv[i][10] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-cacheline'\n";
					return -1;
				}
				CACHE_LINE_SIZE = atoi(argv[i+1]);
				if (CACHE_LINE_SIZE <= 0)
				{
					std::cerr << "Error: invalid argument to '-cacheline': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 't' && argv[i][2] == 'e' && argv[i][3] == 's' && argv[i][4] == 't' && argv[i][5] == '\0')
				test = true;
			else if (argv[i][1] == 't' && argv[i][2] == 'e' && argv[i][3] == 's' && argv[i][4] == 't' && argv[i][5] == '2'
			      && argv[i][6] == '\0')
				test2 = true;
			else if (argv[i][1] == 'x' && argv[i][2] == 'i' && argv[i][3] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-xi'\n";
					return -1;
				}
				xi = atof(argv[i+1]);
				if (xi < 0)
				{
					std::cerr << "Error: invalid argument to '-xi': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'o' && argv[i][2] == 'm' && argv[i][3] == 'e' && argv[i][4] == 'g' && argv[i][5] == 'a'
				  && argv[i][6] == '0' && argv[i][7] == '\0')
			{
				if (i+2 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-omega0'\n";
					return -1;
				}
				omega0.x = atof(argv[i+1]);
				omega0.y = atof(argv[i+2]);
				if (omega0.x < 0 || omega0.y < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-omega0': " << argv[i+1] << ' ' << argv[i+2] << '\n';
					return -1;
				}
				i += 2;
			}
			else if (argv[i][1] == 'x' && argv[i][2] == '\0')
			{
				if (i+3 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-x'\n";
					return -1;
				}
				x.x = atof(argv[i+1]);
				x.y = atof(argv[i+2]);
				x.z = atof(argv[i+3]);
				if (x.x < 0 || x.y < 0 || x.z < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-x': " << argv[i+1] << ' ' << argv[i+2] << ' ' << argv[i+3] << '\n';
					return -1;
				}
				i += 3;
			}
			else if (argv[i][1] == 'u' && argv[i][2] == '\0')
			{
				if (i+3 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-u'\n";
					return -1;
				}
				u.x = atof(argv[i+1]);
				u.y = atof(argv[i+2]);
				u.z = atof(argv[i+3]);
				if (u.x < 0 || u.y < 0 || u.z < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-u': " << argv[i+1] << ' ' << argv[i+2] << ' ' << argv[i+3] << '\n';
					return -1;
				}
				i += 3;
			}
			else
			{
				std::cerr << "Error: unrecognised option '" << argv[i] << "'\n";
				return -1;
			}
		}
		else
		{
			strin = argv[i];
			in = true;
		}
	}

	int bytes, cpyBytes;
	char *c_buf = nullptr;
	SCAL *buf = nullptr;

	if (in)
	{
		std::ifstream fin(strin, std::ios::in | std::ios::binary);
		if (fin)
		{
			fin.ignore(std::numeric_limits<std::streamsize>::max());
			cpyBytes = (int)fin.gcount();
			nBodies = cpyBytes / 2 / sizeof(ALIGNED_VEC);
			bytes = 3 * nBodies * sizeof(ALIGNED_VEC);
			c_buf = (char*)(new ALIGNED_VEC[bytes/sizeof(ALIGNED_VEC)]);
			buf = (SCAL*)c_buf;
			fin.clear();
			fin.seekg(0, std::ios::beg);
			fin.read(c_buf, cpyBytes);
		}
		else
		{
			std::cerr << "Error: cannot read from input location." << std::endl;

			delete[] buf;

			return -1;
		}
	}
	else
	{
		bytes = 3 * nBodies * sizeof(ALIGNED_VEC);
		cpyBytes = 2 * nBodies * sizeof(ALIGNED_VEC);
		c_buf = (char*)(new ALIGNED_VEC[bytes/sizeof(ALIGNED_VEC)]);
		buf = (SCAL*)c_buf;
		
		//std::cout << "emittances: " << x * u << std::endl;

		std::mt19937_64 gen(5351550349027530206);
		gen.discard(624*2);
		initGA((ALIGNED_VEC*)buf, 2 * nBodies, x, u, gen);
		if (test)
			initU((ALIGNED_VEC*)buf, 2 * nBodies, {-1, -1, -1}, {1, 1, 1}, gen);
	}

	if (!test && !test2)
	{
		std::ofstream farg(strout + "/args.txt", std::ios::out);
		if (farg)
			for (int i = 0; i < argc; ++i)
				farg << argv[i] << ' ';
		else
		{
			std::cerr << "Error: cannot write on output location. "
						 "Check that \"" << strout << "\" folder exists. Create it if not." << std::endl;
			delete[] buf;

			return -1;
		}
	}
	
	ALIGNED_VEC apar[]{
		{
			xi/(SCAL)nBodies, // xi/N
			0, // padding
			0, // padding
		},
		{
			omega0.x*omega0.x, // omegax0^2 = kx
			omega0.y*omega0.y, // omegay0^2 = ky
			omega0.z*omega0.z, // omegay0^2 = kz
		},
	};
	SCAL *par = (SCAL*)apar;

	SCAL *d_buf, *d_par;

	if (!cpu)
	{
		 // allocate memory on GPU
		gpuErrchk(cudaMalloc((void**)&d_buf, bytes));
		gpuErrchk(cudaMalloc((void**)&d_par, 2*sizeof(ALIGNED_VEC)));

		// copy data from CPU to GPU
		gpuErrchk(cudaMemcpy(d_buf, buf, cpyBytes, cudaMemcpyHostToDevice)); // acc not copied
		gpuErrchk(cudaMemcpy(d_par, par, 2*sizeof(ALIGNED_VEC), cudaMemcpyHostToDevice));
	}

	auto test_time = [cpu, nBodies, buf, par, d_buf, d_par](bool warming_up = true, SCAL min_loop = 0, int loop_n = 1)
	{
		// warming up
		if (cpu)
			compute_force(fmm_cart3_kdtree_cpu, buf, nBodies, par);
		else
			compute_force(fmm_cart3_kdtree, d_buf, nBodies, d_par);

		SCAL duration;
		int loop_counter = 0;
		auto begin = steady_clock::now();
		decltype(begin) end;

		do
		{
			for (int i = 0; i < loop_n; ++i)
				if (cpu)
					compute_force(fmm_cart3_kdtree_cpu, buf, nBodies, par);
				else
					compute_force(fmm_cart3_kdtree, d_buf, nBodies, d_par);

			end = steady_clock::now();
			loop_counter += loop_n;
			loop_n *= 2;
			duration = duration_cast<microseconds>(end - begin).count() * (SCAL)1.e-6;
		} while (duration < min_loop);

		return duration / loop_counter;
	};

	if (b_accuracy)
	{
		std::vector<int> search_p = {1, 2, 3, 4, 5, 6};
		std::vector<SCAL> search_r = {1.11, 1.25, 1.43, 1.67, 2, 2.5, 3};

		SCAL best_r, best_time = FLT_MAX, best_accuracy, curr_accuracy, curr_time;
		int best_p;

		::coll = true;

		std::cout << "Parameter optimization in progress, please wait" << std::flush;

		for (SCAL r : search_r)
			for (int p : search_p)
			{
				::tree_radius = r;
				::fmm_order = p;

				if (cpu)
					curr_accuracy = test_accuracy_cpu(fmm_cart3_kdtree_cpu, direct3_cpu, buf, nBodies, par);
				else
					curr_accuracy = test_accuracy(fmm_cart3_kdtree, direct3, d_buf, nBodies, d_par);

				if (curr_accuracy < accuracy)
				{
					curr_time = test_time(false);
					if (curr_time < best_time)
					{
						best_r = r;
						best_p = p;
						best_accuracy = curr_accuracy;
						best_time = curr_time;
					}
				}
				std::cout << '.' << std::flush;
			}
		if (best_time == FLT_MAX)
		{
			std::cout << "\nOptimization failed!" << std::endl;
			return -1;
		}
		else
		{
			::tree_radius = best_r;
			::fmm_order = best_p;
			std::cout << "\nBest parameters: ";
			std::cout << "r = " << best_r;
			std::cout << ", p = " << best_p;
			std::cout << ", time = " << best_time;
			std::cout << ", error = " << best_accuracy << std::endl;
		}
	}

	if (test)
	{
		::b_unsort = false;
		std::cout << fmm_order << ": ";
		std::cout << "Average time: "
				  << test_time(true, 1)
				  << " [s]" << std::endl;
		::b_unsort = true;

		for (::fmm_order = 1; ::fmm_order <= 10; ++::fmm_order)
		{
			SCAL relerr;
			std::cout << fmm_order << ": ";

			if (cpu)
				relerr = test_accuracy_cpu(fmm_cart3_kdtree_cpu, direct3_cpu, buf, nBodies, par);
			else
				relerr = test_accuracy(fmm_cart3_kdtree, direct3, d_buf, nBodies, d_par, ::fmm_order == 1);

			std::cout << "Relative error: " << relerr << std::endl;
		}
	}
	else if (test2)
	{
		::b_unsort = false;
		for (int i = 0; i < ::tree_steps+1; ++i)
		{
			SCAL relerr;
			if (cpu)
			{
				relerr = test_accuracy_cpu(fmm_cart3_kdtree_cpu, direct3_cpu, buf, nBodies, par);
				pre_symplectic_euler(add_elastic_cpu, buf, nBodies, par+3, dt, step_cpu);
			}
			else
			{
				relerr = test_accuracy(fmm_cart3_kdtree, direct3, d_buf, nBodies, d_par);
				pre_symplectic_euler(add_elastic, d_buf, nBodies, d_par+3, dt);
			}

			std::cout << "Relative error after " << i << " steps: " << relerr << std::endl;
		}
	}
	else
	{
		::b_unsort = false;
		// precompute accelerations
		if (cpu)
			compute_force(coulombOscillatorFMMKD3_cpu, buf, nBodies, par);
		else
			compute_force(coulombOscillatorFMMKD3, d_buf, nBodies, d_par);

		for (int iter = 0; iter < nIters; ++iter)
		{
			if (cpu)
				symp_integ(coulombOscillatorFMMKD3_cpu, buf, nBodies, par, dt, step_cpu, 1);
			else
				symp_integ(coulombOscillatorFMMKD3, d_buf, nBodies, d_par, dt, step, 1);
			
			if (iter % nSteps == 0)
			{
				std::cout << iter << ' ' << std::flush;
				// copy data from GPU to CPU
				if (!cpu)
					gpuErrchk(cudaMemcpy(buf, d_buf, cpyBytes, cudaMemcpyDeviceToHost)); // acc not copied

				std::ofstream fout(strout + "/out" + std::to_string(iter) + '_' + std::to_string(dt)
								 + ".bin", std::ios::out | std::ios::binary);
				if (fout)
					fout.write(c_buf, cpyBytes); // write to file (note that c_buf = (char*)buf)
				else
				{
					std::cerr << "Error: cannot write on output location. "
								 "Check that \"" << strout << "\" folder exists. Create it if not." << std::endl;
					if (!cpu)
					{
						gpuErrchk(cudaFree(d_buf)); // free memory from GPU
						gpuErrchk(cudaFree(d_par));
					}
					delete[] buf;

					return -1;
				}
			}
		}
	}
	
	if (!cpu)
	{
		gpuErrchk(cudaFree(d_buf)); // free memory from GPU
		gpuErrchk(cudaFree(d_par));
	}
	delete[] buf;

	return 0;
}