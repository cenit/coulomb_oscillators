﻿//  Symplectic integrators
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

#ifndef INTEGRATOR_CUDA_H
#define INTEGRATOR_CUDA_H

#include "kernel.cuh"

void compute_force(void(*f)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), SCAL *d_buf, int n, const SCAL* param)
{
	ParticleSystem d_p = { (ALIGNED_VEC*)d_buf, ((ALIGNED_VEC*)d_buf) + n, ((ALIGNED_VEC*)d_buf) + 2 * n };

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);
}

// symplectic methods

void symplectic_euler(void(*f)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), SCAL *d_buf, int n,
                      const SCAL* param, long double dt,
                      void(*step_func)(ALIGNED_VEC*, const ALIGNED_VEC*, SCAL, int) = step,
                      long double scale = 1)
{
// 1st order
	ParticleSystem d_p = { (ALIGNED_VEC*)d_buf, ((ALIGNED_VEC*)d_buf) + n, ((ALIGNED_VEC*)d_buf) + 2 * n };

	// v += a * dt
	step_func(d_p.vel, d_p.acc, dt * scale, n);

	// x += v * dt
	step_func(d_p.pos, d_p.vel, dt, n);

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);
}

void pre_symplectic_euler(void(*f)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), SCAL *d_buf, int n,
                          const SCAL* param, long double dt,
                          void(*step_func)(ALIGNED_VEC*, const ALIGNED_VEC*, SCAL, int) = step,
                          long double scale = 1)
{
// 1st order
	ParticleSystem d_p = { (ALIGNED_VEC*)d_buf, ((ALIGNED_VEC*)d_buf) + n, ((ALIGNED_VEC*)d_buf) + 2 * n };

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);

	// v += a * dt
	step_func(d_p.vel, d_p.acc, dt * scale, n);

	// x += v * dt
	step_func(d_p.pos, d_p.vel, dt, n);
}

void leapfrog(
	void(*f)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*),	              // pointer to function for the evaulation of the field f(x)
	SCAL *d_buf,                                                          // pointer to buffer containing x, v, a
	int n,                                                                // number of particles
	const SCAL* param,                                                    // additional parameters
	long double dt,                                                       // timestep
	void(*step_func)(ALIGNED_VEC*, const ALIGNED_VEC*, SCAL, int) = step, // pointer to function for step (multiply/addition)
	long double scale = 1                                                 // quantity that rescales the field f(x)
)
{
// 2nd order
	ParticleSystem d_p = { (ALIGNED_VEC*)d_buf, // positions
	                      ((ALIGNED_VEC*)d_buf) + n, // velocities
	                      ((ALIGNED_VEC*)d_buf) + 2 * n // accelerations
	};
	long double ds = dt * scale * 0.5L;

	// v += a * dt / 2
	step_func(d_p.vel, d_p.acc, ds, n);

	// x += v * dt
	step_func(d_p.pos, d_p.vel, dt, n);

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);

	// v += a * dt / 2
	step_func(d_p.vel, d_p.acc, ds, n);
}

constexpr long double fr_par = 1.3512071919596576340476878089715L; // 1 / (2 - cbrt(2))

void forestruth(void(*f)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), SCAL *d_buf, int n,
				const SCAL* param, long double dt, void(*step_func)(ALIGNED_VEC*, const ALIGNED_VEC*, SCAL, int) = step,
				long double scale = 1)
{
// Forest-Ruth method
// 4th order
	ParticleSystem d_p = { (ALIGNED_VEC*)d_buf, // positions
	                      ((ALIGNED_VEC*)d_buf) + n, // velocities
	                      ((ALIGNED_VEC*)d_buf) + 2 * n // accelerations
	};
	long double ds = dt * scale;

	step_func(d_p.pos, d_p.vel, SCAL(dt * fr_par / 2), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * fr_par), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * (1 - fr_par) / 2), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * (1 - 2*fr_par)), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * (1 - fr_par) / 2), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * fr_par), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * fr_par / 2), n);
}

constexpr long double pefrl_parx = +0.1786178958448091E+00L;
constexpr long double pefrl_parl = -0.2123418310626054E+00L;
constexpr long double pefrl_parc = -0.6626458266981849E-01L;

void pefrl(void(*f)(ALIGNED_VEC*, ALIGNED_VEC*, int, const SCAL*), SCAL *d_buf, int n,
				const SCAL* param, long double dt, void(*step_func)(ALIGNED_VEC*, const ALIGNED_VEC*, SCAL, int) = step,
				long double scale = 1)
{
// Position-extended Forest-Ruth-like method
// 4th order, slower but more accurate
	ParticleSystem d_p = { (ALIGNED_VEC*)d_buf, // positions
	                      ((ALIGNED_VEC*)d_buf) + n, // velocities
	                      ((ALIGNED_VEC*)d_buf) + 2 * n // accelerations
	};
	long double ds = dt * scale;

	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parx), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * (1 - 2 * pefrl_parl) / 2), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parc), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * pefrl_parl), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * (1 - 2 * (pefrl_parc + pefrl_parx))), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * pefrl_parl), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parc), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * (1 - 2 * pefrl_parl) / 2), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parx), n);
}

#endif // !INTEGRATOR_CUDA_H