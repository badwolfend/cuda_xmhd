#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cuda_runtime.h>
#include <cmath>

#define KTYPE float

// extern KTYPE L0_host = 1.0e-3;
// extern KTYPE t0_host = 100.0e-9;
// extern KTYPE n0_host = 6.0e28;
// extern KTYPE pi_host = 3.14159265;
// extern KTYPE mu0_host = 4.e-7 * pi_host;
// extern KTYPE eps0_host = 8.85e-12;
// extern KTYPE mu_host = 27.0;
// extern KTYPE mime_host = mu_host * 1836.0;
// extern KTYPE memi_host = 1.0 / mime_host;
// extern KTYPE elc_host = 1.6011E-19;
// extern KTYPE kb_host = 1.38064E-23;

// extern KTYPE v0_host = L0_host / t0_host;
// extern KTYPE b0_host = sqrt(mu_host * 1.67e-27 * mu0_host * n0_host) * v0_host;
// extern KTYPE e0_host = v0_host * b0_host;
// extern KTYPE j0_derived_host = b0_host / (mu0_host * L0_host);
// extern KTYPE eta0_host = mu0_host * v0_host * L0_host;
// extern KTYPE p0_host = mu_host * 1.67e-27 * n0_host * v0_host * v0_host;
// extern KTYPE te0_host = p0_host / n0_host / 1.6e-19;

// extern KTYPE cflm_host = 0.8;
// extern KTYPE gma2_host = 2.8e-8 * mu0_host * L0_host * L0_host * n0_host;
// extern KTYPE lil0_host = 2.6e5 * sqrt(mu_host / n0_host / mu0_host) / L0_host;
// extern KTYPE cab_host = t0_host / L0_host / sqrt(eps0_host * mu0_host);
// extern KTYPE ecyc_host = 1.76e11 * b0_host * t0_host;
// extern KTYPE clt_host = cab_host / 1.0;
// extern KTYPE cfva2_host = (cab_host / 1.0) * (cab_host / 1.0);
// extern KTYPE vhcf_host = 1.0e6;
// extern KTYPE rh_floor_host = 1.0E-9;
// extern KTYPE T_floor_host = 0.026 / te0_host;
// extern KTYPE rh_mult_host = 1.1;
// extern KTYPE P_floor_host = rh_floor_host * T_floor_host;
// extern KTYPE bapp_host = 0;

// extern KTYPE Z_host = 4.0;
// extern KTYPE er2_host = 1.0;
// extern KTYPE aindex_host = 1.1;

// extern KTYPE taui0_host = 6.2e14*sqrt(mu_host)/n0_host;
// extern KTYPE taue0_host = 61.*taui0_host; 
// extern KTYPE omi0_host = 9.6e7*Z_host*b0_host/mu_host;
// extern KTYPE ome0_host = mime_host*omi0_host;
// extern KTYPE visc0_host = 5.3*t0_host;  

// extern KTYPE lamb_host = 527E-9 / L0_host;
// extern KTYPE k_las_host = 2.0 * pi_host / lamb_host;
// extern KTYPE f_las_host = clt_host * k_las_host;
// extern KTYPE tperiod_host = 2.0 * pi_host / f_las_host;
// extern KTYPE dgrate_host = 2.0 * lamb_host;

// extern KTYPE L0_host;
// extern KTYPE t0_host;
// extern KTYPE n0_host;
// extern KTYPE pi_host;
// extern KTYPE mu0_host;
// extern KTYPE eps0_host;
// extern KTYPE mu_host;
// extern KTYPE mime_host;
// extern KTYPE memi_host;
// extern KTYPE elc_host;
// extern KTYPE kb_host;

// extern KTYPE v0_host;
// extern KTYPE b0_host;
// extern KTYPE e0_host;
// extern KTYPE j0_derived_host;
// extern KTYPE eta0_host;
// extern KTYPE p0_host;
// extern KTYPE te0_host;

// extern KTYPE cflm_host;
// extern KTYPE gma2_host;
// extern KTYPE lil0_host;
// extern KTYPE cab_host;
// extern KTYPE ecyc_host;
// extern KTYPE clt_host;
// extern KTYPE cfva2_host;
// extern KTYPE vhcf_host;
// extern KTYPE rh_floor_host;
// extern KTYPE T_floor_host;
// extern KTYPE rh_mult_host;
// extern KTYPE P_floor_host;
// extern KTYPE bapp_host;

// extern KTYPE Z_host;
// extern KTYPE er2_host;
// extern KTYPE aindex_host;

// extern KTYPE taui0_host;
// extern KTYPE taue0_host; 
// extern KTYPE omi0_host;
// extern KTYPE ome0_host;
// extern KTYPE visc0_host;  

// extern KTYPE lamb_host;
// extern KTYPE k_las_host;
// extern KTYPE f_las_host;
// extern KTYPE tperiod_host;
// extern KTYPE dgrate_host;

void initializeGlobals();

// Dimensional units (expressed in MKS)
__constant__ KTYPE L0;
__constant__ KTYPE t0;
__constant__ KTYPE n0;
__constant__ KTYPE pi;
__constant__ KTYPE mu0;
__constant__ KTYPE eps0;
__constant__ KTYPE mu;
__constant__ KTYPE mime;
__constant__ KTYPE memi;
__constant__ KTYPE elc;
__constant__ KTYPE kb;

// Derived units
__constant__ KTYPE v0;
__constant__ KTYPE b0;
__constant__ KTYPE e0;
__constant__ KTYPE j0_derived; // Renamed to avoid conflict
__constant__ KTYPE eta0;
__constant__ KTYPE p0;
__constant__ KTYPE te0;

__constant__ KTYPE cflm;
__constant__ KTYPE gma2;
__constant__ KTYPE lil0;
__constant__ KTYPE cab;
__constant__ KTYPE ecyc;
__constant__ KTYPE clt;
__constant__ KTYPE cfva2;
__constant__ KTYPE vhcf;
__constant__ KTYPE rh_floor;
__constant__ KTYPE T_floor;
__constant__ KTYPE rh_mult;
__constant__ KTYPE P_floor;
__constant__ KTYPE bapp;

__constant__ KTYPE Z;
__constant__ KTYPE er2;
__constant__ KTYPE aindex;

__constant__ KTYPE taui0;
__constant__ KTYPE omi0;
__constant__ KTYPE ome0;
__constant__ KTYPE taue0;
__constant__ KTYPE visc0;

__constant__ KTYPE lamb;
__constant__ KTYPE k_las;
__constant__ KTYPE f_las;
__constant__ KTYPE tperiod;
__constant__ KTYPE dgrate;



    // real(ktype), parameter ::  sigma = 1.2e2, taui0 = 6.2e14*sqrt(mu)/n0, omi0 = 9.6e7*Z*b0/mu, bapp = 0., visc0 = 5.3*t0, ome0 = mime*omi0, taue0 = 61.*taui0 


#endif // CONSTANTS_H
