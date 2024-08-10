#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cuda_runtime.h>
#include <cmath>

#define KTYPE float

#define NX (12*45)
#define NY (12*45)
#define NQ (17)
#define BLOCK_SIZE (32)
#define N_STREAMS (1)

// Plasma model flags
#define XMHD true
#define IMHD false
#define YLBC 0
#define YHBC 0
#define XHBC 0
#define XLBC 0

// Define Fortran-like parameters
#define rh 0
#define mx 1
#define my 2
#define mz 3
#define en 4
#define bx 5
#define by 6
#define bz 7
#define ex 8
#define ey 9
#define ez 10
#define jx 11
#define jy 12
#define jz 13
#define et 14
#define ne 15
#define ep 16

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
