#include "constants.h"

#define KTYPE float

KTYPE L0_host;
KTYPE t0_host;
KTYPE n0_host;
KTYPE pi_host;
KTYPE mu0_host;
KTYPE eps0_host;
KTYPE mu_host;
KTYPE mime_host;
KTYPE memi_host;
KTYPE elc_host;
KTYPE kb_host;

KTYPE v0_host;
KTYPE b0_host;
KTYPE e0_host;
KTYPE j0_derived_host;
KTYPE eta0_host;
KTYPE p0_host;
KTYPE te0_host;

KTYPE cflm_host;
KTYPE gma2_host;
KTYPE lil0_host;
KTYPE cab_host;
KTYPE ecyc_host;
KTYPE clt_host;
KTYPE cfva2_host;
KTYPE vhcf_host;
KTYPE rh_floor_host;
KTYPE T_floor_host;
KTYPE rh_mult_host;
KTYPE P_floor_host;
KTYPE bapp_host;

KTYPE Z_host;
KTYPE er2_host;
KTYPE aindex_host;

KTYPE taui0_host;
KTYPE taue0_host; 
KTYPE omi0_host;
KTYPE ome0_host;
KTYPE visc0_host;  

KTYPE lamb_host;
KTYPE k_las_host;
KTYPE f_las_host;
KTYPE tperiod_host;
KTYPE dgrate_host;

static int isInitialized = 0;

void initializeGlobals() {
  if (!isInitialized) {
    L0_host = 1.0e-3;
    t0_host = 100.0e-9;
    n0_host = 6.0e28;
    pi_host = 3.14159265;
    mu0_host = 4.e-7 * pi_host;
    eps0_host = 8.85e-12;
    mu_host = 27.0;
    mime_host = mu_host * 1836.0;
    memi_host = 1.0 / mime_host;
    elc_host = 1.6011E-19;
    kb_host = 1.38064E-23;

    v0_host = L0_host / t0_host;
    b0_host = sqrt(mu_host * 1.67e-27 * mu0_host * n0_host) * v0_host;
    e0_host = v0_host * b0_host;
    j0_derived_host = b0_host / (mu0_host * L0_host);
    eta0_host = mu0_host * v0_host * L0_host;
    p0_host = mu_host * 1.67e-27 * n0_host * v0_host * v0_host;
    te0_host = p0_host / n0_host / 1.6e-19;

    cflm_host = 0.8;
    gma2_host = 2.8e-8 * mu0_host * L0_host * L0_host * n0_host;
    lil0_host = 2.6e5 * sqrt(mu_host / n0_host / mu0_host) / L0_host;
    cab_host = t0_host / L0_host / sqrt(eps0_host * mu0_host);
    ecyc_host = 1.76e11 * b0_host * t0_host;
    clt_host = cab_host / 1.0;
    cfva2_host = (cab_host / 1.0) * (cab_host / 1.0);
    vhcf_host = 1.0e6;
    rh_floor_host = 1.0E-9;
    T_floor_host = 0.026 / te0_host;
    rh_mult_host = 1.1;
    P_floor_host = rh_floor_host * T_floor_host;
    bapp_host = 0;

    Z_host = 4.0;
    er2_host = 1.0;
    aindex_host = 1.1;

    taui0_host = 6.2e14*sqrt(mu_host)/n0_host;
    taue0_host = 61.*taui0_host; 
    omi0_host = 9.6e7*Z_host*b0_host/mu_host;
    ome0_host = mime_host*omi0_host;
    visc0_host = 5.3*t0_host;  

    lamb_host = 527E-9 / L0_host;
    k_las_host = 2.0 * pi_host / lamb_host;
    f_las_host = clt_host * k_las_host;
    tperiod_host = 2.0 * pi_host / f_las_host;
    dgrate_host = 2.0 * lamb_host;
    isInitialized = 1;
  }
}