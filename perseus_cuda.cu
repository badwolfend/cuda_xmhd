#include <cuda_runtime.h>
#include <stdio.h>
#include "constants.h"

#define NX (12*30)
#define NY (12*30)
#define NQ (17)

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


// Function prototypes
__global__ void calc_flux_x_kernel(float *Qx, float *cfx, float *ffx, int nx, int nq);
__global__ void calc_flux_y_kernel(float *Qy, float *cfy, float *ffy, int ny, int nq);
__global__ void compute_wr_wl(float *Qin, float *ff, float *cfr, float *wr, float *wl, int n, int nq, int pr);
__global__ void compute_fr_fl(float *wr, float *wl, float *fr, float *fl, int n, int nq);
__global__ void compute_dfr_dfl(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, int n, int nq);
__global__ void compute_flux2(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, float *flux2, float sl, int n, int nq);
__global__ void get_flux_kernel(float *Qin, float *flux_x, float *flux_y, float sl);
__global__ void advance_time_level_rz_kernel(float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float dxt, float dyt, float dt, int nx, int ny) ;
__global__ void limit_flow_kernel(float* Q, int nx, int ny);
__device__ float eta_s(float e_Temp, float Za, float dne, float cln_min);
__device__ float xc(int i, float dxi);
__device__ float yc(int j, float dyi);

__device__ float d_dt;

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
    printf("host: cfva2 ... %e\n", cfva2_host);
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


// CUDA kernel to initialize a specific variable slice
__global__ void initVariableSlice(float *arr, int nx, int ny, int nq, int variableIndex, float value) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = x * (ny * nq) + y * nq + variableIndex;

  if (x < nx && y < ny) {
      arr[idx] = value;  // Initialize the specific variable slice to the given value
  }
}

void initialize_constants() {
  initializeGlobals();

  cudaMemcpyToSymbol(L0, &L0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(t0, &t0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(n0, &n0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(pi, &pi_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(mu0, &mu0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(eps0, &eps0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(mu, &mu_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(mime, &mime_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(memi, &memi_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(elc, &elc_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(kb, &kb_host, sizeof(KTYPE));

  cudaMemcpyToSymbol(v0, &v0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(b0, &b0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(e0, &e0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(j0_derived, &j0_derived_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(eta0, &eta0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(p0, &p0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(te0, &te0_host, sizeof(KTYPE));

  cudaMemcpyToSymbol(cflm, &cflm_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(gma2, &gma2_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(lil0, &lil0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(cab, &cab_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(ecyc, &ecyc_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(clt, &clt_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(cfva2, &cfva2_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(vhcf, &vhcf_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(rh_floor, &rh_floor_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(T_floor, &T_floor_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(rh_mult, &rh_mult_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(P_floor, &P_floor_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(bapp, &bapp_host, sizeof(KTYPE));

  cudaMemcpyToSymbol(Z, &Z_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(er2, &er2_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(aindex, &aindex_host, sizeof(KTYPE));

  cudaMemcpyToSymbol(taui0, &taui0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(taue0, &taue0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(omi0, &omi0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(ome0, &ome0_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(visc0, &visc0_host, sizeof(KTYPE));

  cudaMemcpyToSymbol(lamb, &lamb_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(k_las, &k_las_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(f_las, &f_las_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(tperiod, &tperiod_host, sizeof(KTYPE));
  cudaMemcpyToSymbol(dgrate, &dgrate_host, sizeof(KTYPE));
  cudaDeviceSynchronize();
}


__device__ float rc(int i) {
  // In the original Fortran code, rc is always 1.0.
  // If there's more logic, you can implement it here.
  return 1.0f;
}

__global__ void advance_time_level_rz_kernel(float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float mdxt, float mdyt, float dt, int nx, int ny, int nq) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idxnQ = i* (ny * nq) + j* nq ;
  int idxnQ_ip1 = (i+1)* (ny * nq) + j* nq ;
  int idxnQ_im1 = (i-1)* (ny * nq) + j* nq ;
  int idxnQ_jp1 = i* (ny * nq) + (j+1)* nq ;
  int idxnQ_jm1 = i* (ny * nq) + (j-1)* nq ;

  int idx = i* (ny * 1) + j* 1;

  if (i >= 2 && i < nx-1 && j >= 2 && j < ny-1) {
    float rbp = 0.5f * (rc(i+1) + rc(i));
    float rbm = 0.5f * (rc(i) + rc(i-1));
    float rci = 1.0f / rc(i);
    // printf("Cell %d %d %d %e.\n", i, j, rc(i), Qinp1[i * ny * NQ + j * NQ + rh]);

    Qinp1[idxnQ + rh] = Qin[idxnQ + rh] * rc(i) 
                                - mdxt * (flux_x[idxnQ + rh] * rbp - flux_x[idxnQ_im1 + rh] * rbm) 
                                - mdyt * (flux_y[idxnQ + rh] - flux_y[idxnQ_jm1 + rh]) * rc(i);
    
    Qinp1[idxnQ + mx] = Qin[idxnQ + mx] * rc(i) 
                                - mdxt * (flux_x[idxnQ + mx] * rbp - flux_x[idxnQ_im1 + mx] * rbm) 
                                - mdyt * (flux_y[idxnQ + mx] - flux_y[idxnQ_jm1 + mx]) * rc(i) 
                                + dt * source[idxnQ + mx];
                                
    Qinp1[idxnQ + my] = Qin[idxnQ + my] * rc(i) 
                                - mdxt * (flux_x[idxnQ + my] * rbp - flux_x[idxnQ_im1 + my] * rbm) 
                                - mdyt * (flux_y[idxnQ + my] - flux_y[idxnQ_jm1 + my]) * rc(i) 
                                + dt * source[idxnQ + my];
                                
    Qinp1[idxnQ + mz] = Qin[idxnQ + mz] * rc(i) 
                                - mdxt * (flux_x[idxnQ + mz] * rbp - flux_x[idxnQ_im1 + mz] * rbm) 
                                - mdyt * (flux_y[idxnQ + mz] - flux_y[idxnQ_jm1 + mz]) * rc(i) 
                                + dt * source[idxnQ + mz];
                                
    Qinp1[idxnQ + en] = Qin[idxnQ + en] * rc(i) 
                                - mdxt * (flux_x[idxnQ + en] * rbp - flux_x[idxnQ_im1 + en] * rbm) 
                                - mdyt * (flux_y[idxnQ + en] - flux_y[idxnQ_jm1 + en]) * rc(i) 
                                + dt * source[idxnQ + en];
                                Qinp1[i * ny * NQ + j * NQ + bx] = Qin[i * ny * NQ + j * NQ + bx] * rc(i) 
                                - 0.5f * mdyt * (flux_y[idxnQ_jp1 + bx] * rc(i) - flux_y[idxnQ_jm1 + bx] * rc(i));

    Qinp1[idxnQ + bx] = Qin[idxnQ + bx]*rc(i) - mdxt*(flux_x[idxnQ + bx]*rbp-flux_x[idxnQ_im1+bx]*rbm)
                                - mdyt*(flux_y[idxnQ + bx]-flux_y[idxnQ_jm1+bx])*rc(i); 

    Qinp1[idxnQ + by] = Qin[idxnQ + by]*rc(i) - mdxt*(flux_x[idxnQ + by]*rbp-flux_x[idxnQ_im1 + by]*rbm)
                                - mdyt*(flux_y[idxnQ + by]-flux_y[idxnQ_jm1 + by])*rc(i); 

    Qinp1[idxnQ + bz] = Qin[idxnQ + bz]*rc(i) - mdxt*(flux_x[idxnQ + bz]*rbp-flux_x[idxnQ_im1 + bz]*rbm)
                                - mdyt*(flux_y[idxnQ + bz]-flux_y[idxnQ_jm1 + bz])*rc(i) + dt*source[idxnQ + bz];

    Qinp1[idxnQ + ex] = Qin[idxnQ + ex] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + ex] * rbp - flux_x[idxnQ_im1 + ex] * rbm) 
                                      - mdyt * (flux_y[idxnQ + ex] - flux_y[idxnQ_jm1 + ex]) * rc(i) 
                                      + dt * source[idxnQ + ex];

    Qinp1[idxnQ + ey] = Qin[idxnQ + ey] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + ey] * rbp - flux_x[idxnQ_im1 + ey] * rbm) 
                                      - mdyt * (flux_y[idxnQ + ey] - flux_y[idxnQ_jm1 + ey]) * rc(i) 
                                      + dt * source[idxnQ + ey];
    // if ((j == 2 || j == 3) && i == 10) {
    //   printf("flux_x[%d][%d] = %e.\n", i, j, flux_x[idxnQ + ex]);
    //   printf("flux_x[%d][%d] = %e\n", i, j, flux_x[idxnQ_im1 + ex]);
    //   printf("flux_y[%d][%d] = %e\n", i, j, flux_y[idxnQ + ex]);
    //   printf("flux_y[%d][%d] = %e\n", i, j, flux_y[idxnQ_jm1 + ex]);
    // } 

    Qinp1[idxnQ + ez] = Qin[idxnQ + ez] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + ez] * rbp - flux_x[idxnQ_im1 + ez] * rbm) 
                                      - mdyt * (flux_y[idxnQ + ez] - flux_y[idxnQ_jm1 + ez]) * rc(i) 
                                      + dt * source[idxnQ + ez];

    Qinp1[idxnQ + ne] = Qin[idxnQ + ne] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + ne] * rbp - flux_x[idxnQ_im1 + ne] * rbm) 
                                      - mdyt * (flux_y[idxnQ + ne] - flux_y[idxnQ_jm1 + ne]) * rc(i) 
                                      + dt * source[idxnQ + ne];

    Qinp1[idxnQ + jx] = Qin[idxnQ + jx] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + jx] * rbp - flux_x[idxnQ_im1 + jx] * rbm) 
                                      - mdyt * (flux_y[idxnQ + jx] - flux_y[idxnQ_jm1 + jx]) * rc(i) 
                                      + dt * source[idxnQ + jx] 
                                      + mdxt * (flux_x[idxnQ + ep] - flux_x[idxnQ_im1 + ep]) * rc(i);

    Qinp1[idxnQ + jy] = Qin[idxnQ + jy] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + jy] * rbp - flux_x[idxnQ_im1 + jy] * rbm) 
                                      - mdyt * (flux_y[idxnQ + jy] - flux_y[idxnQ_jm1 + jy]) * rc(i) 
                                      + dt * source[idxnQ + jy] 
                                      + mdyt * (flux_y[idxnQ + ep] - flux_y[idxnQ_jm1 + ep]) * rc(i);

    Qinp1[idxnQ + jz] = Qin[idxnQ + jz] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + jz] * rbp - flux_x[idxnQ_im1 + jz] * rbm) 
                                      - mdyt * (flux_y[idxnQ + jz] - flux_y[idxnQ_jm1 + jz]) * rc(i) 
                                      + dt * source[idxnQ + jz];

    Qinp1[idxnQ + et] = Qin[idxnQ + et] * rc(i) 
                                      - mdxt * (flux_x[idxnQ + et] * rbp - flux_x[idxnQ_im1 + et] * rbm) 
                                      - mdyt * (flux_y[idxnQ + et] - flux_y[idxnQ_jm1 + et]) * rc(i) 
                                      + dt * source[idxnQ + et];

    if (isnan(Qinp1[idxnQ + rh])) {
      // Print the number of the cell where NaN was encountered
      printf("NaN encountered in cell %d %d %e. Exiting...\n", i, j, Qinp1[idxnQ + rh]);
      return;
    }

    for (int k = rh; k <= ne; ++k) {
      Qinp1[idxnQ + k] *= rci;
    }
  }
}

void advance_time_level_rz(dim3 bs, dim3 gs, float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float mdxt, float mdyt, float dt, int nx, int ny, int nq) {
  // Launch the kernel
  advance_time_level_rz_kernel<<<gs, bs>>>(Qin, flux_x, flux_y, source, Qinp1, mdxt, mdyt, dt, nx, ny, nq);

  // Synchronize to check for any kernel launch errors
  cudaDeviceSynchronize();
}

__global__ void limit_flow_kernel(float* Q, int nx, int ny, int nq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idxnQ = i* (ny * nq) + j* nq ;

    // if (i < nx && j < ny) {
    //     if (Q[idx] < 0) Q[idx] = 0;
    // }
}

__global__ void set_bc_kernel(float *Qin, float t, float dxi, float dyi, float k_las, float f_las, int nx, int ny, int nq) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idxnQ = i* (ny * nq) + j* nq ;

  // Handle the left and right boundary conditions
  if (j < ny) {
    for (int l = 0; l < nq; ++l) {
        // Right boundary
        if (i == nx - 1 || i == nx - 2) {
            Qin[(nx - 1) * ny * nq + j * nq + l] = Qin[(nx - 2) * ny * nq + j * nq + l] * xc(nx-2, dxi)/xc(nx-1,dxi);
            Qin[(nx - 2) * ny * nq + j * nq + l] = Qin[(nx - 3) * ny * nq + j * nq + l] * xc(nx-3, dxi)/xc(nx-2, dxi);            
          }

        // Left boundary
        if (i == 0 || i == 1) {
          Qin[1 * ny * nq + j * nq + l] = Qin[2 * ny * nq + j * nq + l];
          Qin[0 * ny * nq + j * nq + l] = Qin[1 * ny * nq + j * nq + l];

          // Laser boundary conditions
          Qin[i * ny * nq + j * nq + bz] = 0.1f*cos(k_las * xc(i, dxi) - f_las * t);
          Qin[i * ny * nq + j * nq + ey] = cab*0.1f*cos(k_las * xc(i, dxi) - f_las * t);
        }
    }
  }

  // Handle the top and bottom boundary conditions
  if (i < nx) {
    for (int l = 0; l < nq; ++l) {
        // Top boundary
        if (j == ny - 1 || j == ny - 2) {
          Qin[i * ny * nq + (ny - 1) * nq + l] = Qin[i * ny * nq + (ny - 2) * nq + l];
          Qin[i * ny * nq + (ny - 2) * nq + l] = Qin[i * ny * nq + (ny - 3) * nq + l];
        }
        // Bottom boundary
        if (j == 0 || j == 1) {
          Qin[i * ny * nq + 1 * nq + l] = Qin[i * ny * nq + 2 * nq + l];
          Qin[i * ny * nq + 0 * nq + l] = Qin[i * ny * nq + 1 * nq + l];

          // // Laser boundary conditions
          Qin[i * ny * nq + j * nq + bz] = 0.1f*cos(k_las * yc(j, dyi) - f_las * t);
          Qin[i * ny * nq + j * nq + ex] = -cab*0.1f*cos(k_las * yc(j, dyi) - f_las * t);
        }
    }
  }
  
}

void set_bc(dim3 bs, dim3 gs, float *Qin, float t, float dxi, float dyi, float k_las, float f_las, int nx, int ny, int nq) {
  set_bc_kernel<<<gs, bs>>>(Qin, t, dxi, dyi, k_las, f_las, nx, ny, nq);
  cudaDeviceSynchronize();
}

__device__ float xc(int i, float dxi) {
  return i / dxi;
}

__device__ float yc(int i, float dyi) {
  return i / dyi;
}

__global__ void implicit_source2_kernel(float* Qin, float* flux_x, float* flux_y, float* eta, float* Qout, float dxt, float dyt, float dt, int nx, int ny, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  int idxnQ = i* (ny * nq) + j* nq ;
  int idx = i* (ny * 1) + j* 1;

  if (i < nx - 2 && j < ny - 2) {
    float Ainv[3][3];
    float Qjx, Qjy, Qjz, Qex, Qey, Qez, ma2, mamb, mb2, ma, mb, denom;
    float ux, uy, uz, hx, hy, hz, dnii, sig, mati, fac, zero, dne;

    fac = 1.0f / (mime + Z);
    mb = ecyc * dt;
    mb2 = mb * mb;

    if (XMHD) {
      dne = Qin[idxnQ + ne];
      zero = 1.0f;
      if (dne < Z * rh_floor * rh_mult) zero = 0.0f;

      dnii = 1.0f / Qin[idxnQ+rh];
      ux = Qin[idxnQ + mx] * dnii;
      uy = Qin[idxnQ + my] * dnii;
      uz = Qin[idxnQ + mz] * dnii;

      hx = Qin[idxnQ + bx];
      hy = Qin[idxnQ + by];
      hz = Qin[idxnQ + bz];

      if (!XMHD) {
          ux = ux - fac * 1.0f * Qin[idxnQ + jx] * dnii;
          uy = uy - fac * 1.0f * Qin[idxnQ + jy] * dnii;
          uz = uz - fac * 1.0f * Qin[idxnQ + jz] * dnii;
      }

      ma = 1.0f + dt * gma2 * dne * (eta[idx] + dt * cfva2);
      ma2 = ma * ma;
      mamb = ma * mb;

      denom = zero / (ma * (ma2 + mb2 * (hx * hx + hy * hy + hz * hz)));

      Ainv[0][0] = denom * (ma2 + mb2 * hx * hx);
      Ainv[1][1] = denom * (ma2 + mb2 * hy * hy);
      Ainv[2][2] = denom * (ma2 + mb2 * hz * hz);
      Ainv[0][1] = denom * (mb2 * hx * hy - mamb * hz);
      Ainv[1][2] = denom * (mb2 * hy * hz - mamb * hx);
      Ainv[2][0] = denom * (mb2 * hz * hx - mamb * hy);
      Ainv[1][0] = denom * (mb2 * hx * hy + mamb * hz);
      Ainv[2][1] = denom * (mb2 * hy * hz + mamb * hx);
      Ainv[0][2] = denom * (mb2 * hz * hx + mamb * hy);

      Qjx = Qin[idxnQ + jx] + dt * gma2 * (dne * (Qin[idxnQ + ex] + uy * hz - uz * hy) + dxt * 1.0f * (flux_x[idxnQ + ep] - flux_x[(i - 1) * ny * nq + j * nq + ep]));
      Qjy = Qin[idxnQ + jy] + dt * gma2 * (dne * (Qin[idxnQ + ey] + uz * hx - ux * hz) + dyt * 1.0f * (flux_y[idxnQ + ep] - flux_y[i * ny * nq + (j - 1) * nq + ep]));
      Qjz = Qin[i * NY * NQ + j * NQ + 13] + dt * gma2 * (dne * (Qin[i * NY * NQ + j * NQ + 10] + ux * hy - uy * hx));

      Qout[idxnQ + jx] = Ainv[0][0] * Qjx + Ainv[0][1] * Qjy + Ainv[0][2] * Qjz;
      Qout[idxnQ + jy] = Ainv[1][0] * Qjx + Ainv[1][1] * Qjy + Ainv[1][2] * Qjz;
      Qout[idxnQ + jz] = Ainv[2][0] * Qjx + Ainv[2][1] * Qjy + Ainv[2][2] * Qjz;

      Qout[idxnQ + ex] = Qin[idxnQ + ex] - dt * cfva2 * Qout[idxnQ + jx];
      Qout[idxnQ + ey] = Qin[idxnQ + ey] - dt * cfva2 * Qout[idxnQ + jy];
      Qout[idxnQ + ez] = Qin[idxnQ + ez] - dt * cfva2 * Qout[idxnQ + jz];
    }

    if (IMHD) {
      float dni = 1.0f / Qin[idxnQ+rh];

      sig = 1.0f / (eta[idx] + 1.0e6f * rh_floor * dni);

      Qex = Qin[idxnQ + ex] - cfva2 * dni * dt * sig * (Qin[idxnQ + my] * Qin[idxnQ + bz] - Qin[idxnQ + mz] * Qin[idxnQ + by]);
      Qey = Qin[idxnQ + ey] - cfva2 * dni * dt * sig * (Qin[idxnQ + mz] * Qin[idxnQ + bx] - Qin[idxnQ + mx] * Qin[idxnQ + bz]);
      Qez = Qin[idxnQ + ez] - cfva2 * dni * dt * sig * (Qin[idxnQ + mx] * Qin[idxnQ + by] - Qin[idxnQ + my] * Qin[idxnQ + bx]);

      mati = 1.0f / (1.0f + dt * cfva2 * sig);

      Qout[idxnQ + ex] = mati * Qex;
      Qout[idxnQ + ey] = mati * Qey;
      Qout[idxnQ + ez] = mati * Qez;

      Qout[idxnQ + jx] = sig * (Qout[idxnQ + ex] + dni * (Qin[idxnQ + my] * Qin[idxnQ + bz] - Qin[idxnQ + mz] * Qin[idxnQ + by]));
      Qout[idxnQ + jy] = sig * (Qout[idxnQ + ey] + dni * (Qin[idxnQ + mz] * Qin[idxnQ + bx] - Qin[idxnQ + mx] * Qin[idxnQ + bz]));
      Qout[idxnQ + jz] = sig * (Qout[idxnQ + ez] + dni * (Qin[idxnQ + mx] * Qin[idxnQ + by] - Qin[idxnQ+my] * Qin[idxnQ+bx]));
    }
  }
}

void implicit_source2(dim3 bs, dim3 gs, float* Qin, float* flux_x, float* flux_y, float* eta, float* Qout, float dxi, float dyi, float dt, int nx, int ny, int nq) {
  float dxt = dxi * dt;
  float dyt = dyi * dt;

  implicit_source2_kernel<<<gs, bs>>>(Qin, flux_x, flux_y, eta, Qout, dxt, dyt, dt, nx, ny, nq);
  cudaDeviceSynchronize();
}

void print_flux_x(float *flux_x, int nx, int ny, int nq, int x, int y, int q_index) {
  int idxnQ = x* (ny * nq) + y* nq ;
  printf("flux_x[%d][%d][%d] = %e\n", x, y, q_index, flux_x[idxnQ + q_index]);
}

// Function to print an array (for testing purposes)
void printArray(const char *name, float *array, int n, int nq) {
  printf("%s:\n", name);
  for (int i = 0; i < n; ++i) {
    printf("i = %d ", i);
    for (int q = 0; q < nq; ++q) {
      printf("%f ", array[i * nq + q]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void copy3DTo2DSliceX(float *d_Qin, float *d_Qx, int nx, int ny, int nq, int j) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && q < nq) {
      d_Qx[i * nq + q] = d_Qin[i * ny * nq + j * nq + q];
      // if (i == 4 && j == 4 ) {
      //   printf("d_Qin[%d][%d] = %e\n", i, q, d_Qin[i * ny * nq + j * nq + bz]);
      //   // printf("d_Qx[%d][%d] = %e\n", i, q, d_Qx[i * nq + q]);
      // }
  }
}

__global__ void copy3DTo2DSliceY(float *d_Qin, float *d_Qy, int nx, int ny, int nq, int i) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < nx && q < nq) {
      d_Qy[j * nq + q] = d_Qin[i * ny * nq + j * nq + q];

      // if (i == 4 && j == 1 && q == ex) {
      //   printf("d_Qin[%d][%d] = %e\n", i, q, d_Qin[i * ny * nq + j * nq + ex]);
      // }
  }
}

__global__ void copy2DTo3DSliceX(float *d_Qx, float *d_Qin, int nx, int ny, int nq, int j) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && q < nq) {
      int dst_idx = i * ny * nq + j * nq + q;
      int src_idx = i * nq + q;
      d_Qin[dst_idx] = d_Qx[src_idx];
  }
}

__global__ void copy2DTo3DSliceY(float *d_Qy, float *d_Qin, int nx, int ny, int nq, int i) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < ny && q < nq) {
      int dst_idx = i * ny * nq + j * nq + q;
      int src_idx = j * nq + q;
      d_Qin[dst_idx] = d_Qy[src_idx];
  }
}

// Function to perform get_flux operations
void get_flux(dim3 bs, dim3 gs, float *d_Qin, float *d_flux_x, float *d_flux_y, float sl, int nx, int ny, int nq) {
  size_t size_3d = nx * ny * nq * sizeof(float);
  size_t size_2d_x = nx * nq * sizeof(float);
  size_t size_2d_y = ny * nq * sizeof(float);

  // Define grid and block dimensions
  dim3 bs1d(16);
  dim3 gridDimX((nx + bs1d.x - 1) / bs1d.x, 1);
  dim3 gridDimY((ny + bs1d.x - 1) / bs1d.x, 1);

  dim3 gridDimXQ((nx + bs.x - 1) / bs.x, (nq + bs.y - 1) / bs.y);
  dim3 gridDimYQ((ny + bs.x - 1) / bs.x, (nq + bs.y - 1) / bs.y);

  // Allocate memory on the device
  float *d_Qx, *d_Qy, *d_cfx, *d_ffx, *d_fx, *d_cfy, *d_ffy, *d_fy;
  float *d_wr, *d_wl, *d_fr, *d_fl, *d_dfrp, *d_dfrm, *d_dflp, *d_dflm;
  float *d_wry, *d_wly, *d_fry, *d_fly, *d_dfrpy, *d_dfrmy, *d_dflpy, *d_dflmy;
  int flag;
  cudaMalloc(&d_Qx, size_2d_x);
  cudaMalloc(&d_Qy, size_2d_y);
  cudaMalloc(&d_cfx, size_2d_x);
  cudaMalloc(&d_ffx, size_2d_x);
  cudaMalloc(&d_fx, size_2d_x);
  cudaMalloc(&d_cfy, size_2d_y);
  cudaMalloc(&d_ffy, size_2d_y);
  cudaMalloc(&d_fy, size_2d_y);
  cudaMalloc(&d_wr, size_2d_x);
  cudaMalloc(&d_wl, size_2d_x);
  cudaMalloc(&d_fr, size_2d_x);
  cudaMalloc(&d_fl, size_2d_x);
  cudaMalloc(&d_dfrp, size_2d_x);
  cudaMalloc(&d_dfrm, size_2d_x);
  cudaMalloc(&d_dflp, size_2d_x);
  cudaMalloc(&d_dflm, size_2d_x);

  cudaMalloc(&d_wry, size_2d_y);
  cudaMalloc(&d_wly, size_2d_y);
  cudaMalloc(&d_fry, size_2d_y);
  cudaMalloc(&d_fly, size_2d_y);
  cudaMalloc(&d_dfrpy, size_2d_y);
  cudaMalloc(&d_dfrmy, size_2d_y);
  cudaMalloc(&d_dflpy, size_2d_y);
  cudaMalloc(&d_dflmy, size_2d_y);
  flag = 0;
  // Launch kernels for each slice in the y-dimension
  for (int j = 0; j < ny; j++) {
    // Copy the slice from the 3D array to the 2D array
    // cudaMemcpy(d_Qx, d_Qin + j * nx * nq, size_2d_x, cudaMemcpyDeviceToDevice);
    copy3DTo2DSliceX<<<gridDimXQ, bs>>>(d_Qin, d_Qx, nx, ny, nq, j);
    cudaDeviceSynchronize();
    
    // Launch the kernel to copy the j-th slice to d_Qx
    // Launch calc_flux_x_kernel
    calc_flux_x_kernel<<<gridDimX, bs1d>>>(d_Qx, d_cfx, d_ffx, nx, nq);
    cudaDeviceSynchronize();

    // Launch TVD kernels
    compute_wr_wl<<<gridDimXQ, bs>>>(d_Qx, d_ffx, d_cfx, d_wr, d_wl, nx, nq, flag);
    cudaDeviceSynchronize();

    compute_fr_fl<<<gridDimXQ, bs>>>(d_wr, d_wl, d_fr, d_fl, nx, nq);
    cudaDeviceSynchronize();

    compute_dfr_dfl<<<gridDimXQ, bs>>>(d_fr, d_fl, d_dfrp, d_dfrm, d_dflp, d_dflm, nx, nq);
    cudaDeviceSynchronize();

    compute_flux2<<<gridDimXQ, bs>>>(d_fr, d_fl, d_dfrp, d_dfrm, d_dflp, d_dflm, d_fx, sl, nx, nq);
    cudaDeviceSynchronize();

    // Copy results back to the corresponding slice in flux_x
    // cudaMemcpy(d_flux_x + j * nx * nq, d_fx, size_2d_x, cudaMemcpyDeviceToDevice);
    copy2DTo3DSliceX<<<gridDimXQ, bs>>>(d_fx, d_flux_x, nx, ny, nq, j);
    cudaDeviceSynchronize();

  }

  // Launch kernels for each slice in the x-dimension
  for (int i = 0; i < nx; i++) {
    // // Copy the slice from the 3D array to the 2D array
    // cudaMemcpy(d_Qy, d_Qin + i * ny * nq, size_2d_y, cudaMemcpyDeviceToDevice);

    // Launch the kernel to copy the j-th slice to d_Qx
    copy3DTo2DSliceY<<<gridDimYQ, bs>>>(d_Qin, d_Qy, nx, ny, nq, i);
    cudaDeviceSynchronize();

    // Launch calc_flux_y_kernel
    calc_flux_y_kernel<<<gridDimY, bs1d>>>(d_Qy, d_cfy, d_ffy, ny, nq);
    cudaDeviceSynchronize();

    // // Launch TVD kernels
    if ( i == 4) {
      flag = 1;
    }
    else {
      flag = 0;
    }
    compute_wr_wl<<<gridDimYQ, bs>>>(d_Qy, d_ffy, d_cfy, d_wry, d_wly, ny, nq, flag);
    cudaDeviceSynchronize();

    compute_fr_fl<<<gridDimYQ, bs>>>(d_wry, d_wly, d_fry, d_fly, ny, nq);
    cudaDeviceSynchronize();

    compute_dfr_dfl<<<gridDimYQ, bs>>>(d_fry, d_fly, d_dfrpy, d_dfrmy, d_dflpy, d_dflmy, ny, nq);
    cudaDeviceSynchronize();

    compute_flux2<<<gridDimYQ, bs>>>(d_fry, d_fly, d_dfrpy, d_dfrmy, d_dflpy, d_dflmy, d_fy, sl, ny, nq);
    cudaDeviceSynchronize();

    // Copy results back to the corresponding slice in flux_y
    // cudaMemcpy(d_flux_y + i * ny * nq, d_fy, size_2d_y, cudaMemcpyDeviceToDevice);
    copy2DTo3DSliceY<<<gridDimYQ, bs>>>(d_fy, d_flux_y, nx, ny, nq, i);
    cudaDeviceSynchronize();
  }

  // Free device memory
  cudaFree(d_Qx);
  cudaFree(d_Qy);
  cudaFree(d_cfx);
  cudaFree(d_ffx);
  cudaFree(d_fx);
  cudaFree(d_cfy);
  cudaFree(d_ffy);
  cudaFree(d_fy);

  cudaFree(d_wr);
  cudaFree(d_wl);
  cudaFree(d_fr);
  cudaFree(d_fl);
  cudaFree(d_dfrp);
  cudaFree(d_dfrm);
  cudaFree(d_dflp);
  cudaFree(d_dflm);

  cudaFree(d_wry);
  cudaFree(d_wly);
  cudaFree(d_fry);
  cudaFree(d_fly);
  cudaFree(d_dfrpy);
  cudaFree(d_dfrmy);
  cudaFree(d_dflpy);
  cudaFree(d_dflmy);

}

__global__ void calc_flux_x_kernel(float *Qx, float *cfx, float *flx, int nx, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i* nq;

  if (i >= nx) return;
  if (i <= 0) return;

  float dni, dnii, vx, vy, vz, dne, dnei, vex, vey, vez;
  float hx, hy, hz, Pri, Pre, P, asqr, vf1, vf2, vf3;
  const float fac = 1.0f / (mime + Z);

  dni = Qx[idx + rh];
  dnii = 1.0f / dni;
  vx = Qx[idx + mx] * dnii;
  vy = Qx[idx + my] * dnii;
  vz = Qx[idx + mz] * dnii;

  dne = Qx[idx + ne];
  dnei = 1.0f / dne;
  vex = vx - lil0 * Qx[idx + jx] * dnei;
  vey = vy - lil0 * Qx[idx + jy] * dnei;
  vez = vz - lil0 * Qx[idx + jz] * dnei;

  hx = Qx[idx + bx];
  hy = Qx[idx + by];
  hz = Qx[idx + bz];

  Pri = (aindex - 1) * (Qx[idx + en] - 0.5f * dni * (vx * vx + vy * vy + vz * vz));
  if (Pri < P_floor) Pri = P_floor;

  Pre = (aindex - 1) * (Qx[idx + et] - 0.5f * memi * dne * (vex * vex + vey * vey + vez * vez));
  if (Pre < Z * P_floor) Pre = Z * P_floor;

  if (XMHD) {
      P = Pri + Pre;
      if (IMHD) P = Pri;
  } else {
      P = Pri;
      vx = (mime * vx + Z * vex) * fac;
      vy = (mime * vy + Z * vey) * fac;
      vz = (mime * vz + Z * vez) * fac;
  }

  flx[idx + rh] = Qx[idx + mx];
  flx[idx + mx] = Qx[idx + mx] * vx + P;
  flx[idx + my] = Qx[idx + my] * vx;
  flx[idx + mz] = Qx[idx + mz] * vx;
  flx[idx + en] = (Qx[idx + en] + Pri) * vx;

  flx[idx + bx] = 0.0f;
  flx[idx + by] = -Qx[idx + ez];
  flx[idx + bz] = Qx[idx + ey];

  flx[idx + ex] = 0.0f;
  flx[idx + ey] = cfva2 * hz;
  flx[idx + ez] = -cfva2 * hy;
  flx[idx + ne] = dne * vex;
  flx[idx + jx] = vx * Qx[idx + jx] + Qx[idx + jx] * vx - lil0 * dnei * Qx[idx + jx] * Qx[idx + jx];
  flx[idx + jy] = vx * Qx[idx + jy] + Qx[idx + jx] * vy - lil0 * dnei * Qx[idx + jx] * Qx[idx + jy];
  flx[idx + jz] = vx * Qx[idx + jz] + Qx[idx + jx] * vz - lil0 * dnei * Qx[idx + jx] * Qx[idx + jz];
  flx[idx + et] = (Qx[idx + et] + Pre) * vex;
  flx[idx + ep] = Pre;

  asqr = sqrt(aindex * P * dnii);
  vf1 = sqrt(vx * vx + hz * hz * dnii + aindex * P * dnii);
  vf2 = clt;
  vf3 = fabs(vex) + asqr;
  
  cfx[idx + rh] = vf1;
  cfx[idx + mx] = vf1;
  cfx[idx + my] = vf1;
  cfx[idx + mz] = vf1;
  cfx[idx + en] = vf1;
  cfx[idx + ne] = vf3;
  cfx[idx + jx] = vf3;
  cfx[idx + jy] = vf3;
  cfx[idx + jz] = vf3;
  cfx[idx + et] = vf3;
  cfx[idx + bx] = vf2;
  cfx[idx + by] = vf2;
  cfx[idx + bz] = vf2;
  cfx[idx + ex] = vf2;
  cfx[idx + ey] = vf2;
  cfx[idx + ez] = vf2;

  // if (i == 4 || i == 5 || i == 6) {
  //   // printf("Qx[%d]=%e\n", i, Qx[idx+bz]);
  //   // printf("p[%d], i[%d]\n", ((i + 1) % n) * nq + q, idx);
  //   // printf("frp[%d] = %e, fr[%d]=%e\n", idx, fr[((i + 1) % n) * nq + q], idx, fr[idx]);
  //   // printf("dfrp[%d] = %e, dfrm[%d] = %e, fr[%d]=%e\n", idx, dfrp[idx], idx, dfrm[idx], idx, fr[idx]);
  //   // printf("dfrp[%d] = %e, dfrm[%d] = %e\n", idx, dfrp[idx], idx, dfrm[idx]);
  // } 

  // if (i == 4) {
  //   printf("cfx[%d] = %e, ff[%d] = %e\n", idx, cfx[idx+bz], idx, flx[idx+bz]);
  // } 
}

__global__ void calc_flux_y_kernel(float* Qy, float* cfy, float* fly, int ny, int nq) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = j* nq;

  if (j >= ny) return;
  if (j <= 0) return;

  float vx, vy, vz, P, vf1, vf2, vf3, asqr, hx, hy, hz, dni, dne, dnii, dnei, vex, vey, vez, Pri, Pre;
  float fac = 1.0f / (mime + Z);

  dni = Qy[idx+ rh];
  dnii = 1.0f / dni;
  vx = Qy[idx+ mx] * dnii;
  vy = Qy[idx+ my] * dnii;
  vz = Qy[idx+ mz] * dnii;

  dne = Qy[idx+ ne];
  dnei = 1.0f / dne;
  vex = vx - lil0 * Qy[idx+ jx] * dnei;
  vey = vy - lil0 * Qy[idx+ jy] * dnei;
  vez = vz - lil0 * Qy[idx+ jz] * dnei;

  hx = Qy[idx+ bx];
  hy = Qy[idx+ by];
  hz = Qy[idx+ bz];

  Pri = (aindex - 1) * (Qy[idx+ en] - 0.5f * dni * (vx * vx + vy * vy + vz * vz));
  if (Pri < P_floor) Pri = P_floor;

  Pre = (aindex - 1) * (Qy[idx+ et] - 0.5f * memi * dne * (vex * vex + vey * vey + vez * vez));
  if (Pre < Z * P_floor) Pre = Z * P_floor;

  if (XMHD) {
    P = Pri + Pre;
    if (IMHD) P = Pri;
  } else {
    P = Pri;
    vx = (mime * vx + Z * vex) * fac;
    vy = (mime * vy + Z * vey) * fac;
    vz = (mime * vz + Z * vez) * fac;
  }

  fly[idx+ rh] = Qy[idx+ my];
  fly[idx+ mx] = Qy[idx+ mx] * vy;
  fly[idx+ my] = Qy[idx+ my] * vy + P;
  fly[idx+ mz] = Qy[idx+ mz] * vy;
  fly[idx+ en] = (Qy[idx+ en] + Pri) * vy;

  fly[idx+ bx] = Qy[idx+ ez];
  fly[idx+ by] = 0.0f;
  fly[idx+ bz] = -Qy[idx+ ex];

  fly[idx+ ex] = -cfva2 * hz;
  fly[idx+ ey] = 0.0f;
  fly[idx+ ez] = cfva2 * hx;

  fly[idx+ ne] = dne * vey;
  fly[idx+ jx] = vy * Qy[idx+ jx] + Qy[idx+ jy] * vx - lil0 * dnei * Qy[idx+ jy] * Qy[idx+ jx];
  fly[idx+ jy] = vy * Qy[idx+ jy] + Qy[idx+ jy] * vy - lil0 * dnei * Qy[idx+ jy] * Qy[idx+ jy];
  fly[idx+ jz] = vy * Qy[idx+ jz] + Qy[idx+ jy] * vz - lil0 * dnei * Qy[idx+ jy] * Qy[idx+ jz];
  fly[idx+ et] = (Qy[idx+ et] + Pre) * vey;
  fly[idx+ ep] = Pre;

  asqr = sqrtf(aindex * P * dnii);
  vf1 = sqrtf(vy * vy + hz * hz * dnii + aindex * P * dnii);
  vf2 = clt;
  vf3 = fabsf(vey) + asqr;

  cfy[idx+ rh] = vf1;
  cfy[idx+ mx] = vf1;
  cfy[idx+ my] = vf1;
  cfy[idx+ mz] = vf1;
  cfy[idx+ en] = vf1;
  cfy[idx+ ne] = vf3;
  cfy[idx+ jx] = vf3;
  cfy[idx+ jy] = vf3;
  cfy[idx+ jz] = vf3;
  cfy[idx+ et] = vf3;
  cfy[idx+ bx] = vf2;
  cfy[idx+ by] = vf2;
  cfy[idx+ bz] = vf2;
  cfy[idx+ ex] = vf2;
  cfy[idx+ ey] = vf2;
  cfy[idx+ ez] = vf2;


  // if (j == 1 || j == 2 || j == 3) {
  //   printf("fly[%d]=%e\n", j, fly[idx+ex]);
  //   // printf("p[%d], i[%d]\n", ((i + 1) % n) * nq + q, idx);
  //   // printf("frp[%d] = %e, fr[%d]=%e\n", idx, fr[((i + 1) % n) * nq + q], idx, fr[idx]);
  //   // printf("dfrp[%d] = %e, dfrm[%d] = %e, fr[%d]=%e\n", idx, dfrp[idx], idx, dfrm[idx], idx, fr[idx]);
  //   // printf("dfrp[%d] = %e, dfrm[%d] = %e\n", idx, dfrp[idx], idx, dfrm[idx]);
  // } 
  
}

__global__ void compute_wr_wl(float *Qin, float *ff, float *cfr, float *wr, float *wl, int n, int nq, int tp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < n && q < nq) {
    int idx = i * nq + q;
    wr[idx] = cfr[idx] * Qin[idx] + ff[idx];
    wl[idx] = cfr[idx] * Qin[idx] - ff[idx];

    // if (tp == 1)  {
    //   if ((i == 0 || i == 1 || i == 2 || i == 3 || i ==4 )) {
    //     printf("ff[%d]=%e\n", i,  ff[idx+q]);
    //     // printf("p[%d], i[%d]\n", ((i + 1) % n) * nq + q, idx);
    //     // printf("frp[%d] = %e, fr[%d]=%e\n", idx, fr[((i + 1) % n) * nq + q], idx, fr[idx]);
    //     // printf("dfrp[%d] = %e, dfrm[%d] = %e, fr[%d]=%e\n", idx, dfrp[idx], idx, dfrm[idx], idx, fr[idx]);
    //     // printf("dfrp[%d] = %e, dfrm[%d] = %e\n", idx, dfrp[idx], idx, dfrm[idx]);
    //   } 
    // }

    // if (i == 4 && q == bz) {
    //   printf("cfr[%d] = %e, ff[%d] = %e\n", idx, cfr[idx], idx, ff[idx]);
    // } 
  }
}

__global__ void compute_fr_fl(float *wr, float *wl, float *fr, float *fl, int n, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < n && q < nq) {
    int idx = i * nq + q;
    fr[idx] = wr[idx];
    fl[idx] = wl[((i + 1) % n) * nq + q];
    // if (i == 4 && q == bz) {
    //   printf("wl[92]=%e, wl[75]=%e, wl[109]=%e\n", wl[92], wl[75], wl[109]);
    //   // printf("p[%d], i[%d]\n", ((i + 1) % n) * nq + q, idx);
    //   // printf("frp[%d] = %e, fr[%d]=%e\n", idx, fr[((i + 1) % n) * nq + q], idx, fr[idx]);
    //   // printf("dfrp[%d] = %e, dfrm[%d] = %e, fr[%d]=%e\n", idx, dfrp[idx], idx, dfrm[idx], idx, fr[idx]);
    //   // printf("dfrp[%d] = %e, dfrm[%d] = %e\n", idx, dfrp[idx], idx, dfrm[idx]);
    // } 
  }
}

__global__ void compute_dfr_dfl(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, int n, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && q < nq) {
    int idx = i * nq + q;
    dfrp[idx] = fr[((i + 1) % n) * nq + q] - fr[idx];
    dfrm[idx] = fr[idx] - fr[((i - 1 + n) % n) * nq + q];
    dflp[idx] = fl[idx] - fl[((i + 1) % n) * nq + q];
    dflm[idx] = fl[((i - 1 + n) % n) * nq + q] - fl[idx];

    // if (i == 4 && q == bz) {
    //   printf("fr[92]=%e, fr[75]=%e, fr[109]=%e\n", fr[92], fr[75], fr[109]);
    //   // printf("p[%d], i[%d]\n", ((i + 1) % n) * nq + q, idx);
    //   // printf("frp[%d] = %e, fr[%d]=%e\n", idx, fr[((i + 1) % n) * nq + q], idx, fr[idx]);
    //   // printf("dfrp[%d] = %e, dfrm[%d] = %e, fr[%d]=%e\n", idx, dfrp[idx], idx, dfrm[idx], idx, fr[idx]);
    //   // printf("dfrp[%d] = %e, dfrm[%d] = %e\n", idx, dfrp[idx], idx, dfrm[idx]);
    // } 
  }
}

__global__ void compute_flux2(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, float *flux2, float sl, int n, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && q < nq) {
    int idx = i * nq + q;
    
    float dfr, dfl;
    // if (i == 4 && q == bz) {
    //   printf("dfrp[%d] = %e, dfrm[%d] = %e\n", idx, dfrp[idx], idx, dfrm[idx]);
    // } 
    if (dfrp[idx] * dfrm[idx] > 0) {
      dfr = dfrp[idx] * dfrm[idx] / (dfrp[idx] + dfrm[idx]);
    } else {
      dfr = 0.0f;
    }

    if (dflp[idx] * dflm[idx] > 0) {
      dfl = dflp[idx] * dflm[idx] / (dflp[idx] + dflm[idx]);
    } else {
      dfl = 0.0f;
    }

    flux2[idx] = 0.5f * (fr[idx] - fl[idx] + sl * (dfr - dfl));
  }
}

__global__ void get_sources_kernel(
  float* Qin, float* sourcesin, float* eta_in, float* Tiev, float* Teev, float* nuei, float* kap_i, float* kap_e, float* vis_i, int nx, int ny, int nq, float dt, float dxi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idxnQ = i* (ny * nq) + j* nq ;
  int idx = i* (ny * 1) + j* 1;

  if (i < nx && j < ny) {
    float ri, vx, vy, vz, P, viscx = 0.0f, viscy = 0.0f, viscz = 0.0f, mage2, magi2, toi, viscoeff, dx, mui, t0i, dx2;
    float gyro, vix, viy, viz, vxr, vyr, vzr, vex, vey, vez, ux, uy, uz, linv, Zi, dnei, dni, dne, dnii, Zin, dti, fac, theta_np;

    gyro = memi * ecyc;
    linv = 1.0f / lil0;
    fac = lil0 / (mime + Z);
    Zi = 1.0f / Z;
    Zin = 1.0f / (Z + 1.0f);
    dti = 0.2f / dt;
    dx2 = 1.0f / (dt * dxi * dxi);
    dx = 1.0f / dxi;
    mui = (1.0f + Z) * t0;
    t0i = 1.0f / t0;

    // dni = Qin[i + j * nx + rh * nx * ny];
    // dne = Qin[i + j * nx + ne * nx * ny];
    dni = Qin[idxnQ + rh ];
    dne = Qin[idxnQ + ne];
    dnii = 1.0f / dni;
    dnei = 1.0f / dne;

    vix = Qin[idxnQ+mx] * dnii;
    viy = Qin[idxnQ+my] * dnii;
    viz = Qin[idxnQ+mz] * dnii;

    vex = vix - lil0 * Qin[idxnQ+jx] * dnei;
    vey = viy - lil0 * Qin[idxnQ+jy] * dnei;
    vez = viz - lil0 * Qin[idxnQ+jz] * dnei;

    Tiev[idx] = (aindex - 1) * (Qin[idxnQ+en] * dnii - 0.5f * (vix * vix + viy * viy + viz * viz));
    if (Tiev[idx] < T_floor) Tiev[idx] = T_floor;

    Teev[idx] = (aindex - 1) * (Qin[idxnQ+et] * dnei - 0.5f * memi * (vex * vex + vey * vey + vez * vez));
    if (Teev[idx] < T_floor) Teev[idx] = T_floor;

    theta_np = 0.5f * (1.0f + tanh(50.0f * (dni * n0 / 5e28f - 1.0f)));
    eta_in[idx] = eta_s(Teev[idx], Z, dne, 0.001f);
    // printf("eta_in: %e\n",eta_in[idx]);

    if (dni * n0 / 5e28f >= 1.0f) {
      theta_np = 0.5f * (1.0f + tanh(50.0f * (Teev[idx] * te0 / 0.4f - 1.0f)));
      eta_in[idx] = eta_s(Teev[idx], Z, dne, 0.001f);
    }

    nuei[idx] = gma2 * dne * eta_in[idx];

    sourcesin[idxnQ+ne] = dti * (Z * Qin[idxnQ+rh] - Qin[idxnQ+ne]);

    if (XMHD) {
      sourcesin[idxnQ+mx] = Qin[idxnQ+jy] * Qin[idxnQ+bz] - Qin[idxnQ+jz] * Qin[idxnQ+by] + viscx;
      sourcesin[idxnQ+my] = Qin[idxnQ+jz] * Qin[idxnQ+bx] - Qin[idxnQ+jx] * Qin[idxnQ+bz] + viscy;
      sourcesin[idxnQ+mz] = Qin[idxnQ+jx] * Qin[idxnQ+by] - Qin[idxnQ+jy] * Qin[idxnQ+bx] + viscz;

      ux = vix;
      uy = viy;
      uz = viz;
      P = dni * Tiev[idx] + dne * Teev[idx];
    } else {
      sourcesin[idxnQ+mx] = Z * gyro * dni * (Qin[idxnQ+ex] + viy * Qin[idxnQ+bz] - viz * Qin[idxnQ+by]) - memi * lil0 * nuei[idx] * Qin[idxnQ+jx];
      sourcesin[idxnQ+my] = Z * gyro * dni * (Qin[idxnQ+ey] + viz * Qin[idxnQ+bx] - vix * Qin[idxnQ+bz]) - memi * lil0 * nuei[idx] * Qin[idxnQ+jy];
      sourcesin[idxnQ+mz] = Z * gyro * dni * (Qin[idxnQ+ez] + vix * Qin[idxnQ+by] - viy * Qin[idxnQ+bx]) - memi * lil0 * nuei[idx] * Qin[idxnQ+jz];

      ux = vix - fac * Qin[idxnQ+jx] * dnii;
      uy = viy - fac * Qin[idxnQ+jy] * dnii;
      uz = viz - fac * Qin[idxnQ+jz] * dnii;
      P = dni * Tiev[idx];
    }

    if (dni > rh_floor) {
      magi2 = omi0 * omi0 * (Qin[idxnQ+bx] * Qin[idxnQ+bx] + Qin[idxnQ+by] * Qin[idxnQ+by] + Qin[idxnQ+bz] * Qin[idxnQ+bz]);
      kap_i[idx] = min(3.9f * taui0 * t0i * Tiev[idx] * Tiev[idx] * Tiev[idx] * Tiev[idx] * Tiev[idx] / (1.0f + 1.95f * magi2 * taui0 * taui0 * Tiev[idx] * Tiev[idx] * Tiev[idx] * dnii * dnii), 0.001f * dni * dx2);
    }

    if (dne > Z * rh_floor) {
      mage2 = ome0 * ome0 * (Qin[idxnQ+bx] * Qin[idxnQ+bx] + Qin[idxnQ+by] * Qin[idxnQ+by] + Qin[idxnQ+bz] * Qin[idxnQ+bz]);
      kap_e[idx] = min(3.2f * taue0 * t0i * Teev[idx] * Teev[idx] * Teev[idx] * Teev[idx] * Teev[idx] / (1.0f + 0.76f * mage2 * taue0 * taue0 * Teev[idx] * Teev[idx] * Teev[idx] * dnei * dnei), 0.001f * dni * dx2);
    }

    if (dni > rh_floor) {
      vis_i[idx] = min(visc0 * dnii * Tiev[idx] * Tiev[idx] * Tiev[idx] * Tiev[idx] * Tiev[idx] / (1.0f + 0.3f * magi2 * taui0 * taui0 * Tiev[idx] * Tiev[idx] * Tiev[idx] * dnii * dnii), 0.01f * dx2);
    } else {
      vis_i[idx] = 0.0f;
    }

    if (IMHD) {
      sourcesin[idxnQ+en] = Qin[idxnQ+jx] * Qin[idxnQ+ex] + Qin[idxnQ+jy] * Qin[idxnQ+ey] + Qin[idxnQ+jz] * Qin[idxnQ+ez];
      sourcesin[idxnQ+et] = 0.0f;
    } else {
      sourcesin[idxnQ+en] = Z * gyro * dni * (vix * Qin[idxnQ+ex] + viy * Qin[idxnQ+ey] + viz * Qin[idxnQ+ez])
          - memi * lil0 * nuei[idx] * (vix * Qin[idxnQ+jx] + viy * Qin[idxnQ+jy] + viz * Qin[idxnQ+jz])
          + 3 * memi * nuei[idx] * dni * (Teev[idx] - Tiev[idx]);

      sourcesin[idxnQ+et] = -gyro * dne * (vex * Qin[idxnQ+ex] + vey * Qin[idxnQ+ey] + vez * Qin[idxnQ+ez])
          + memi * lil0 * nuei[idx] * (vix * Qin[idxnQ+jx] + viy * Qin[idxnQ+jy] + viz * Qin[idxnQ+jz])
          - 3 * memi * nuei[idx] * dni * (Teev[idx] - Tiev[idx]);
    }

    Qin[idxnQ+ep] = dne * Teev[idx];
    if (rc(i) != rc(i + 1)) {
      sourcesin[idxnQ+ne] *= rc(i);
      sourcesin[idxnQ+mx] = sourcesin[idxnQ+mx] * rc(i) + (Qin[idxnQ+mz] * viz + P);
      sourcesin[idxnQ+mz] = sourcesin[idxnQ+mz] * rc(i) - Qin[idxnQ+mz] * vix;
      sourcesin[idxnQ+en] *= rc(i);
      sourcesin[idxnQ+et] *= rc(i);
      sourcesin[idxnQ+jx] = uz * Qin[idxnQ+jz] + Qin[idxnQ+jz] * uz - lil0 * dnei * Qin[idxnQ+jz] * Qin[idxnQ+jz];
      sourcesin[idxnQ+jz] = -ux * Qin[idxnQ+jz] + Qin[idxnQ+jx] * uz - lil0 * dnei * Qin[idxnQ+jz] * Qin[idxnQ+jx];
      sourcesin[idxnQ+bz] = Qin[idxnQ+ey];
      sourcesin[idxnQ+ez] = -cfva2 * Qin[idxnQ+by];
    }
  }
}

__device__ float eta_s(float e_Temp, float Za, float dne, float cln_min) {
  float Cln;
  const float c_eta = 1.0f; // Assuming c_eta is defined elsewhere in your program
  const float cln1 = 23.5f; // Assuming cln1 is defined elsewhere in your program
  const float cln2 = 24.0f; // Assuming cln2 is defined elsewhere in your program
  const float te0 = 1.0f;   // Assuming te0 is defined elsewhere in your program

  if (e_Temp >= 10.0f * Za * Za / te0) {
      Cln = c_eta * (cln2 - 0.5f * logf(dne / (e_Temp * e_Temp)));
  } else {
      Cln = c_eta * (cln1 - 0.5f * logf(Za * Za * dne / (e_Temp * e_Temp * e_Temp)));
  }

  if (Cln < cln_min) {
      Cln = cln_min;
  }

  float eta_s_value = Za * Cln / powf(e_Temp, 1.5f);
  eta_s_value = c_eta * 3.0f * Cln / powf(e_Temp, 1.5f);

  return eta_s_value;
}

void check_Iv(float lxu, float pin_height) {
    // Implement check_Iv logic here
}

void get_min_dt(float* dt, float cfl, float dxi, float vmax) {
  // Add logic to compute the minimum dt based on the CFL condition
  *dt = cfl/(dxi*vmax); // Placeholder value

  // Copy the computed dt value to the device constant
  cudaMemcpyToSymbol(d_dt, dt, sizeof(float), 0, cudaMemcpyHostToDevice);
}


__global__ void initial_condition_kernel(float *Qin, int nx, int ny, int nq, int variableIndex, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (ny * nq) + j * nq + variableIndex;

  if (i < nx && j < ny) {
      Qin[idx] = value;  // Initialize the specific variable slice to the given value
  }
}

__global__ void add_Q_kernel(float *Q1, float *Q2, float *Qout, int nx, int ny, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (ny * nq) + j * nq;

  if (i < nx && j < ny) {
      for (int q = 0; q < nq; q++) {
          Qout[idx + q] = 0.5*(Q1[idx + q] + Q2[idx + q]);  // Add the source term to the specific variable slice
      }
  }
}

void initial_condition(dim3 bs, dim3 gs, float *Qin, float rh_floor, float Z, float T_floor, float aindex, float bapp, float b0, int nx, int ny, int nq) {

  // Set all to zero first
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, rh, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, mx, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, my, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, mz, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, en, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, bx, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, by, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, bz, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, ex, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, ey, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, ez, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, jx, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, jy, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, jz, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, et, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, ne, 0.0f);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, ep, 0.0f);  
  cudaDeviceSynchronize();

  
  // Set the initial conditions for the variables but first set them all to zero
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, rh, rh_floor);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, ne, Z * rh_floor);
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, en, T_floor * rh_floor / (aindex - 1.0f));
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, en, T_floor * rh_floor / (aindex - 1.0f));
  cudaDeviceSynchronize();
  initial_condition_kernel<<<gs, bs>>>(Qin, nx, ny, nq, et, Z*T_floor * rh_floor / (aindex - 1.0f));
  cudaDeviceSynchronize();
}

void write_vtk(const char* prefix, float* Q, float* fluxx, int nx, int ny, int nq, int timestep) {
  char filename[256];
  sprintf(filename, "%s_%04d.vtk", prefix, timestep);

  FILE* file = fopen(filename, "w");
  if (file == NULL) {
      fprintf(stderr, "Failed to open file for writing: %s\n", filename);
      return;
  }

  // Write VTK header
  fprintf(file, "# vtk DataFile Version 3.0\n");
  fprintf(file, "VTK output\n");
  fprintf(file, "ASCII\n");
  fprintf(file, "DATASET STRUCTURED_GRID\n");
  fprintf(file, "DIMENSIONS %d %d 1\n", nx, ny);
  fprintf(file, "POINTS %d float\n", nx * ny);

  // Write grid points
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%d %d 0\n", i, j);
      }
  }

  // Write scalar field: rh
  fprintf(file, "POINT_DATA %d\n", nx * ny);
  fprintf(file, "SCALARS rh float 1\n");
  fprintf(file, "LOOKUP_TABLE default\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e\n", Q[i * ny * nq + j * nq + rh]);
          // printf("rh: in print %e\n", Q[i * ny * nq + j * nq + rh]);
      }
  }


  // Write scalar field: fluxx
  fprintf(file, "SCALARS fluxx float 1\n");
  fprintf(file, "LOOKUP_TABLE default\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e\n", fluxx[i * ny * nq + j * nq + ey]);
      }
  }

  // Write vector field: B
  fprintf(file, "VECTORS B float\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e %e %e\n", Q[i * ny * nq + j * nq + bx], Q[i * ny * nq + j * nq + by], Q[i * ny * nq + j * nq + bz]);
          // printf("B: in print %e %e %e\n", Q[i * ny * nq + j * nq + bx], Q[i * ny * nq + j * nq + by], Q[i * ny * nq + j * nq + bz]);
      }
  }

  // Write vector field: E
  fprintf(file, "VECTORS E float\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e %e %e\n", Q[i * ny * nq + j * nq + ex], Q[i * ny * nq + j * nq + ey], Q[i * ny * nq + j * nq + ez]);
          // printf("E: in print %e %e %e\n", Q[i * ny * nq + j * nq + ex], Q[i * ny * nq + j * nq + ey], Q[i * ny * nq + j * nq + ez]);
      }
  }

  fclose(file);
}


int main() {
    // Initialize constants
    initialize_constants();
    float *Q, *flux_x, *flux_y, *sources, *Q1, *Q2, *Q3, *Q4, *eta, *ect, *Tiev, *Teev, *nuei, *kap_i, *kap_e, *vis_i;
    float *d_Q, *d_flux_x, *d_flux_y, *d_sources, *d_Q1, *d_Q2, *d_Q3, *d_Q4, *d_eta, *d_ect, *d_Tiev, *d_Teev, *d_nuei, *d_kap_i, *d_kap_e, *d_vis_i;
    float dxt, dyt;
    float dt, t = 0.0f, dx=12.0f*lamb_host/NX, dxi = 1.0f/dx, dy=12.0f*lamb_host/NY, dyi = 1.0f/dy;
    int nprint = 0, niter = 0, iorder = 2;
    int nout = 0;

    Q = (float*)malloc(NX * NY * NQ * sizeof(float));
    flux_x = (float*)malloc(NX * NY * NQ * sizeof(float));
    flux_y = (float*)malloc(NX * NY * NQ * sizeof(float));
    sources = (float*)malloc(NX * NY * NQ * sizeof(float));
    Q1 = (float*)malloc(NX * NY * NQ * sizeof(float));
    Q2 = (float*)malloc(NX * NY * NQ * sizeof(float));
    Q3 = (float*)malloc(NX * NY * NQ * sizeof(float));
    Q4 = (float*)malloc(NX * NY * NQ * sizeof(float));
    eta = (float*)malloc(NX * NY * sizeof(float));
    Tiev = (float*)malloc(NX * NY * sizeof(float));
    Teev = (float*)malloc(NX * NY * sizeof(float));
    nuei = (float*)malloc(NX * NY * sizeof(float));
    ect = (float*)malloc(NX * NY * sizeof(float));
    kap_i = (float*)malloc(NX * NY * sizeof(float));
    kap_e = (float*)malloc(NX * NY * sizeof(float));
    vis_i = (float*)malloc(NX * NY * sizeof(float));

    cudaMalloc(&d_Q, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_flux_x, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_flux_y, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_sources, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_Q1, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_Q2, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_Q3, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_Q4, NX * NY * NQ * sizeof(float));
    cudaMalloc(&d_eta, NX * NY * sizeof(float));
    cudaMalloc(&d_Tiev, NX * NY * sizeof(float));
    cudaMalloc(&d_Teev, NX * NY * sizeof(float));
    cudaMalloc(&d_nuei, NX * NY * sizeof(float));
    cudaMalloc(&d_ect, NX * NY * sizeof(float));
    cudaMalloc(&d_kap_i, NX * NY * sizeof(float));
    cudaMalloc(&d_kap_e, NX * NY * sizeof(float));
    cudaMalloc(&d_vis_i, NX * NY * sizeof(float));

    cudaMemcpy(d_Q, Q, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flux_x, flux_x, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flux_y, flux_y, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sources, sources, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q1, Q1, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q2, Q2, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q3, Q3, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q4, Q4, NX * NY * NQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eta, eta, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tiev, Tiev, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Teev, Teev, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nuei, nuei, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ect, ect, NX * NY * sizeof(float), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_kap_i, kap_i, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kap_e, kap_e, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vis_i, vis_i, NX * NY * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDim(32, 32);
    dim3 gridDim((NX + blockDim.x - 1) / blockDim.x, (NY + blockDim.y - 1) / blockDim.y);

    // Set initial conditions
    initial_condition(blockDim, gridDim, d_Q, rh_floor_host, Z_host, T_floor_host, aindex_host, bapp_host, b0_host, NX, NY, NQ);
    cudaMemcpy(Q, d_Q, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
    write_vtk("output", Q, flux_x, NX, NY, NQ, nout);  // Add timestep to the filename

    if (true) {
    while (nprint <= 100) {
      get_min_dt(&dt, 0.5, dxi, clt_host);
      // printf("lamb ... %e\n", lamb_host);

      dxt = dt * dxi;
      dyt = dt * dyi;

      if (iorder == 1) {
        limit_flow_kernel<<<gridDim, blockDim>>>(d_Q, NX, NY, NQ);
        cudaDeviceSynchronize();
        get_sources_kernel<<<gridDim, blockDim>>>(d_Q, d_sources, d_eta, d_Tiev, d_Teev, d_nuei, d_kap_i, d_kap_e, d_vis_i, NX, NY, NQ, dt, dxi);
        cudaDeviceSynchronize();
        get_flux(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, 0.75, NX, NY, NQ);
        advance_time_level_rz(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, d_sources, d_Q, dxt, dyt, dt, NX, NY, NQ);
        implicit_source2(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, d_eta, d_Q, dxi, dyi, dt, NX, NY, NQ);
        set_bc(blockDim, gridDim, d_Q, t + dt, dxi, dyi, k_las_host, f_las_host, NX, NY, NQ);

        cudaMemcpy(Q, d_Q, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_x, d_flux_x, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_y, d_flux_y, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);

      }

      if (iorder == 2) {
        get_sources_kernel<<<gridDim, blockDim>>>(d_Q, d_sources, d_eta, d_Tiev, d_Teev, d_nuei, d_kap_i, d_kap_e, d_vis_i, NX, NY, NQ, dt, dxi);
        cudaDeviceSynchronize();
        get_flux(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, 1, NX, NY, NQ);
        advance_time_level_rz(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, d_sources, d_Q1, dxt, dyt, dt, NX, NY, NQ);
        implicit_source2(blockDim, gridDim, d_Q1, d_flux_x, d_flux_y, d_eta, d_Q1, dxi, dyi, dt, NX, NY, NQ);
        limit_flow_kernel<<<gridDim, blockDim>>>(d_Q1, NX, NY, NQ);
        cudaDeviceSynchronize();
        set_bc(blockDim, gridDim, d_Q1, t + dt, dxi, dyi, k_las_host, f_las_host, NX, NY, NQ);

        get_sources_kernel<<<gridDim, blockDim>>>(d_Q1, d_sources, d_eta, d_Tiev, d_Teev, d_nuei, d_kap_i, d_kap_e, d_vis_i, NX, NY, NQ, dt, dxi);
        cudaDeviceSynchronize();
        get_flux(blockDim, gridDim, d_Q1, d_flux_x, d_flux_y, 1, NX, NY, NQ);
        advance_time_level_rz(blockDim, gridDim, d_Q1, d_flux_x, d_flux_y, d_sources, d_Q2, dxt, dyt, dt, NX, NY, NQ);
        implicit_source2(blockDim, gridDim, d_Q2, d_flux_x, d_flux_y, d_eta, d_Q2, dxi, dyi, dt, NX, NY, NQ);
        limit_flow_kernel<<<gridDim, blockDim>>>(d_Q2, NX, NY, NQ);
        cudaDeviceSynchronize();

        // Now add Q and Q2 together 
        add_Q_kernel<<<gridDim, blockDim>>>(d_Q, d_Q2, d_Q, NX, NY, NQ);
        cudaDeviceSynchronize();
        limit_flow_kernel<<<gridDim, blockDim>>>(d_Q, NX, NY, NQ);
        cudaDeviceSynchronize();
        set_bc(blockDim, gridDim, d_Q, t + dt, dxi, dyi, k_las_host, f_las_host, NX, NY, NQ);

        cudaMemcpy(Q, d_Q, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_x, d_flux_x, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_y, d_flux_y, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
      }

      niter++;
      t += dt;
      if (niter % 10 == 0) {
        printf("\nIteration time: %f seconds\n", (float)clock() / CLOCKS_PER_SEC);
        printf("nout=%d\n", nout);
        printf("t= %e\n dt= %e\n niter= %d\n", t * 100, dt*100, niter);
        printf("dxt= %e, dyt= %e\n", dxi*dt, dyi*dt, niter);
        // printf("lambda= %e, test= %e, NX= %d, dx= %e, dy= %e\n", lamb_host, lamb_host/NX, NX, dx, dy);
        nprint++;

        check_Iv(NX - 1 / dxi, NY / 2);
        cudaDeviceSynchronize();

        write_vtk("output", Q, flux_y, NX, NY, NQ, nout+1);  // Add timestep to the filename
        nout++;
      }
    }
  }
  free(Q);
  free(flux_x);
  free(flux_y);
  free(sources);
  free(Q1);
  free(Q2);
  free(Q3);
  free(Q4);
  free(eta);
  cudaFree(d_Q);
  cudaFree(d_flux_x);
  cudaFree(d_flux_y);
  cudaFree(d_sources);
  cudaFree(d_Q1);
  cudaFree(d_Q2);
  cudaFree(d_Q3);
  cudaFree(d_Q4);
  cudaFree(d_eta);

    return 0;
}
