#include <cuda_runtime.h>
#include <stdio.h>
#include "constants.h"

// Function prototypes
__global__ void calc_flux_x_kernel(float *Qx, float *cfx, float *ffx, int nx, int nq);
__global__ void calc_flux_y_kernel(float *Qy, float *cfy, float *ffy, int ny, int nq);
__global__ void compute_wr_wl(float *Qin, float *ff, float *cfr, float *wr, float *wl, int n, int nq, int pr);
__global__ void compute_fr_fl(float *wr, float *wl, float *fr, float *fl, int n, int nq);
__global__ void compute_dfr_dfl(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, int n, int nq);
__global__ void compute_flux2(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, float *flux2, float sl, int n, int nq);
__global__ void advance_time_level_rz_kernel(float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float dxt, float dyt, float dt, int nx, int ny) ;
__global__ void limit_flow(float *Qin, float rh_floor, float T_floor, float aindex, float Z, float vhcf, int nx, int ny, int nq);
__device__ float eta_s(float e_Temp, float Za, float dne, float cln_min, int tp);
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

KTYPE PLas;
KTYPE Emax;
KTYPE Bmax;
KTYPE focal_length;
KTYPE w0;

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

    PLas=1.e17;
    Emax=sqrt(2.*t0_host*PLas/(clt_host*L0_host*eps0_host));
    Bmax=(t0_host*Emax/(L0_host*clt_host));
    Emax=Emax/e0_host;
    Bmax=Bmax/b0_host;
    focal_length = 12.0*lamb_host/3.0;
    w0 = 1.0*lamb_host;

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


__global__ void advance_time_level_rz_kernel_sm(float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float mdxt, float mdyt, float dt, int nx, int ny, int nq) {
  // Calculate shared memory size based on the number of threads and nq
  extern __shared__ float sharedMem[];

  // Allocate shared memory with halo cells (1 cell padding in each direction)
  int haloSizeX = blockDim.x + 2; // +2 for the halo cells on both sides
  int haloSizeY = blockDim.y + 2; // +2 for the halo cells on both sides
  float *shQin = &sharedMem[0];
  float *shFluxX = &sharedMem[haloSizeX * haloSizeY * nq];
  float *shFluxY = &sharedMem[2 * haloSizeX * haloSizeY * nq];
  float *shSource = &sharedMem[3 * haloSizeX * haloSizeY * nq];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Calculate thread indices within the shared memory block
  int sh_i = threadIdx.x + 1; // Offset by 1 to account for halo cells
  int sh_j = threadIdx.y + 1; // Offset by 1 to account for halo cells
  int threadIdx2D = sh_j * haloSizeX + sh_i;

  // Load the main block data into shared memory, including halo cells
  for (int k = 0; k < nq; ++k) {
      int globalIndex = i * (ny * nq) + j * nq + k;

      // Load the central block
      shQin[threadIdx2D * nq + k] = (i < nx && j < ny) ? Qin[globalIndex] : 0.0f;
      shFluxX[threadIdx2D * nq + k] = (i < nx && j < ny) ? flux_x[globalIndex] : 0.0f;
      shFluxY[threadIdx2D * nq + k] = (i < nx && j < ny) ? flux_y[globalIndex] : 0.0f;
      shSource[threadIdx2D * nq + k] = (i < nx && j < ny) ? source[globalIndex] : 0.0f;

      // Load halo cells
      if (threadIdx.x == 0 && i > 0) {
          shQin[(sh_j * haloSizeX) * nq + (sh_i - 1) * nq + k] = Qin[(i-1) * (ny * nq) + j * nq + k];
          shFluxX[(sh_j * haloSizeX) * nq + (sh_i - 1) * nq + k] = flux_x[(i-1) * (ny * nq) + j * nq + k];
          shFluxY[(sh_j * haloSizeX) * nq + (sh_i - 1) * nq + k] = flux_y[(i-1) * (ny * nq) + j * nq + k];
      }
      if (threadIdx.x == blockDim.x - 1 && i < nx-1) {
          shQin[(sh_j * haloSizeX) * nq + (sh_i + 1) * nq + k] = Qin[(i+1) * (ny * nq) + j * nq + k];
          shFluxX[(sh_j * haloSizeX) * nq + (sh_i + 1) * nq + k] = flux_x[(i+1) * (ny * nq) + j * nq + k];
          shFluxY[(sh_j * haloSizeX) * nq + (sh_i + 1) * nq + k] = flux_y[(i+1) * (ny * nq) + j * nq + k];
      }
      if (threadIdx.y == 0 && j > 0) {
          shQin[((sh_j - 1) * haloSizeX) * nq + sh_i * nq + k] = Qin[i * (ny * nq) + (j-1) * nq + k];
          shFluxX[((sh_j - 1) * haloSizeX) * nq + sh_i * nq + k] = flux_x[i * (ny * nq) + (j-1) * nq + k];
          shFluxY[((sh_j - 1) * haloSizeX) * nq + sh_i * nq + k] = flux_y[i * (ny * nq) + (j-1) * nq + k];
      }
      if (threadIdx.y == blockDim.y - 1 && j < ny-1) {
          shQin[((sh_j + 1) * haloSizeX) * nq + sh_i * nq + k] = Qin[i * (ny * nq) + (j+1) * nq + k];
          shFluxX[((sh_j + 1) * haloSizeX) * nq + sh_i * nq + k] = flux_x[i * (ny * nq) + (j+1) * nq + k];
          shFluxY[((sh_j + 1) * haloSizeX) * nq + sh_i * nq + k] = flux_y[i * (ny * nq) + (j+1) * nq + k];
      }
  }

  // Synchronize to ensure all data is loaded into shared memory
  __syncthreads();

  // Perform the calculations using shared memory with halo cells
  if (i >= 2 && i < nx-1 && j >= 2 && j < ny-1) {
      float rbp = 0.5f * (rc(i+1) + rc(i));
      float rbm = 0.5f * (rc(i) + rc(i-1));
      float rci = 1.0f / rc(i);

      for (int k = 0; k < nq; ++k) {
          int idx_i_j = threadIdx2D * nq + k;
          int idx_im1_j = ((sh_i - 1) * haloSizeX + sh_j) * nq + k;
          int idx_ip1_j = ((sh_i + 1) * haloSizeX + sh_j) * nq + k;
          int idx_i_jm1 = (sh_i * haloSizeX + (sh_j - 1)) * nq + k;
          int idx_i_jp1 = (sh_i * haloSizeX + (sh_j + 1)) * nq + k;

          Qinp1[i * (ny * nq) + j * nq + k] = shQin[idx_i_j]
                                              - mdxt * (shFluxX[idx_ip1_j] * rbp - shFluxX[idx_im1_j] * rbm)
                                              - mdyt * (shFluxY[idx_i_jp1] - shFluxY[idx_i_jm1]) * rci
                                              + dt * shSource[idx_i_j];

          if (isnan(Qinp1[i * (ny * nq) + j * nq + k])) {
              printf("NaN encountered in cell i=%d j=%d %e. Exiting...\n", i, j, Qinp1[i * (ny * nq) + j * nq + k]);
              return;
          }

          Qinp1[i * (ny * nq) + j * nq + k] *= rci;
      }
  }
}

__global__ void advance_time_level_rz_kernel(float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float mdxt, float mdyt, float dt, int nx, int ny, int nq) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // float rbp = 0.5f * (rc(i+1) + rc(i));
  // float rbm = 0.5f * (rc(i) + rc(i-1));
  // float rci = 1.0f / rc(i);
  //   if (i >= 2 && i < nx-1 && j >= 2 && j < ny-1) {

  // for (int k = 0; k < nq; ++k) {
  //   int idxnQ = i* (ny * nq) + j* nq +k;
  //   int idxnQ_ip1 = (i+1)* (ny * nq) + j* nq+k ;
  //   int idxnQ_im1 = (i-1)* (ny * nq) + j* nq +k;
  //   int idxnQ_jp1 = i* (ny * nq) + (j+1)* nq +k;
  //   int idxnQ_jm1 = i* (ny * nq) + (j-1)* nq +k;

  //   Qinp1[i * (ny * nq) + j * nq + k] = Qin[idxnQ]
  //   - mdxt * (flux_x[idxnQ] * rbp - flux_x[idxnQ_im1] * rbm)
  //   - mdyt * (flux_y[idxnQ] - flux_y[idxnQ_jm1]) * rci
  //   + dt * source[idxnQ];

  //   if (isnan(Qinp1[i * (ny * nq) + j * nq + k])) {
  //   printf("NaN encountered in cell i=%d j=%d %e. Exiting...\n", i, j, Qinp1[i * (ny * nq) + j * nq + k]);
  //   return;
  //   }
  //   }
  // }

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
      printf("NaN encountered in cell i=%d j=%d %e. Exiting...\n", i, j, Qinp1[idxnQ + rh]);
      return;
    }

    for (int k = rh; k <= ne; ++k) {
      Qinp1[idxnQ + k] *= rci;
    }
  }
}

void advance_time_level_rz(dim3 bs, dim3 gs, float *Qin, float *flux_x, float *flux_y, float *source, float *Qinp1, float mdxt, float mdyt, float dt, int nx, int ny, int nq) {

  if (1==2) {
    int blockSizeX = 8; // Example block size, adjust as needed
    int blockSizeY = 8;
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 numBlocks((nx + blockSizeX - 1) / blockSizeX, (ny + blockSizeY - 1) / blockSizeY);
    int sharedMemSize = 4 * (blockSizeX +2)* (blockSizeY+2) * nq * sizeof(float);
    printf("Shared memory required per block: %d bytes\n", sharedMemSize);
    advance_time_level_rz_kernel_sm<<<numBlocks, threadsPerBlock, sharedMemSize>>>(Qin, flux_x, flux_y, source, Qinp1, mdxt, mdyt, dt, nx, ny, nq);
  }
  else {
    advance_time_level_rz_kernel<<<gs, bs>>>(Qin, flux_x, flux_y, source, Qinp1, mdxt, mdyt, dt, nx, ny, nq);
  }

  // Synchronize to check for any kernel launch errors
  cudaDeviceSynchronize();
}

__global__ void limit_flow(float *Qin, float rh_floor, float T_floor, float aindex, float Z, float vhcf, int nx, int ny, int nq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idxnQ = i* (ny * nq) + j* nq ;

    if (i < nx && j < ny) {
      float en_floor = rh_floor * T_floor / (aindex - 1.0);

      if (Qin[idxnQ + rh] <= rh_floor || Qin[idxnQ + ne] <= Z * rh_floor) {
          Qin[idxnQ + rh] = rh_floor;
          Qin[idxnQ + ne] = Z * rh_floor;
          Qin[idxnQ + mx] = 0.0;
          Qin[idxnQ + my] = 0.0;
          Qin[idxnQ + mz] = 0.0;
          Qin[idxnQ + jx] = 0.0;
          Qin[idxnQ + jy] = 0.0;
          Qin[idxnQ + jz] = 0.0;
          Qin[idxnQ + en] = en_floor;
          Qin[idxnQ + et] = Z * en_floor;
      }

      if (abs(Qin[idxnQ + jx]) > vhcf * Qin[idxnQ + rh]) {
          Qin[idxnQ + jx] = vhcf * Qin[idxnQ + rh] * copysign(1.0, Qin[idxnQ + jx]);
      }
      if (abs(Qin[idxnQ + jy]) > vhcf * Qin[idxnQ + rh]) {
          Qin[idxnQ + jy] = vhcf * Qin[idxnQ + rh] * copysign(1.0, Qin[idxnQ + jy]);
      }
      if (abs(Qin[idxnQ + jz]) > vhcf * Qin[idxnQ + rh]) {
          Qin[idxnQ + jz] = vhcf * Qin[idxnQ + rh] * copysign(1.0, Qin[idxnQ + jz]);
      }
  }

}

__global__ void set_bc_kernel(float *Qin, float t, float dxi, float dyi, float k_las, float f_las, float Emax, float Bmax, int nx, int ny, int nq, float focus_distance, float w0) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idxnQ = i* (ny * nq) + j* nq ;

  // Laser beam parameters
  float lambda = 2.0f * pi / k_las;  // Wavelength
  float focus_x =0.5f * xc(nx , dxi);  // Focus in the center of the domain along x-axis
  float focus_y = 0.0f;  // Focus in the center of the domain along y-axis
  float z0 =focus_distance;        // Focal point distance from the boundary
  float z_R = pi * w0 * w0 / lambda; // Rayleigh range

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
          // Qin[i * ny * nq + j * nq + by] = -1.0f*Bmax*cos(k_las * xc(i, dxi) - f_las * t);
          // Qin[i * ny * nq + j * nq + ez] = 1.0f*Emax*cos(k_las * xc(i, dxi) - f_las * t);

          // Calculate spatial phase shift for focusing
          float x = xc(i, dxi);
          float y = yc(j , dyi);
          float hlyu = yc(ny, dyi)/2.0;  // Focus in the center of the domain along y-axis
          y = y - hlyu;  // Shift the focus to the center of the domain along y-axis
          float z = x-z0;
          // Parabolic phase shift for focusing
          float r_squared = (y - focus_y) * (y - focus_y);
           
          // Beam width at distance z (along the x-axis)
          float wz = w0 * sqrtf(1.0f + (z / z_R) * (z/ z_R));  // Beam width as a function of distance
          float gaussian_envelope = (w0/wz)*expf(-((y - focus_y) * (y - focus_y)) / (wz * wz));
          float Rz = z * (1.0f + (z_R * z_R) / (z * z));  // Radius of curvature of the wavefront
          float phase_shift = (k_las * r_squared) / (2.0f * Rz);  // Parabolic phase shift for focusing

          // Apply Gaussian envelope and phase shift to the laser boundary conditions
          Qin[i * ny * nq + j * nq + by] = -1.0f * Bmax * gaussian_envelope * cos(k_las * x - f_las * t + phase_shift);
          Qin[i * ny * nq + j * nq + ez] = 1.0f * Emax * gaussian_envelope * cos(k_las * x - f_las * t + phase_shift);
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
          // Qin[i * ny * nq + j * nq + bz] = 0.1f*cos(k_las * yc(j, dyi) - f_las * t);
          // Qin[i * ny * nq + j * nq + ex] = -cab*0.1f*cos(k_las * yc(j, dyi) - f_las * t);
        }
    }
  }
  
}

void set_bc(dim3 bs, dim3 gs, float *Qin, float t, float dxi, float dyi, float k_las, float f_las, float Emax, float Bmax, int nx, int ny, int nq, float focus_distance, float w0) {
  set_bc_kernel<<<gs, bs>>>(Qin, t, dxi, dyi, k_las, f_las, Emax, Bmax, nx, ny, nq, focus_distance, w0);
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

  if (i < nx - 2 && j < ny - 2 && j > 1 && i > 1) {
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
      // ma = 1.0f + dt * gma2 * dne * (0.1 + dt * cfva2);

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
      Qjz = Qin[i * ny * nq + j * nq + jz] + dt * gma2 * (dne * (Qin[i * ny * nq + j * nq + ez] + ux * hy - uy * hx));

      Qout[idxnQ + jx] = Ainv[0][0] * Qjx + Ainv[0][1] * Qjy + Ainv[0][2] * Qjz;
      Qout[idxnQ + jy] = Ainv[1][0] * Qjx + Ainv[1][1] * Qjy + Ainv[1][2] * Qjz;
      Qout[idxnQ + jz] = Ainv[2][0] * Qjx + Ainv[2][1] * Qjy + Ainv[2][2] * Qjz;

      // if (i == 3 && j == 100) {
      //   printf("Before implicit source\n");
      //   printf("dt= %e\n", dt );
      //   printf("gma2= %e\n", gma2 );
      //   printf("dne= %e\n", dne );
      //   printf("cfva2= %e\n", cfva2 );
      //   printf("denom= %e\n", denom );
      //   printf("ma2= %e\n", ma2 );
      //   printf("mb2= %e\n", mb2 );
      //   printf("ecyc= %e\n", ecyc );

      //   // printf("dt= %e\n", dt );
      //   // printf("cfva2= %e\n", cfva2 );
      //   // printf("dt*cfva2 = %e\n", dt * cfva2);
      //   // printf("Qout[%d][%d][%d] = %e\n", i, j, ey, Qout[idxnQ + ey]);
      //   printf("Qout[%d][%d][%d] = %e\n", i, j, jy, Qout[idxnQ + jz]);
      // }

      Qout[idxnQ + ex] = Qin[idxnQ + ex] - dt * cfva2 * Qout[idxnQ + jx];
      Qout[idxnQ + ey] = Qin[idxnQ + ey] - dt * cfva2 * Qout[idxnQ + jy];
      Qout[idxnQ + ez] = Qin[idxnQ + ez] - dt * cfva2 * Qout[idxnQ + jz];

      // if (i == 3 && j == 100) {
      //   printf("After implicit source\n");
      //   printf("Qout[%d][%d][%d] = %e\n", i, j, ey, Qout[idxnQ + ez]);
      //   printf("Qout[%d][%d][%d] = %e\n", i, j, jy, Qout[idxnQ + jz]);
      // }
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

void implicit_source2(dim3 bs, dim3 gs, float* Qin, float* flux_x, float* flux_y, float* eta, float* Qout, float dxt, float dyt, float dt, int nx, int ny, int nq) {
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
  }
}

// __global__ void copy3DTo2DSliceX(float *d_Qin, float *d_Qx, int nx, int ny, int nq, int j) {
//   // Determine the size of the shared memory array
//   extern __shared__ float shared_Qin[];

//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int q = blockIdx.y * blockDim.y + threadIdx.y;

//   // Calculate the 1D thread index within the block
//   int threadIdx1D = threadIdx.y * blockDim.x + threadIdx.x;

//   // Each thread loads its corresponding value from global memory into shared memory
//   if (i < nx && q < nq) {
//       int globalIdx = i * ny * nq + j * nq + q;
//       shared_Qin[threadIdx1D] = d_Qin[globalIdx];
//   }

//   // Synchronize threads to ensure all data is loaded into shared memory
//   __syncthreads();

//   // Use the shared memory to copy the data to d_Qx
//   if (i < nx && q < nq) {
//       int idx2D = i * nq + q;
//       d_Qx[idx2D] = shared_Qin[threadIdx1D];
//   }
// }

__global__ void copy3DTo2DSliceY(float *d_Qin, float *d_Qy, int nx, int ny, int nq, int i) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < nx && q < nq) {
    
      d_Qy[j * nq + q] = d_Qin[i * ny * nq + j * nq + q];
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

// // Function to perform get_flux operations
// void get_flux(dim3 bs, dim3 gs, float *d_Qin, float *d_flux_x, float *d_flux_y, float sl, int nx, int ny, int nq) {
//   size_t size_3d = nx * ny * nq * sizeof(float);
//   size_t size_2d_x = nx * nq * sizeof(float);
//   size_t size_2d_y = ny * nq * sizeof(float);

//   // Define grid and block dimensions
//   dim3 bs1d(16);
//   dim3 gridDimX((nx + bs1d.x - 1) / bs1d.x, 1);
//   dim3 gridDimY((ny + bs1d.x - 1) / bs1d.x, 1);

//   dim3 gridDimXQ((nx + bs.x - 1) / bs.x, (nq + bs.y - 1) / bs.y);
//   dim3 gridDimYQ((ny + bs.x - 1) / bs.x, (nq + bs.y - 1) / bs.y);

//   // Allocate memory on the device
//   float *d_Qx, *d_Qy, *d_cfx, *d_ffx, *d_fx, *d_cfy, *d_ffy, *d_fy;
//   float *d_wr, *d_wl, *d_fr, *d_fl, *d_dfrp, *d_dfrm, *d_dflp, *d_dflm;
//   float *d_wry, *d_wly, *d_fry, *d_fly, *d_dfrpy, *d_dfrmy, *d_dflpy, *d_dflmy;
//   int flag;
//   cudaMalloc(&d_Qx, size_2d_x);
//   cudaMalloc(&d_Qy, size_2d_y);
//   cudaMalloc(&d_cfx, size_2d_x);
//   cudaMalloc(&d_ffx, size_2d_x);
//   cudaMalloc(&d_fx, size_2d_x);
//   cudaMalloc(&d_cfy, size_2d_y);
//   cudaMalloc(&d_ffy, size_2d_y);
//   cudaMalloc(&d_fy, size_2d_y);
//   cudaMalloc(&d_wr, size_2d_x);
//   cudaMalloc(&d_wl, size_2d_x);
//   cudaMalloc(&d_fr, size_2d_x);
//   cudaMalloc(&d_fl, size_2d_x);
//   cudaMalloc(&d_dfrp, size_2d_x);
//   cudaMalloc(&d_dfrm, size_2d_x);
//   cudaMalloc(&d_dflp, size_2d_x);
//   cudaMalloc(&d_dflm, size_2d_x);

//   cudaMalloc(&d_wry, size_2d_y);
//   cudaMalloc(&d_wly, size_2d_y);
//   cudaMalloc(&d_fry, size_2d_y);
//   cudaMalloc(&d_fly, size_2d_y);
//   cudaMalloc(&d_dfrpy, size_2d_y);
//   cudaMalloc(&d_dfrmy, size_2d_y);
//   cudaMalloc(&d_dflpy, size_2d_y);
//   cudaMalloc(&d_dflmy, size_2d_y);
//   flag = 0;


//   // Create CUDA streams for concurrent kernel execution
//   cudaStream_t stream1, stream2;
//   cudaStreamCreate(&stream1);
//   cudaStreamCreate(&stream2);

//   // Launch kernels for each slice in the y-dimension
//   for (int j = 0; j < ny; j++) {
//     // Copy the slice from the 3D array to the 2D array
//     // cudaMemcpy(d_Qx, d_Qin + j * nx * nq, size_2d_x, cudaMemcpyDeviceToDevice);
//     copy3DTo2DSliceX<<<gridDimXQ, bs, 0, stream1>>>(d_Qin, d_Qx, nx, ny, nq, j);
//     cudaDeviceSynchronize();
    
//     // Launch the kernel to copy the j-th slice to d_Qx
//     // Launch calc_flux_x_kernel
//     calc_flux_x_kernel<<<gridDimX, bs1d, 0, stream1>>>(d_Qx, d_cfx, d_ffx, nx, nq);
//     cudaDeviceSynchronize();

//     // Launch TVD kernels
//     compute_wr_wl<<<gridDimXQ, bs, 0, stream1>>>(d_Qx, d_ffx, d_cfx, d_wr, d_wl, nx, nq, flag);
//     cudaDeviceSynchronize();

//     compute_fr_fl<<<gridDimXQ, bs, 0, stream1>>>(d_wr, d_wl, d_fr, d_fl, nx, nq);
//     cudaDeviceSynchronize();

//     compute_dfr_dfl<<<gridDimXQ, bs, 0, stream1>>>(d_fr, d_fl, d_dfrp, d_dfrm, d_dflp, d_dflm, nx, nq);
//     cudaDeviceSynchronize();

//     compute_flux2<<<gridDimXQ, bs, 0, stream1>>>(d_fr, d_fl, d_dfrp, d_dfrm, d_dflp, d_dflm, d_fx, sl, nx, nq);
//     cudaDeviceSynchronize();

//     // Copy results back to the corresponding slice in flux_x
//     // cudaMemcpy(d_flux_x + j * nx * nq, d_fx, size_2d_x, cudaMemcpyDeviceToDevice);
//     copy2DTo3DSliceX<<<gridDimXQ, bs, 0, stream1>>>(d_fx, d_flux_x, nx, ny, nq, j);
//     cudaDeviceSynchronize();

//   }

//   // Launch kernels for each slice in the x-dimension
//   for (int i = 0; i < nx; i++) {
//     // // Copy the slice from the 3D array to the 2D array
//     // cudaMemcpy(d_Qy, d_Qin + i * ny * nq, size_2d_y, cudaMemcpyDeviceToDevice);

//     // Launch the kernel to copy the j-th slice to d_Qx
//     copy3DTo2DSliceY<<<gridDimYQ, bs, 0, stream2>>>(d_Qin, d_Qy, nx, ny, nq, i);
//     cudaDeviceSynchronize();

//     // Launch calc_flux_y_kernel
//     calc_flux_y_kernel<<<gridDimY, bs1d, 0, stream2>>>(d_Qy, d_cfy, d_ffy, ny, nq);
//     cudaDeviceSynchronize();

//     // // Launch TVD kernels
//     if ( i == 4) {
//       flag = 1;
//     }
//     else {
//       flag = 0;
//     }
//     compute_wr_wl<<<gridDimYQ, bs, 0, stream2>>>(d_Qy, d_ffy, d_cfy, d_wry, d_wly, ny, nq, flag);
//     cudaDeviceSynchronize();

//     compute_fr_fl<<<gridDimYQ, bs, 0, stream2>>>(d_wry, d_wly, d_fry, d_fly, ny, nq);
//     cudaDeviceSynchronize();

//     compute_dfr_dfl<<<gridDimYQ, bs, 0, stream2>>>(d_fry, d_fly, d_dfrpy, d_dfrmy, d_dflpy, d_dflmy, ny, nq);
//     cudaDeviceSynchronize();

//     compute_flux2<<<gridDimYQ, bs, 0, stream2>>>(d_fry, d_fly, d_dfrpy, d_dfrmy, d_dflpy, d_dflmy, d_fy, sl, ny, nq);
//     cudaDeviceSynchronize();

//     // Copy results back to the corresponding slice in flux_y
//     // cudaMemcpy(d_flux_y + i * ny * nq, d_fy, size_2d_y, cudaMemcpyDeviceToDevice);
//     copy2DTo3DSliceY<<<gridDimYQ, bs, 0, stream2>>>(d_fy, d_flux_y, nx, ny, nq, i);
//     cudaDeviceSynchronize();
//   }
//   // Clean up CUDA streams
//   cudaStreamDestroy(stream1);
//   cudaStreamDestroy(stream2);

//   // Free device memory
//   cudaFree(d_Qx);
//   cudaFree(d_Qy);
//   cudaFree(d_cfx);
//   cudaFree(d_ffx);
//   cudaFree(d_fx);
//   cudaFree(d_cfy);
//   cudaFree(d_ffy);
//   cudaFree(d_fy);

//   cudaFree(d_wr);
//   cudaFree(d_wl);
//   cudaFree(d_fr);
//   cudaFree(d_fl);
//   cudaFree(d_dfrp);
//   cudaFree(d_dfrm);
//   cudaFree(d_dflp);
//   cudaFree(d_dflm);

//   cudaFree(d_wry);
//   cudaFree(d_wly);
//   cudaFree(d_fry);
//   cudaFree(d_fly);
//   cudaFree(d_dfrpy);
//   cudaFree(d_dfrmy);
//   cudaFree(d_dflpy);
//   cudaFree(d_dflmy);

// }

// Function to perform get_flux operations
void get_flux(dim3 bs, dim3 gs, int nStreams, float *d_Qin, float *d_flux_x, float *d_flux_y, float sl, int nx, int ny, int nq) {
  size_t size_3d = nx * ny * nq * sizeof(float);
  size_t size_2d_x = nx * nq * sizeof(float);
  size_t size_2d_y = ny * nq * sizeof(float);

  // Define grid and block dimensions
  dim3 bs1d(16);
  dim3 gridDimX((nx + bs1d.x - 1) / bs1d.x, 1);
  dim3 gridDimY((ny + bs1d.x - 1) / bs1d.x, 1);

  dim3 gridDimXQ((nx + bs.x - 1) / bs.x, (nq + bs.y - 1) / bs.y);
  dim3 gridDimYQ((ny + bs.x - 1) / bs.x, (nq + bs.y - 1) / bs.y);

  // The size of the shared memory array in bytes
  int sharedMemSize = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

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


// Create CUDA streams for concurrent kernel execution
cudaStream_t *streams = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
for (int i = 0; i < nStreams ; i++) {
    cudaStreamCreate(&streams[i]);
}

// Launch kernels for each slice in the y-dimension
for (int j = 0; j < ny; j++) {
    int streamIdx = j % nStreams;

    // Copy the slice from the 3D array to the 2D array
    copy3DTo2DSliceX<<<gridDimXQ, bs, sharedMemSize, streams[streamIdx]>>>(d_Qin, d_Qx, nx, ny, nq, j);

    // Launch calc_flux_x_kernel
    calc_flux_x_kernel<<<gridDimX, bs1d, 0, streams[streamIdx]>>>(d_Qx, d_cfx, d_ffx, nx, nq);

    // Launch TVD kernels
    compute_wr_wl<<<gridDimXQ, bs, 0, streams[streamIdx]>>>(d_Qx, d_ffx, d_cfx, d_wr, d_wl, nx, nq, flag);
    compute_fr_fl<<<gridDimXQ, bs, 0, streams[streamIdx]>>>(d_wr, d_wl, d_fr, d_fl, nx, nq);
    compute_dfr_dfl<<<gridDimXQ, bs, 0, streams[streamIdx]>>>(d_fr, d_fl, d_dfrp, d_dfrm, d_dflp, d_dflm, nx, nq);
    compute_flux2<<<gridDimXQ, bs, 0, streams[streamIdx]>>>(d_fr, d_fl, d_dfrp, d_dfrm, d_dflp, d_dflm, d_fx, sl, nx, nq);

    // Copy results back to the corresponding slice in flux_x
    copy2DTo3DSliceX<<<gridDimXQ, bs, 0, streams[streamIdx]>>>(d_fx, d_flux_x, nx, ny, nq, j);
}

// Synchronize all streams for y-dimension operations
for (int i = 0; i < nStreams; i++) {
    cudaStreamSynchronize(streams[i]);
}

// Launch kernels for each slice in the x-dimension
for (int i = 0; i < nx; i++) {
    int streamIdx = i % nStreams;

    // Copy the slice from the 3D array to the 2D array
    copy3DTo2DSliceY<<<gridDimYQ, bs, 0, streams[streamIdx]>>>(d_Qin, d_Qy, nx, ny, nq, i);

    // Launch calc_flux_y_kernel
    calc_flux_y_kernel<<<gridDimY, bs1d, 0, streams[streamIdx]>>>(d_Qy, d_cfy, d_ffy, ny, nq);

    // Launch TVD kernels
    flag = (i == 4) ? 1 : 0;
    compute_wr_wl<<<gridDimYQ, bs, 0, streams[streamIdx]>>>(d_Qy, d_ffy, d_cfy, d_wry, d_wly, ny, nq, flag);
    compute_fr_fl<<<gridDimYQ, bs, 0, streams[streamIdx]>>>(d_wry, d_wly, d_fry, d_fly, ny, nq);
    compute_dfr_dfl<<<gridDimYQ, bs, 0, streams[streamIdx]>>>(d_fry, d_fly, d_dfrpy, d_dfrmy, d_dflpy, d_dflmy, ny, nq);
    compute_flux2<<<gridDimYQ, bs, 0, streams[streamIdx]>>>(d_fry, d_fly, d_dfrpy, d_dfrmy, d_dflpy, d_dflmy, d_fy, sl, ny, nq);

    // Copy results back to the corresponding slice in flux_y
    copy2DTo3DSliceY<<<gridDimYQ, bs, 0, streams[streamIdx]>>>(d_fy, d_flux_y, nx, ny, nq, i);
}

// Synchronize all streams for x-dimension operations
for (int i = 0; i < nStreams; i++) {
    cudaStreamSynchronize(streams[i]);
}

// Clean up CUDA streams
for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
}

free(streams);

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
  if (i < 0) return;

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
}

__global__ void calc_flux_y_kernel(float* Qy, float* cfy, float* fly, int ny, int nq) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = j* nq;

  if (j >= ny) return;
  if (j < 0) return;

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
  
}

__global__ void compute_wr_wl(float *Qin, float *ff, float *cfr, float *wr, float *wl, int n, int nq, int tp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < n && q < nq ) {
    int idx = i * nq + q;
    wr[idx] = cfr[idx] * Qin[idx] + ff[idx];
    wl[idx] = cfr[idx] * Qin[idx] - ff[idx];
  }
}

__global__ void compute_fr_fl(float *wr, float *wl, float *fr, float *fl, int n, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < n && q < nq ) {
    int idx = i * nq + q;
    fr[idx] = wr[idx];
    fl[idx] = wl[((i + 1) % n) * nq + q];

  }
}

__global__ void compute_dfr_dfl(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, int n, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && q < nq ) {
    int idx = i * nq + q;
    dfrp[idx] = fr[((i + 1) % n) * nq + q] - fr[idx];
    dfrm[idx] = fr[idx] - fr[((i - 1 + n) % n) * nq + q];
    dflp[idx] = fl[idx] - fl[((i + 1) % n) * nq + q];
    dflm[idx] = fl[((i - 1 + n) % n) * nq + q] - fl[idx];

  }
}

__global__ void compute_flux2(float *fr, float *fl, float *dfrp, float *dfrm, float *dflp, float *dflm, float *flux2, float sl, int n, int nq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && q < nq ) {
    int idx = i * nq + q;
    
    float dfr, dfl;
    float epsilon = 1e-10f;

    if (dfrp[idx] * dfrm[idx] > 0) {
      // dfr = dfrp[idx] * dfrm[idx] / (dfrp[idx] + dfrm[idx]);
      float denom = dfrp[idx] + dfrm[idx];
      dfr = denom > epsilon ? (dfrp[idx] * dfrm[idx]) / denom : 0.0f;
    } else {
      dfr = 0.0f;
    }

    if (dflp[idx] * dflm[idx] > 0) {
      float denom = dflp[idx] + dflm[idx];
      dfl = denom > epsilon ? (dflp[idx] * dflm[idx]) / denom : 0.0f;
      // dfl = dflp[idx] * dflm[idx] / (dflp[idx] + dflm[idx]);
    } else {
      dfl = 0.0f;
    }

    flux2[idx] = 0.5f * (fr[idx] - fl[idx] + sl * (dfr - dfl));
    // if (flux2[idx] > 100) {
    //   printf("i: %d, q: %d, flux2: %f\n", i, q, flux2[idx]);
    // }
  }
}

__global__ void get_sources_kernel(
  float* Qin, float* sourcesin, float* eta_in, float* Tiev, float* Teev, float* nuei, float* kap_i, float* kap_e, float* vis_i, int nx, int ny, int nq, float dt, float dxi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idxnQ = i* (ny * nq) + j* nq ;
  int idx = i* (ny * 1) + j* 1;

  if (i < nx && j < ny && i > 0 && j > 0) {
    float ri, vx, vy, vz, P, viscx = 0.0f, viscy = 0.0f, viscz = 0.0f, mage2, magi2, toi, viscoeff, dx, mui, t0i, dx2;
    float gyro, vix, viy, viz, vxr, vyr, vzr, vex, vey, vez, ux, uy, uz, linv, Zi, dnei, dni, dne, dnii, Zin, dti, fac, theta_np;
    float Cln, eta_s;

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
    Zin = 1.0f / (Z + 1.0f);

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

    // theta_np = 0.5f * (1.0f + tanh(50.0f * (dni * n0 / 5e28f - 1.0f)));
    // if (i == 150 && j == 150) {
    //   eta_in[idx] = eta_s(Teev[idx], Z, dne, 0.001f, 0);
    // }
    // else{
    //   eta_in[idx] = eta_s(Teev[idx], Z, dne, 0.001f, 0);
    // }

    // if (dni * n0 / 5e28f >= 1.0f) {
    //   theta_np = 0.5f * (1.0f + tanh(50.0f * (Teev[idx] * te0 / 0.4f - 1.0f)));
    //   eta_in[idx] = eta_s(Teev[idx], Z, dne, 0.001f, 0);
    // }

    //!------------ Spitzer and LMD resistivity model -------------

    if(Teev[idx] >= 0.357*pow(Z,2)){
        Cln = 0.054*(1.110 - 0.5*log(dne/pow(Teev[idx],2)));                //! valid for T_e > 10Z^2 eV 
    }
    else  {
        Cln = 0.054*(1.625 - 0.5*log(Z*dne/pow(Teev[idx],3)));             // ! Valid for T_e < 10Z^2 eV
    }

    if(Cln < 0.01)Cln = 0.01;      
    if(!IMHD) {                               
        eta_s = 0.333*Cln*Z/pow(Teev[idx], 1.5);     //! extra factor of 0.5 to reduce
    }
    else  {
        eta_s = 0.333*Cln*Z/pow(Zin*Tiev[idx], 1.5);
    }
    eta_in[idx] = 1./sqrt(1./pow(eta_s, 2) + 400*pow(dni, 2)) ;

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

__device__ float eta_s(float e_Temp, float Za, float dne, float cln_min, int tp) {
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

  if (tp == 1) {
    printf("Za=%e\n", Za);
    printf("Cln=%e\n", Cln);
    printf("e_Temp=%e\n", e_Temp);
    printf("dne=%e\n", dne);
    printf("eta_s_value=%e\n", eta_s_value);
  }
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

// Kernel function to fill the middle third of the computational domain
__global__ void fill_rod(float *Q_in, int nx, int ny, int nq, float te0, float lxu, float lyu, float n0, float Z, float lamb, float rh_floor, float dgrate, float dxi, float dyi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * (ny * nq) + j * nq;

  if (i < nx && j < ny) {
    float wtev = 1.0 / te0;
    float nioncrit = 3.96e27 / n0 / Z;

    if (xc(i, dxi) >= 4.0 * lamb && xc(i, dxi) <= 8.0 * lamb) {
      Q_in[idx + rh] = 0.5 * 1 * nioncrit + rh_floor;  // rh
      Q_in[idx + ne] = Z * Q_in[idx + rh];  // ne
      Q_in[idx + en] = wtev * Q_in[idx + rh] * (1 + Z);  // en
      Q_in[idx + et] = wtev * Q_in[idx + ne] * (1 + Z);  // et
    }
  }
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

void write_vtk(const char* prefix, float* Q, float* eta, float* fluxx, int nx, int ny, int nq, int timestep) {
  char filename[256];
  float dne, vex, vey, vez, P;

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
          fprintf(file, "%e\n", n0_host*Q[i * ny * nq + j * nq + rh]);
      }
  }

  // Write scalar field: fluxx
  fprintf(file, "SCALARS e_temp float 1\n");
  fprintf(file, "LOOKUP_TABLE default\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        dne = 1./Q[i * ny * nq + j * nq + ne];
        vex = Q[i * ny * nq + j * nq + mx]/Q[i * ny * nq + j * nq + rh] - lil0_host*Q[i * ny * nq + j * nq + jx]*dne;
        vey = Q[i * ny * nq + j * nq + my]/Q[i * ny * nq + j * nq + rh] - lil0_host*Q[i * ny * nq + j * nq + jx]*dne;
        vez = Q[i * ny * nq + j * nq + mz]/Q[i * ny * nq + j * nq + rh] - lil0_host*Q[i * ny * nq + j * nq + jx]*dne;
        P = (aindex_host - 1)*(Q[i * ny * nq + j * nq + et] - 0.5*memi_host*Q[i * ny * nq + j * nq + ne]*(vex*vex + vey*vey + vez*vez));
        if(P < 0.)  {
        P = Q[i * ny * nq + j * nq + ne]*T_floor_host;
        }
        fprintf(file, "%e\n", P*dne*te0_host);
      }
  }


  // Write scalar field: fluxx
  fprintf(file, "SCALARS eta float 1\n");
  fprintf(file, "LOOKUP_TABLE default\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e\n", eta0_host*eta[i * ny + j  ]);
      }
  }

    // Write scalar field: fluxx
    fprintf(file, "SCALARS fluxx float 1\n");
    fprintf(file, "LOOKUP_TABLE default\n");
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fprintf(file, "%e\n", fluxx[i * ny * nq + j * nq + ez]);
        }
    }

  // Write vector field: B
  fprintf(file, "VECTORS B float\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e %e %e\n", b0_host*Q[i * ny * nq + j * nq + bx], b0_host*Q[i * ny * nq + j * nq + by], b0_host*Q[i * ny * nq + j * nq + bz]);
      }
  }

  // Write vector field: E
  fprintf(file, "VECTORS E float\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        fprintf(file, "%e %e %e\n", e0_host*Q[i * ny * nq + j * nq + ex], e0_host*Q[i * ny * nq + j * nq + ey], e0_host*Q[i * ny * nq + j * nq + ez]);
      }
  }

  // Write vector field: E
  fprintf(file, "VECTORS J float\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          fprintf(file, "%e %e %e\n", j0_derived_host*Q[i * ny * nq + j * nq + jx], j0_derived_host*Q[i * ny * nq + j * nq + jy], j0_derived_host*Q[i * ny * nq + j * nq + jz]);
      }
  }

  // Write vector field: Vion
  fprintf(file, "VECTORS U float\n");
  for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
          float dnii = 1.0f / Q[i * ny * nq + j * nq + rh];
          fprintf(file, "%e %e %e\n", v0_host*dnii*Q[i * ny * nq + j * nq + mx], v0_host*dnii*Q[i * ny * nq + j * nq + my], v0_host*dnii*Q[i * ny * nq + j * nq + mz]);
      }
  }

  fclose(file);
}


int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);  // Assuming you're using device 0

  printf("Total available shared memory per block: %d bytes\n", prop.sharedMemPerBlock);

    // Initialize constants
    initialize_constants();
    float *Q, *flux_x, *flux_y, *sources, *Q1, *Q2, *Q3, *Q4, *eta, *ect, *Tiev, *Teev, *nuei, *kap_i, *kap_e, *vis_i;
    float *d_Q, *d_flux_x, *d_flux_y, *d_sources, *d_Q1, *d_Q2, *d_Q3, *d_Q4, *d_eta, *d_ect, *d_Tiev, *d_Teev, *d_nuei, *d_kap_i, *d_kap_e, *d_vis_i;
    float dxt, dyt, lxu=12.0f*lamb_host, lyu=12.0f*lamb_host, dgrate=0.1f;
    float dt, t = 0.0f, dx=lxu/NX, dxi = 1.0f/dx, dy=lyu/NY, dyi = 1.0f/dy;
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


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((NX + blockDim.x - 1) / blockDim.x, (NY + blockDim.y - 1) / blockDim.y);

    // Set initial conditions
    initial_condition(blockDim, gridDim, d_Q, rh_floor_host, Z_host, T_floor_host, aindex_host, bapp_host, b0_host, NX, NY, NQ);
    //fill_rod<<<gridDim, blockDim>>>(d_Q, NX, NY, NQ, te0_host, lxu, lyu, n0_host, Z_host, lamb_host, rh_floor_host, dgrate, dxi, dyi);
    cudaMemcpy(Q, d_Q, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
    write_vtk("output", Q, eta, flux_x, NX, NY, NQ, nout);  // Add timestep to the filename

    if (true) {
    while (nprint < 10) {
      
      get_min_dt(&dt, 0.5, dxi, clt_host);

      dxt = dt * dxi;
      dyt = dt * dyi;

      if (iorder == 1) {
        // limit_flow_kernel<<<gridDim, blockDim>>>(d_Q, NX, NY, NQ);
        limit_flow<<<gridDim, blockDim>>>(d_Q, rh_floor_host, T_floor_host, aindex_host, Z_host, vhcf_host, NX, NY, NQ);       
        cudaDeviceSynchronize();
        get_sources_kernel<<<gridDim, blockDim>>>(d_Q, d_sources, d_eta, d_Tiev, d_Teev, d_nuei, d_kap_i, d_kap_e, d_vis_i, NX, NY, NQ, dt, dxi);
        cudaDeviceSynchronize();
        get_flux(blockDim, gridDim, N_STREAMS, d_Q, d_flux_x, d_flux_y, 0.75, NX, NY, NQ);
        advance_time_level_rz(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, d_sources, d_Q, dxt, dyt, dt, NX, NY, NQ);
        implicit_source2(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, d_eta, d_Q, dxt, dyt, dt, NX, NY, NQ);
        set_bc(blockDim, gridDim, d_Q, t + dt, dxi, dyi, k_las_host, f_las_host, Emax, Bmax, NX, NY, NQ, focal_length, w0);

        cudaMemcpy(Q, d_Q, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(eta, d_eta, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_x, d_flux_x, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_y, d_flux_y, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);

      }

      if (iorder == 2) {
        get_sources_kernel<<<gridDim, blockDim>>>(d_Q, d_sources, d_eta, d_Tiev, d_Teev, d_nuei, d_kap_i, d_kap_e, d_vis_i, NX, NY, NQ, dt, dxi);
        cudaDeviceSynchronize();
        get_flux(blockDim, gridDim, N_STREAMS, d_Q, d_flux_x, d_flux_y, 1, NX, NY, NQ);
        advance_time_level_rz(blockDim, gridDim, d_Q, d_flux_x, d_flux_y, d_sources, d_Q1, dxt, dyt, dt, NX, NY, NQ);
        implicit_source2(blockDim, gridDim, d_Q1, d_flux_x, d_flux_y, d_eta, d_Q1, dxi, dyi, dt, NX, NY, NQ);
        limit_flow<<<gridDim, blockDim>>>(d_Q1, rh_floor_host, T_floor_host, aindex_host, Z_host, vhcf_host, NX, NY, NQ);       
        cudaDeviceSynchronize();
        set_bc(blockDim, gridDim, d_Q1, t + dt, dxi, dyi, k_las_host, f_las_host, Emax, Bmax, NX, NY, NQ, focal_length, w0);

        get_sources_kernel<<<gridDim, blockDim>>>(d_Q1, d_sources, d_eta, d_Tiev, d_Teev, d_nuei, d_kap_i, d_kap_e, d_vis_i, NX, NY, NQ, dt, dxi);
        cudaDeviceSynchronize();
        get_flux(blockDim, gridDim, N_STREAMS, d_Q1, d_flux_x, d_flux_y, 1, NX, NY, NQ);
        advance_time_level_rz(blockDim, gridDim, d_Q1, d_flux_x, d_flux_y, d_sources, d_Q2, dxt, dyt, dt, NX, NY, NQ);
        implicit_source2(blockDim, gridDim, d_Q2, d_flux_x, d_flux_y, d_eta, d_Q2, dxi, dyi, dt, NX, NY, NQ);
        limit_flow<<<gridDim, blockDim>>>(d_Q2, rh_floor_host, T_floor_host, aindex_host, Z_host, vhcf_host, NX, NY, NQ);       
        cudaDeviceSynchronize();

        // Now add Q and Q2 together 
        add_Q_kernel<<<gridDim, blockDim>>>(d_Q, d_Q2, d_Q, NX, NY, NQ);
        cudaDeviceSynchronize();
        limit_flow<<<gridDim, blockDim>>>(d_Q, rh_floor_host, T_floor_host, aindex_host, Z_host, vhcf_host, NX, NY, NQ);       
        cudaDeviceSynchronize();
        set_bc(blockDim, gridDim, d_Q, t + dt, dxi, dyi, k_las_host, f_las_host, Emax, Bmax, NX, NY, NQ, focal_length, w0);

        cudaMemcpy(Q, d_Q, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(eta, d_eta, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_x, d_flux_x, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flux_y, d_flux_y, NX * NY * NQ * sizeof(float), cudaMemcpyDeviceToHost);
      }

      niter++;
      t += dt;
      if (niter % 100 == 0) {
        printf("\nIteration time: %f seconds\n", (float)clock() / CLOCKS_PER_SEC);
        printf("nout=%d\n", nout);
        printf("t= %e ns, dt= %e ns, niter= %d\n", t * 100, dt*100, niter);
        // printf("dxt= %e, dyt= %e\n", dxi*dt, dyi*dt, niter);
        // printf("lambda= %e, test= %e, NX= %d, dx= %e, dy= %e\n", lamb_host, lamb_host/NX, NX, dx, dy);
        nprint++;

        check_Iv(NX - 1 / dxi, NY / 2);
        cudaDeviceSynchronize();

        write_vtk("output", Q, eta, flux_x, NX, NY, NQ, nout+1);  // Add timestep to the filename
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
