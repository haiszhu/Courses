/*
 * cuda lap slp pot & matlab mex interface
 * 11/15/23 Hai, do not form matrix explicitly
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
// #include "cublas_v2.h"
/* these are for matlab */
#include "mex.h"
#include "gpu/mxGPUArray.h"

#define THRESH 1e-15

/*
 * Device code
 */
void __global__ culapslppot(double const * const src,
                            double const * const targ,
                            double const * const x,
                            double * const y,
                            int const N,
                            int const M)
{
  /* Calculate the global linear index, assuming a 1-d grid. */
  int const i = blockDim.x * blockIdx.x + threadIdx.x;
  double dx, dy, dz, dd, threshsq;
  threshsq = THRESH*THRESH;
  if (i<M) {
    y[i] = 0.0;
    for (int j = 0; j < N; ++j) {
      dx = src[3*j]   - targ[3*i];
      dy = src[3*j+1] - targ[3*i+1];
      dz = src[3*j+2] - targ[3*i+2];
      dd = dx*dx + dy*dy + dz*dz;
      if (dd>threshsq){
        y[i] += x[j]*rsqrt(dd);
        // y[i] += x[j]/sqrt(dd);
      }
    }
  }
}

/*
 * Host code (just slp potential)
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  /* Declare all variables.*/
  mxDouble *src, *targ, *x;  /* input: source, target, density */
  mxDouble *y;               /* output: potential */
  double *d_src, *d_targ;
  double *d_x, *d_y;
  float *curuntime;
  //double *d_A;
  int N, M;                  /* dimension */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  char const * const errId = "parallel:gpu:culapslppot:InvalidInput";
  char const * const errMsg = "Invalid input to MEX file.";
  
  src = mxGetDoubles(prhs[0]);   /* source */
  targ = mxGetDoubles(prhs[1]);  /* target */
  x = mxGetDoubles(prhs[2]);     /* density */
  M = (int)(mxGetN(prhs[1]));
  N = (int)(mxGetN(prhs[0]));
  
  /* Choose a reasonably sized number of threads for the block. */
  int const threadsPerBlock = 64;
  int blocksPerGrid;
  
  /* Initialize the MathWorks GPU API. */
  mxInitGPU();
  if ((nrhs != 3) || (nlhs != 2)) {
      mexErrMsgIdAndTxt(errId, errMsg);
  }
  
  /* potential, no explicit lap slp mat */
  blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
  cudaMalloc((void**)&d_src, 3*N*sizeof(double));
  cudaMemcpy(d_src, src, 3*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_targ, 3*M*sizeof(double));
  cudaMemcpy(d_targ, targ, 3*M*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_x, N*sizeof(double));
  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_y, M*sizeof(double));
  cudaEventRecord(start);
  culapslppot<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_targ, d_x, d_y, N, M);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  /* Copy result back to host */
  plhs[0] = mxCreateDoubleMatrix((mwSize)M, (mwSize)1, mxREAL);
  y = mxGetDoubles(plhs[0]); /* 1st output */
  cudaMemcpy(y, d_y, M*sizeof(double), cudaMemcpyDeviceToHost);
  plhs[1] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
  curuntime = (float *) mxGetData(plhs[1]);
  curuntime[0] = milliseconds;
  
  /* Free GPU memory */
  cudaFree(d_src);
  cudaFree(d_targ);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}