/*
 * cuda lap slp & matlab mex interface
 * 11/12/23 Hai, 
 * pass src & targ...; (done) 
 * to do: 1. wrap slp & cublas into one kernel? 2. slpn or dlp?; 3. 2d grid?
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
/* these are for matlab */
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"

#ifndef mxGetDoubles
#define mxGetDoubles(p) mxGetPr(p)
#endif

/*
 * Device code, still 1d array... instead of original [3,N] source, [3,M] target 
 * did not check each entry of A...
 */
void __global__ culapslpmat(double const * const src,
                            double const * const targ,
                            double * const A,
                            int const N,
                            int const M)
{
  /* Calculate the global linear index, assuming a 1-d grid. */
  int const i = blockDim.x * blockIdx.x + threadIdx.x;
  int row = i/N;
  int col = i - row*N;
  double dx, dy, dz;
  if (i < N * M) {
    dx = src[3*col]   - targ[3*row];
    dy = src[3*col+1] - targ[3*row+1];
    dz = src[3*col+2] - targ[3*row+2];
    A[i] = 1.0/sqrt(dx*dx+dy*dy+dz*dz);
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
  double *d_A;
  double *d_x, *d_y;
  int N, M;                  /* dimension */
  char const * const errId = "parallel:gpu:culapslpmat:InvalidInput";
  char const * const errMsg = "Invalid input to MEX file.";
  
  src = mxGetDoubles(prhs[0]);   /* source */
  targ = mxGetDoubles(prhs[1]);  /* target */
  x = mxGetDoubles(prhs[2]);     /* density */
  M = (int)(mxGetN(prhs[1]));
  N = (int)(mxGetN(prhs[0]));
  
  /* Choose a reasonably sized number of threads for the block. */
  int const threadsPerBlock = 128;
  int blocksPerGrid;
  
  /* Initialize the MathWorks GPU API. */
  mxInitGPU();
  if ((nlhs != 1)||(nrhs != 3)) {
    mexErrMsgIdAndTxt(errId, errMsg);
  }
  
  /* explicit slp mat */
  blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
  cudaMalloc((void**)&d_src, 3*N*sizeof(double));
  cudaMemcpy(d_src, src, 3*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_targ, 3*M*sizeof(double));
  cudaMemcpy(d_targ, targ, 3*M*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_A, M*N*sizeof(double));
  culapslpmat<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_targ, d_A, N, M);
  
  /* cuBLAS matvec potential */
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc((void**)&d_x, N*sizeof(double));
  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_y, M*sizeof(double));
  double alpha = 1.0, beta = 0.0;
  cublasDgemv(handle, CUBLAS_OP_T, N, M, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);

  /* Copy result back to host */
  plhs[0] = mxCreateDoubleMatrix((mwSize)M, (mwSize)1, mxREAL);
  y = mxGetDoubles(plhs[0]); /* 1st output */
  cudaMemcpy(y, d_y, M*sizeof(double), cudaMemcpyDeviceToHost);
  
  /* free gpu memory */
  cudaFree(d_src);
  cudaFree(d_targ);
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
  cublasDestroy(handle);
}


/*
 * Host code (previous mex function, 2nd output is slp matrix)
 */
// void mexFunction_v0(int nlhs, mxArray *plhs[],
//                  int nrhs, mxArray const *prhs[])
// {
//     /* Declare all variables.*/
//     mxDouble *srcx, *srcy, *srcz, *targx, *targy, *targz, *x;  /* input: source, target, density */
//     mxDouble *y, *A;  /* output: potential, slp mat */
//     double *d_srcx, *d_srcy, *d_srcz;
//     double *d_targx, *d_targy, *d_targz;
//     double *d_A, *d_At;
//     int N, M; /* dimension */
//     char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
//     char const * const errMsg = "Invalid input to MEX file.";
//     double *d_x, *d_y;
// 
//     srcx = mxGetDoubles(prhs[0]);   /* source */
//     srcy = mxGetDoubles(prhs[1]);
//     srcz = mxGetDoubles(prhs[2]);
//     targx = mxGetDoubles(prhs[3]);  /* target */
//     targy = mxGetDoubles(prhs[4]);
//     targz = mxGetDoubles(prhs[5]);
//     x = mxGetDoubles(prhs[6]);      /* density */
//     M = (int)(mxGetN(prhs[3]));
//     N = (int)(mxGetN(prhs[0]));
// 
//     /* Choose a reasonably sized number of threads for the block. */
//     int const threadsPerBlock = 128;
//     int blocksPerGrid;
// 
//     /* Initialize the MathWorks GPU API. */
//     mxInitGPU();
//     if ((nrhs!=7)) {
//         mexErrMsgIdAndTxt(errId, errMsg);
//     }
// 
//     /* explicit slp mat */
//     blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
//     cudaMalloc((void**)&d_srcx, N*sizeof(double));
//     cudaMemcpy(d_srcx, srcx, N*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_srcy, N*sizeof(double));
//     cudaMemcpy(d_srcy, srcy, N*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_srcz, N*sizeof(double));
//     cudaMemcpy(d_srcz, srcz, N*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_targx, M*sizeof(double));
//     cudaMemcpy(d_targx, targx, M*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_targy, M*sizeof(double));
//     cudaMemcpy(d_targy, targy, M*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_targz, M*sizeof(double));
//     cudaMemcpy(d_targz, targz, M*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_A, M*N*sizeof(double));
//     culapslpmat<<<blocksPerGrid, threadsPerBlock>>>(d_srcx, d_srcy, d_srcz, d_targx, d_targy, d_targz, d_A, N, M);
// 
//     /* cuBLAS matvec potential */
//     plhs[0] = mxCreateDoubleMatrix((mwSize)M, (mwSize)1, mxREAL); 
//     y = mxGetDoubles(plhs[0]); /* 1st output */
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaMalloc((void**)&d_x, N*sizeof(double));
//     cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&d_y, M*sizeof(double));
//     double alpha = 1.0, beta = 0.0;
//     cublasDgemv(handle, CUBLAS_OP_T, N, M, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
//     cudaMemcpy(y, d_y, M*sizeof(double), cudaMemcpyDeviceToHost);
// 
//     /* need transpose? yes */
//     plhs[1] = mxCreateDoubleMatrix(M, N, mxREAL); 
//     A = mxGetDoubles(plhs[1]); /* 2nd output */
// //     cudaMemcpy(A, d_A, M*N*sizeof(double), cudaMemcpyDeviceToHost); /* if just copy, then outsie needs A = reshape(A(:),N,M)'; */
//     cudaMalloc((void**)&d_At, M*N*sizeof(double));
//     cumextranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_At, N, M);
//     cudaMemcpy(A, d_At, M*N*sizeof(double), cudaMemcpyDeviceToHost); /* A transpose? */
// 
//     /* free gpu memory */
//     cudaFree(d_srcx);
//     cudaFree(d_srcy);
//     cudaFree(d_srcz);
//     cudaFree(d_targx);
//     cudaFree(d_targy);
//     cudaFree(d_targz);
//     cudaFree(d_A);
//     cudaFree(d_At);
//     cudaFree(d_x);
//     cudaFree(d_y);
//     cublasDestroy(handle);
// }
