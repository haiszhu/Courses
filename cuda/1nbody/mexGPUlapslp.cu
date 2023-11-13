/*
 * cuda lap slp & matlab mex interface
 * 11/12/23 Hai, to do: 1. pass src & targ...; 2. slpn or dlp?; 3. 2d grid?
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
/* these are for matlab */
#include "mex.h"
#include "gpu/mxGPUArray.h"

/*
 * Device code
 */
void __global__ culapslpmat(double const * const srcx,
                            double const * const srcy,
                            double const * const srcz,
                            double const * const targx,
                            double const * const targy,
                            double const * const targz,
                            double * const A,
                            int const N,
                            int const M)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    int row = i/N;
    int col = i - row*N;
    double dx, dy, dz;
    if (i < N*M) {
        dx = srcx[col] - targx[row];
        dy = srcy[col] - targy[row];
        dz = srcz[col] - targz[row];
        A[i] = 1.0/sqrt(dx*dx+dy*dy+dz*dz);
    }
}

/*
 * mex transpose to assist 1d grid
 */
void __global__ cumextranspose(double const * const A,
                               double * const At,
                               int const N,
                               int const M)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    int row = i/N;
    int col = i - row*N;
    if (i < N*M) {
        At[col*M+row] = A[i];
    }
}


/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxDouble *srcx, *srcy, *srcz, *targx, *targy, *targz, *x;  /* input: source, target, density */
    mxDouble *A, *y;  /* output: slp mat, potential */
    double *d_srcx, *d_srcy, *d_srcz;
    double *d_targx, *d_targy, *d_targz;
    double *d_A, *d_At;
    int N, M; /* dimension */
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    double *d_x, *d_y;

    srcx = mxGetDoubles(prhs[0]);   /* source */
    srcy = mxGetDoubles(prhs[1]);
    srcz = mxGetDoubles(prhs[2]);
    targx = mxGetDoubles(prhs[3]);  /* target */
    targy = mxGetDoubles(prhs[4]);
    targz = mxGetDoubles(prhs[5]);
    x = mxGetDoubles(prhs[6]);      /* density */
    M = (int)(mxGetN(prhs[3]));
    N = (int)(mxGetN(prhs[0]));

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 128;
    int blocksPerGrid;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    if ((nrhs!=7)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /* explicit slp mat */
    blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc((void**)&d_srcx, N*sizeof(double));
    cudaMemcpy(d_srcx, srcx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_srcy, N*sizeof(double));
    cudaMemcpy(d_srcy, srcy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_srcz, N*sizeof(double));
    cudaMemcpy(d_srcz, srcz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_targx, M*sizeof(double));
    cudaMemcpy(d_targx, targx, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_targy, M*sizeof(double));
    cudaMemcpy(d_targy, targy, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_targz, M*sizeof(double));
    cudaMemcpy(d_targz, targz, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_A, M*N*sizeof(double));
    culapslpmat<<<blocksPerGrid, threadsPerBlock>>>(d_srcx, d_srcy, d_srcz, d_targx, d_targy, d_targz, d_A, N, M);

    /* need transpose? */
    cudaMalloc((void**)&d_At, M*N*sizeof(double));
    cumextranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_At, N, M);
    plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL); 
    A = mxGetDoubles(plhs[0]); /* 1st output */
    cudaMemcpy(A, d_At, M*N*sizeof(double), cudaMemcpyDeviceToHost); /* A transpose? */

    /* cuBLAS matvec potential */
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void**)&d_x, N*sizeof(double));
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_y, M*sizeof(double));
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_At, M, d_x, 1, &beta, d_y, 1);
    plhs[1] = mxCreateDoubleMatrix((mwSize)M, (mwSize)1, mxREAL); 
    y = mxGetDoubles(plhs[1]); /* 2nd output */
    cudaMemcpy(y, d_y, M*sizeof(double), cudaMemcpyDeviceToHost);

    /* free gpu memory */
    cudaFree(d_srcx);
    cudaFree(d_targx);
    cudaFree(d_A);
    cudaFree(d_At);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
}
