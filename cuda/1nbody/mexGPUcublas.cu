#include "mex.h"
#include "cublas_v2.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check input and output arguments
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("matrixVectorMultiply:invalidInput", "Two input arguments required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("matrixVectorMultiply:invalidOutput", "One output argument required.");
    }

    // Get input matrices
    double* A = mxGetPr(prhs[0]); // Matrix A
    double* x = mxGetPr(prhs[1]); // Vector x
    mwSize M = mxGetM(prhs[0]);   // Number of rows in A
    mwSize N = mxGetN(prhs[0]);   // Number of columns in A and length of x

    // Allocate output matrix
    plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL);
    double* y = mxGetPr(plhs[0]); // Output vector y

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory
    double *d_A, *d_x, *d_y;
    cudaMalloc((void**)&d_A, M * N * sizeof(double));
    cudaMalloc((void**)&d_x, N * sizeof(double));
    cudaMalloc((void**)&d_y, M * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix-vector multiplication
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);

    // Copy result from device to host
    cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
}

// #include <iostream>
// #include <cstdlib>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// 
// void matrixVectorMultiply(const double* A, const double* x, double* y, int M, int N) {
//     // Initialize cuBLAS
//     cublasHandle_t handle;
//     cublasCreate(&handle);
// 
//     // Allocate device memory
//     double *d_A, *d_x, *d_y;
//     cudaMalloc((void**)&d_A, M * N * sizeof(double));
//     cudaMalloc((void**)&d_x, N * sizeof(double));
//     cudaMalloc((void**)&d_y, M * sizeof(double));
// 
//     // Copy data from host to device
//     cudaMemcpy(d_A, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
// 
//     // Perform matrix-vector multiplication
//     double alpha = 1.0, beta = 0.0;
//     cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
// 
//     // Copy result from device to host
//     cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
// 
//     // Clean up
//     cudaFree(d_A);
//     cudaFree(d_x);
//     cudaFree(d_y);
//     cublasDestroy(handle);
// }
// 
// int main() {
//     const int M = 3; // Number of rows in the matrix
//     const int N = 4; // Number of columns in the matrix and length of the vector
// 
//     // Host matrix and vector
//     double h_A[M * N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
//     double h_x[N] = {2.0, 1.0, 0.0, -1.0};
//     double h_y[M];
// 
//     // Perform matrix-vector multiplication
//     matrixVectorMultiply(h_A, h_x, h_y, M, N);
// 
//     // Print the result
//     std::cout << "Resulting vector y:" << std::endl;
//     for (int i = 0; i < M; ++i) {
//         std::cout << h_y[i] << std::endl;
//     }
// 
//     return 0;
// }
