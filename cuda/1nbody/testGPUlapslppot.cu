/*
 * nvcc testGPUlapslppot.cu
 * ./a.out 
 * then check testGPUlapslppot.m in matlab
 * certain part of testGPUlapslppot.m load data and verify results
 * not done yet, should initialize src to be 3 by N
 * 11/17/23
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
// #include "cublas_v2.h"

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
 * Host code, not verified yet...
 */
void culapslppot_cpu(double const * const src,
                     double const * const targ,
                     double const * const x,
                     double * const y,
                     int const N,
                     int const M)
{
  /* Calculate the global linear index, assuming a 1-d grid. */
	double dx, dy, dz, dd, threshsq;
	threshsq = THRESH*THRESH;
	for (int i = 0; i < M; ++i) {
		y[i] = 0.0;
		for (int j = 0; j < N; ++j) {
			dx = src[3*j]   - targ[3*i];
      dy = src[3*j+1] - targ[3*i+1];
      dz = src[3*j+2] - targ[3*i+2];
      dd = dx*dx + dy*dy + dz*dz;
			if (dd>threshsq) {
				y[i] += x[j]/sqrt(dd);
			}
		}
	}
}

// void transpose(...) {
// }

int main() {
  const int N = 100000;
  const int M = 100000;

  // double (*src)[N]  = (double(*)[N])malloc(3*N*sizeof(double));
  // double (*targ)[M] = (double(*)[M])malloc(3*M*sizeof(double));
  double (*src)[3] = (double(*)[3])malloc(N*3*sizeof(double));
  double (*targ)[3] = (double(*)[3])malloc(M*3*sizeof(double));
  double *mu, *pot, *pot_gpu;
  mu = (double*)malloc(N*sizeof(double));
  pot = (double*)malloc(M*sizeof(double));
  pot_gpu = (double*)malloc(M*sizeof(double)); // still cpu, result copied from gpu
  double *d_src, *d_targ;
  double *d_mu, *d_pot;
  float curuntime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Initialize data */
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < N; j++) {
      src[j][i]  = (double)rand()/RAND_MAX; // src
    }
    for (int j = 0; j < M; j++) {
      targ[j][i] = (double)rand()/RAND_MAX; // targ
    }
  }
  for (int j = 0; j < N; j++) {
    mu[j] = (double)rand()/RAND_MAX; // mu
  }

  /* write src targ and mu data to file */
  FILE *srcFile = fopen("src.txt", "w");
  if (srcFile != NULL) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < N; j++) {
        fprintf(srcFile, "%.17g ", src[j][i]);
      }
      fprintf(srcFile, "\n");
    }
    fclose(srcFile);
  }
  FILE *targFile = fopen("targ.txt", "w");
  if (targFile != NULL) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < M; j++) {
        fprintf(targFile, "%.17g ", targ[j][i]);
      }
      fprintf(targFile, "\n");
    }
    fclose(targFile);
  }
  FILE *muFile = fopen("mu.txt", "w");
  if (muFile != NULL) {
    for (int j = 0; j < N; j++) {
      fprintf(muFile, "%.17g\n", mu[j]);
    }
    fclose(muFile);
  } 

  /* Choose a reasonably sized number of threads for the block. */
  int const threadsPerBlock = 64;
  int blocksPerGrid;

  /* potential, no explicit lap slp mat */
  blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
  cudaMalloc((void**)&d_src, 3*N*sizeof(double));
  cudaMemcpy(d_src, src, 3*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_targ, 3*M*sizeof(double));
  cudaMemcpy(d_targ, targ, 3*M*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_mu, N*sizeof(double));
  cudaMemcpy(d_mu, mu, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_pot, M*sizeof(double));
  cudaEventRecord(start);
  culapslppot<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_targ, d_mu, d_pot, N, M);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&curuntime, start, stop);
  printf("GPU Calculation Time: %f milliseconds\n", curuntime);

  /* Copy result back to host */
  cudaMemcpy(pot_gpu, d_pot, M*sizeof(double), cudaMemcpyDeviceToHost);

  /* write potential data to file */
  FILE *potFile = fopen("pot.txt", "w");
  if (potFile != NULL) {
    for (int i = 0; i < M; i++) {
      fprintf(potFile, "%.17g\n", pot_gpu[i]);
    }
    fclose(potFile);
  }

  /* free up memory */
  free(src);
  free(targ);
  free(mu);
  free(pot);
  free(pot_gpu); // to do: finish testing cpu version

  cudaFree(d_src);
  cudaFree(d_targ);
  cudaFree(d_mu);
  cudaFree(d_pot);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}