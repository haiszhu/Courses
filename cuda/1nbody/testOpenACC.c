// https://docs.nvidia.com/hpc-sdk/compilers/openacc-gs/index.html for reference
// install hpc-sdk apt version ... https://developer.nvidia.com/hpc-sdk-downloads
// create ~/nvidia.sh ... https://docs.nvidia.com/hpc-sdk//hpc-sdk-install-guide/index.html or https://www.scivision.dev/install-nvidia-hpc-free-compiler/
// to compile ... nvc -acc -gpu=cc80 testOpenACC.c
// ./a.out ... Success!
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <omp.h>

#define N 40000
#define M 50000
int main() {
  double (*src)[N] = malloc(3*N*sizeof(double));
  double (*targ)[M] = malloc(3*M*sizeof(double)); 
  double (*mu) = malloc(N*sizeof(double));
  double (*pot) = malloc(M*sizeof(double));
  double runtime;
  // double src[3][N], targ[3][M]; 
  // double mu[N], pot[M];
  srand(time(NULL));

  /* Initialize src, targ, and mu */
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < N; j++) {
      src[i][j]  = (double)rand()/RAND_MAX; // src
    }
    for (int j = 0; j < M; j++) {
      targ[i][j] = (double)rand()/RAND_MAX; // targ
    }
  }
  for (int j = 0; j < N; j++) {
    mu[j] = (double)rand()/RAND_MAX;
  }

  /* write src targ and mu data to file */
  FILE *srcFile = fopen("src.txt", "w");
  if (srcFile != NULL) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < N; j++) {
        fprintf(srcFile, "%.17g ", src[i][j]);
      }
      fprintf(srcFile, "\n");
    }
    fclose(srcFile);
  }
  FILE *targFile = fopen("targ.txt", "w");
  if (targFile != NULL) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < M; j++) {
        fprintf(targFile, "%.17g ", targ[i][j]);
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

  /* laplace slp */
  double start_time = omp_get_wtime(); // Start timing
  #pragma acc parallel loop copy(mu[0:N], pot[0:M], src[0:3][0:N], targ[0:3][0:M])
  for (int i = 0; i < M; i++) {
    pot[i] = 0.0;
    for (int j = 0; j < N; j++) {
      double dx = src[0][j] - targ[0][i];
      double dy = src[1][j] - targ[1][i];
      double dz = src[2][j] - targ[2][i];
      double dd = dx*dx + dy*dy + dz*dz;
      pot[i] += mu[j]/sqrt(dd);
    }
  }
  double end_time = omp_get_wtime(); // End timing
  runtime = 1000*(end_time-start_time);

  /* write potential data to file */
  FILE *potFile = fopen("pot.txt", "w");
  if (potFile != NULL) {
    for (int i = 0; i < M; i++) {
      fprintf(potFile, "%.17g\n", pot[i]);
    }
    fclose(potFile);
  }
  FILE *runtimeFile = fopen("runtime.txt", "w");
  if (potFile != NULL) {
    fprintf(runtimeFile, "%.17g\n", runtime);
    fclose(runtimeFile);
  }

  free(src);
  free(targ);
  free(mu);
  free(pot);

  return 0;
}