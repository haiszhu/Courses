// https://docs.nvidia.com/hpc-sdk/compilers/openacc-gs/index.html for reference
// install hpc-sdk apt version ... https://developer.nvidia.com/hpc-sdk-downloads
// create ~/nvidia.sh ... https://docs.nvidia.com/hpc-sdk//hpc-sdk-install-guide/index.html or https://www.scivision.dev/install-nvidia-hpc-free-compiler/
// to compile ... nvc -acc -gpu=cc80 testOpenACC.c
// ./a.out ... Success!
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>

#define N 10000
#define M 10000

int main() {
  double srcx[N], targx[M]; 
  double srcy[N], targy[M];
  double srcz[N], targz[M]; 
  double mu[N], pot[M];
  srand(time(NULL));
  
  /* srcx y z */
  FILE *srcxFile = fopen("srcx.txt", "w");
  if (srcxFile != NULL) {
    for (int i = 0; i < N; i++) {
      srcx[i] = (double)rand()/RAND_MAX;
      fprintf(srcxFile, "%.17g\n", srcx[i]);
    }
    fclose(srcxFile);
  }
  FILE *srcyFile = fopen("srcy.txt", "w");
  if (srcyFile != NULL) {
    for (int i = 0; i < N; i++) {
      srcy[i] = (double)rand()/RAND_MAX;
      fprintf(srcyFile, "%.17g\n", srcy[i]);
    }
    fclose(srcyFile);
  }
  FILE *srczFile = fopen("srcz.txt", "w");
  if (srczFile != NULL) {
    for (int i = 0; i < N; i++) {
      srcz[i] = (double)rand()/RAND_MAX;
      fprintf(srczFile, "%.17g\n", srcz[i]);
    }
    fclose(srczFile);
  }
  
  /* targx y z*/
  FILE *targxFile = fopen("targx.txt", "w");
  if (targxFile != NULL) {
    for (int i = 0; i < M; i++) {
      targx[i] = (double)rand() / RAND_MAX;
      fprintf(targxFile, "%.17g\n", targx[i]);
    }
    fclose(targxFile);
  }
  FILE *targyFile = fopen("targy.txt", "w");
  if (targyFile != NULL) {
    for (int i = 0; i < M; i++) {
      targy[i] = (double)rand() / RAND_MAX;
      fprintf(targyFile, "%.17g\n", targy[i]);
    }
    fclose(targyFile);
  }
  FILE *targzFile = fopen("targz.txt", "w");
  if (targzFile != NULL) {
    for (int i = 0; i < M; i++) {
      targz[i] = (double)rand() / RAND_MAX;
      fprintf(targzFile, "%.17g\n", targz[i]);
    }
    fclose(targzFile);
  }

  /* mu */
  FILE *muFile = fopen("mu.txt", "w");
  if (muFile != NULL) {
    for (int i = 0; i < N; i++) {
      mu[i] = (double)rand() / RAND_MAX;
      fprintf(muFile, "%.17g\n", mu[i]);
    }
    fclose(muFile);
  }

  /* nbody */
  #pragma acc parallel loop copy(mu[0:N], pot[0:M],srcx[0:N], srcy[0:N], srcz[0:N], targx[0:M], targy[0:M], targz[0:M])
  for (int i = 0; i < M; i++) {
    pot[i] = 0.0;
    for (int j = 0; j < N; j++) {
      double dx = srcx[j]-targx[i];
      double dy = srcy[j]-targy[i];
      double dz = srcz[j]-targz[i];
      double dd = dx*dx + dy*dy + dz*dz;
      pot[i] += mu[j]/sqrt(dd);
    }
  }

  /* pot */
  FILE *potFile = fopen("pot.txt", "w");
  if (potFile != NULL) {
    for (int i = 0; i < M; i++) {
      fprintf(potFile, "%.17g\n", pot[i]);
    }
    fclose(potFile);
  } 
  
  return 0;
}
