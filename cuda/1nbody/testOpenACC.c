// https://docs.nvidia.com/hpc-sdk/compilers/openacc-gs/index.html for reference
// install hpc-sdk apt version ... https://developer.nvidia.com/hpc-sdk-downloads
// create ~/nvidia.sh ... https://docs.nvidia.com/hpc-sdk//hpc-sdk-install-guide/index.html or https://www.scivision.dev/install-nvidia-hpc-free-compiler/
// to compile ... nvc -acc -gpu=cc80 testOpenACC.c
// ./a.out ... Success!
#include <stdio.h>
#define N 1000
int array[N];
int main() {
#pragma acc parallel loop copy(array[0:N])
   for(int i = 0; i < N; i++) {
      array[i] = 3.0;
   }
   printf("Success!\n");
}