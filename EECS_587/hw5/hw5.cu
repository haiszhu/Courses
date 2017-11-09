#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <math.h>

#include "device_launch_parameters.h"

//#define N 128
#define TPB 16

using namespace std;


__global__ void MatAdd(double* A, double* B, double* C, int N, int N0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //if (i>N-1 || j>N-1) return;
    if (i < N0 && j < N0)
        //C[i][j] = A[i][j] + B[i][j];
        C[i + j*N] = A[i + j*N] + B[i + j*N];
}

__global__ void MatMedian(double* A, double* B, int N, int N0){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ double temp[TPB+2][TPB+2];
  if((threadIdx.x == TPB-1)&&(i < N0)){
    temp[threadIdx.y+1][TPB+1] = A[i+1+j*N];
  }
  if((threadIdx.x == 0)&&(i >0)){
    temp[threadIdx.y+1][0] = A[i-1+j*N];
  }
  if((threadIdx.y == 0)&&(j > 0)){
    temp[0][threadIdx.x+1] = A[i+(j-1)*N];
  }
  if((threadIdx.y == TPB-1)&&(j < N0)){
    temp[TPB+1][threadIdx.x+1] = A[i+(j+1)*N];
  }
  temp[threadIdx.y+1][threadIdx.x+1] = A[i+j*N];
  __syncthreads();
  if ((0 < i) && (i < N0-1) && (0 < j) && (j < N0-1)){
    //B[i + j*N] = 2*temp[threadIdx.y][threadIdx.x];
    //B[i+j*N] = A[i+j*N]+A[i+1+j*N];
    B[i+j*N] = temp[threadIdx.y+1][threadIdx.x+1] + temp[threadIdx.y+1][threadIdx.x+2]+temp[threadIdx.y+1][threadIdx.x]+temp[threadIdx.y][threadIdx.x+1]+temp[threadIdx.y+2][threadIdx.x+1];
  }
    //B[i + j*N] = 2*A[i + j*N];
}

//int a[N][N], b[N][N], c[N][N];

int main(int argc, char *argv[]){
    int N0;
    for(int i=1; i<argc; i++){
      N0 = atoi(argv[i]);
    }
    int N = (ceil((N0-1)/(4*TPB))+1)*TPB*4;
    cout << "Number of elements each direction is " << N << endl;
    double** a = new double*[N];
    double** b = new double*[N];
    
    a[0] = new double[N*N]();   //initialize to 0, inside kernal, 0 gets copied to temp on unused thread
    b[0] = new double[N*N]();
    
    for(int i=1; i<N; i++){
       a[i] = a[i-1]+N;
       b[i] = b[i-1]+N;
    }
    

    double* dev_a;
    double* dev_b;
    double* dev_block_sum;
    
    size_t pitch;
    for (int i = 0; i < N0; ++i)
    {
        for (int j = 0; j < N0; ++j)
        {
            a[i][j] = pow(sin(i*i+j), 2) + cos(i-j);
        }
    }
    
    cudaMallocPitch(&dev_a,&pitch,sizeof(double)*N,N);
    cudaMallocPitch(&dev_b,&pitch,sizeof(double)*N,N);
    cudaMemcpy2D(dev_a,pitch,a[0],sizeof(double)*N,sizeof(double)*N,N,cudaMemcpyHostToDevice);
    
    
    dim3 gridDim((N-1)/TPB + 1, (N-1)/TPB + 1);
    dim3 blockDim(TPB, TPB);
    MatMedian <<<gridDim, blockDim >>>(dev_a, dev_b, N, N0);

    cudaMemcpy2D(b[0],pitch,dev_b,sizeof(double)*N,sizeof(double)*N,N,cudaMemcpyDeviceToHost);

    double temp = 0;
    for (int i = 1; i < N0-1; ++i)
    {
        for (int j = 1; j < N0-1; ++j)
        {
            temp = 1+a[i][j]+a[i][j+1]+a[i][j-1]+a[i-1][j]+a[i+1][j];
            if (temp != b[i][j])
            {
                printf("Failure at %d  %d\n", i, j);
            }
            //cout << c[i][j] << " ";
        }
       //cout << endl;
    }
    cout << temp << endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    //return 0;
    //system("pause");
}

