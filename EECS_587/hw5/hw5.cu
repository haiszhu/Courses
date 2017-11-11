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

__global__ void MatBlockSum(double* A, double* blockSum, int N, int N0, int Nb){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ double tempSum;
  __shared__ double temp[TPB][TPB];
  tempSum = 0;  
  
  if ((0 <= i) && (i < N0) && (0 <= j) && (j < N0)){
    temp[threadIdx.y][threadIdx.x] = A[i+j*N];
  } else{
    temp[threadIdx.y][threadIdx.x] = 0;
  }
  __syncthreads();
  
  int incr;
  incr = TPB/2;
  __syncthreads();
  if((threadIdx.x+incr < TPB)&&(threadIdx.y+incr < TPB)){
    temp[threadIdx.y][threadIdx.x] = temp[threadIdx.y][threadIdx.x] + temp[threadIdx.y][threadIdx.x+incr] + temp[threadIdx.y+incr][threadIdx.x] + temp[threadIdx.y+incr][threadIdx.x+incr];
  }
  __syncthreads();
  
  incr = incr/2;
  if((threadIdx.x+incr < TPB/2)&&(threadIdx.y+incr < TPB/2)){
    temp[threadIdx.y][threadIdx.x] = temp[threadIdx.y][threadIdx.x] + temp[threadIdx.y][threadIdx.x+incr] + temp[threadIdx.y+incr][threadIdx.x] + temp[threadIdx.y+incr][threadIdx.x+incr];
  }
  __syncthreads();

  incr = incr/2;
  if((threadIdx.x+incr < TPB/4)&&(threadIdx.y+incr < TPB/4)){
    temp[threadIdx.y][threadIdx.x] = temp[threadIdx.y][threadIdx.x] + temp[threadIdx.y][threadIdx.x+incr] + temp[threadIdx.y+incr][threadIdx.x] + temp[threadIdx.y+incr][threadIdx.x+incr];
  }
  __syncthreads();

  incr = incr/2;
  if((threadIdx.x+incr < TPB/8)&&(threadIdx.y+incr < TPB/8)){
    temp[threadIdx.y][threadIdx.x] = temp[threadIdx.y][threadIdx.x] + temp[threadIdx.y][threadIdx.x+incr] + temp[threadIdx.y+incr][threadIdx.x] + temp[threadIdx.y+incr][threadIdx.x+incr];
  }   
  __syncthreads();

  blockSum[blockIdx.x+blockIdx.y*Nb/TPB] = temp[0][0];
}

__global__ void MatMedian(double* A, double* B, int N, int N0){
//__global__ void MatMedian(double* A, double* B, double* blockSum, int N, int N0){
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
    double* blockSum = new double[((N-1)/TPB + 1)*((N-1)/TPB + 1)];
    
    a[0] = new double[N*N]();
    b[0] = new double[N*N];
    
    for(int i=1; i<N; i++){
       a[i] = a[i-1]+N;
       b[i] = b[i-1]+N;
    }

    double* dev_a;
    double* dev_b;
    double* dev_blockSum;
    double* dev_sum;

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
    cudaMalloc((void **)&dev_blockSum, ((N-1)/TPB + 1)*((N-1)/TPB + 1)*sizeof(double));
    cudaMalloc((void **)&dev_sum, 1*sizeof(double));
    cudaMemcpy2D(dev_a,pitch,a[0],sizeof(double)*N,sizeof(double)*N,N,cudaMemcpyHostToDevice);
    
    
    dim3 gridDim((N-1)/TPB + 1, (N-1)/TPB + 1);
    dim3 blockDim(TPB, TPB);
    //MatMedian <<<gridDim, blockDim >>>(dev_a, dev_b, N, N0);
    MatMedian <<<gridDim, blockDim >>>(dev_a, dev_b, N, N0);

    //cudaMemcpy2D(b[0],pitch,dev_b,sizeof(double)*N,sizeof(double)*N,N,cudaMemcpyDeviceToHost);
    cudaMemcpy2D(b[0],sizeof(double)*N,dev_b,pitch,sizeof(double)*N,N,cudaMemcpyDeviceToHost);
    //cudaMemcpy2D(blockSum[0],pitch_b,dev_blockSum,sizeof(double)*((N-1)/TPB + 1),sizeof(double)*((N-1)/TPB + 1),(N-1)/TPB + 1,cudaMemcpyDeviceToHost);
    MatBlockSum <<<gridDim, blockDim >>>(dev_a, dev_blockSum, N, N0, N);
     cudaMemcpy(blockSum, dev_blockSum, ((N-1)/TPB + 1)*((N-1)/TPB + 1)*sizeof(double),cudaMemcpyDeviceToHost);
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
    
 /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int flag = (N-1)/TPB;
    
    while (flag > 0){
      double* blockSum2 = new double[(flag/TPB+1)*(flag/TPB+1)*sizeof(double)];
      double* dev_blockSum2;
      cudaMalloc((void **)&dev_blockSum2, (flag/TPB + 1)*(flag/TPB + 1)*sizeof(double));
      gridDim.x = flag/TPB + 1;
      gridDim.y = flag/TPB + 1;
      MatBlockSum <<< gridDim, blockDim>>>(dev_blockSum, dev_blockSum2,flag + 1,flag + 1, (flag/TPB + 1)*TPB);
      cudaMemcpy(blockSum2, dev_blockSum2, (flag/TPB + 1)*(flag/TPB + 1)*sizeof(double),cudaMemcpyDeviceToHost);
      flag = flag/TPB;
      dev_blockSum = dev_blockSum2;
        double totalSum2 = 0;
        //for (int i=0; i<((N/TPB-1)/TPB + 1)*((N/TPB-1)/TPB + 1); i++){
        for (int i=0; i<(flag + 1)*(flag + 1); i++){
            totalSum2 = totalSum2 + blockSum2[i];
            //cout << blockSum[i] << " ";
        }
      cout << blockSum2[0] << endl;
    }
    
    
    
    
    double temp = 0;
    double verifyValue = 0;
    for(int i=0; i<N0; i++)
      for(int j=0; j<N0; j++)
        verifyValue = verifyValue + a[i][j];
    for (int i = 1; i < N0-1; ++i)
    {
        for (int j = 1; j < N0-1; ++j)
        {
            temp = a[i][j]+a[i][j+1]+a[i][j-1]+a[i-1][j]+a[i+1][j];
            if (temp != b[i][j])
            {
                //printf("Failure at %d  %d\n", i, j);
            }
            //cout << c[i][j] << " ";
        }
       //cout << endl;
    }
    double totalSum = 0;
    for (int i=0; i<((N-1)/TPB + 1)*((N-1)/TPB + 1); i++){
      totalSum = totalSum + blockSum[i];
      //cout << blockSum[i] << " ";
    }
    
    cout << endl;
    cout << totalSum << endl;
    //cout << totalSum2 << endl;
    //cout << totalSum3 << endl;
    cout << verifyValue << endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    //return 0;
    //system("pause");
}






