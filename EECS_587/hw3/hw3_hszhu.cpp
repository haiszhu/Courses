/* ----------------------------------------------------------------------
 * hw3_hszhu.cpp
 *
 * 10/10/2017 Hai Zhu
 *
 * EECS587 Parallel Computing Class, programming assignment
 * template calculation on an n x n matrix, where each entry are long int
 * 
 * A[i,j] depends on previous entry [i,j], [i+1,j], [i,j+1] and [i+1,j+1]
 * To run this programm on flux
 *    $ module load intel/17.0.1
 *    $ module load openmpi/1.10.2/intel/17.0.1
 *    $ mpic++ hw3_hszhu.cpp -o hw3_hszhu
 *    $ mpirun -np 4 ./hw3_hszhu 1000
 * where 1000 is the size of matrix A, and -np needs to perfect square
 ------------------------------------------------------------------------
 */

#include<iostream>
#include<cstdlib>
#include<mpi.h>
#include<math.h>

#include "f.h"
using namespace std;

int main(int argc, char *argv[])
{
   
  char *nTemp;
  if(argc < 2){
    cout << "You must provide at least one argument!" << endl;
  }
  for(int i=1; i<argc; i++){
    nTemp = argv[i];
  }
  
  int procID;
  int ierr;
  int p;
  double wtime;
  
  ierr = MPI_Init ( &argc, &argv );
  ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );
  ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &procID );
  
  MPI_Status status;
  MPI_Request  send_request,recv_request;
  
  if( p != sqrt(p)*sqrt(p) ){
    if( procID == 0 ){ 
      cout << "Num of processor should be perfect square, i.e. 4, 9, 16, 36."<< endl;
    }
    exit(0);
  }
  
  //index 
  int n = atoi(nTemp);                //take matrix size as input
  int idx = floor(procID/sqrt(p));    //the row current proc is in the proc grid
  int idy = procID - sqrt(p)*idx;     //the column current proc is in the grid
  
  int nLocal = ceil(n/sqrt(p));       //submatrix size(roughly, not all are squares)
  int nLocalx, nLocaly;               //exact submatrix num of rows and columns 
  
  if( idx == sqrt(p)-1){              //last row of proc modification for nLocal
    nLocalx = n - nLocal*(sqrt(p)-1);
    if( idy == sqrt(p)-1 ){
      nLocaly = nLocalx;
    } else{
      nLocaly = nLocal;
    }
  } else if( idy == sqrt(p)-1 ){      //last column of proc modification 
    nLocalx = nLocal;
    nLocaly = n - nLocal*(sqrt(p)-1);
  } else{                             //upper left procs are all squares
    nLocalx = nLocal;
    nLocaly = nLocal;
  }
  
  
  //declare submatrix, row, and column vector
  long long** matrixALocal = new long long*[nLocalx+1];
  long long* aCol = new long long[nLocalx]();
  long long* aRow = new long long[nLocaly]();
  long long aEntry = 0;
  int iGlobal, jGlobal;

  //initialization of Alocal, aRow, aCol, and aEntry
  for(int i=0; i<nLocalx; i++){
    matrixALocal[i] = new long long[nLocaly+1];
    for(int j=0; j<nLocaly; j++){
      iGlobal = idx*nLocal+i;         //global row index
      jGlobal = idy*nLocal+j;         //global column index
      matrixALocal[i][j] = iGlobal + jGlobal*n;
    }
    aCol[i] = matrixALocal[i][0];
  }
  matrixALocal[nLocalx] = new long long[nLocaly+1];
  for(int j=0; j<nLocaly; j++){
    aRow[j] = matrixALocal[0][j];
  }
  aEntry = matrixALocal[0][0];
  
  if( procID == 0 ){
    wtime = MPI_Wtime();
  }
  
  //10 iteration
  for(int iter=0; iter<10; iter++){
    //MPI_Send
    if( idx >= 1 ){
      ierr = MPI_Isend(&aRow[0], nLocaly, MPI_LONG_LONG, (idx-1)*sqrt(p)+idy, 666, MPI_COMM_WORLD,&send_request);
      if( idy >= 1 ){
        ierr = MPI_Isend(&aEntry, 1, MPI_LONG_LONG, (idx-1)*sqrt(p)+idy-1, 666, MPI_COMM_WORLD,&send_request);
      }
    }
    if( idy >= 1 ){
      ierr = MPI_Isend(&aCol[0], nLocalx, MPI_LONG_LONG, procID-1, 666, MPI_COMM_WORLD,&send_request);
    }
    //MPI_Recv
    if( idx < sqrt(p)-1 ){
      ierr = MPI_Recv(aRow, nLocaly, MPI_LONG_LONG, (idx+1)*sqrt(p)+idy, 666, MPI_COMM_WORLD, &status);
      if( idy < sqrt(p)-1 ){
        ierr = MPI_Recv(&aEntry, 1, MPI_LONG_LONG, (idx+1)*sqrt(p)+idy+1, 666, MPI_COMM_WORLD, &status);
      }
    }
    if( idy < sqrt(p)-1 ){
      ierr = MPI_Recv(aCol, nLocalx, MPI_LONG_LONG, procID+1, 666, MPI_COMM_WORLD, &status);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  
    //fiugre out which entry stay put in submatrix by modifying nLocal
    int idxILow(0), idyJLow(0);
    int idxIHi(nLocalx), idyJHi(nLocaly);
    if( idx == 0 ){
      idxILow = 1;
      matrixALocal[1][nLocaly] = aCol[1];
    }
    if( idy == 0 ){
      idyJLow = 1;
    }
    if( idx == sqrt(p)-1 ){
      idxIHi = nLocalx-1;
      matrixALocal[nLocalx-1][nLocaly] = aCol[nLocalx-1];
    }
    if( idy == sqrt(p)-1 ){
      idyJHi = nLocaly-1;
    }
    matrixALocal[nLocalx][nLocaly] = aEntry;
    for(int j=0; j<nLocaly; j++){
      matrixALocal[nLocalx][j] = aRow[j];
    }
    matrixALocal[0][nLocaly] = aCol[0];         //last column
    for(int i=idxILow; i<idxIHi; i++){
      if(i<nLocalx-1){
        matrixALocal[i+1][nLocaly] = aCol[i+1]; //(i+1)th row, last column
      }
      for(int j=idyJLow; j<idyJHi; j++){
        matrixALocal[i][j] = f( matrixALocal[i][j], matrixALocal[i+1][j], matrixALocal[i][j+1], matrixALocal[i+1][j+1]);
      }
      aCol[i] = matrixALocal[i][0];             //ith row, first column, to be sent
    }
    for(int j=0; j<nLocaly; j++){
      aRow[j] = matrixALocal[0][j];
    }
    aEntry = matrixALocal[0][0];
    
  }
  
  //result output
  if( procID == 0){
    wtime = MPI_Wtime ( ) - wtime;
    cout << "==========================================================================" << endl;
    cout << "  Matrix A size is " << n << " X " << n << " running on " << p << " processors." << endl;
    cout << "  Elapsed wall clock time = " << wtime << " seconds." << endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  //compute local sum
  long long sumLocal(0);
  for(int i=0; i<nLocalx; i++){
    for(int j=0; j<nLocaly; j++){
      iGlobal = idx*nLocal+i;
      jGlobal = idy*nLocal+j;
      if( (iGlobal == n/2) && (jGlobal == n/2) ){
        cout << "  A[n/2][n/2] value is " << "A["<< n/2 << "," << n/2 << "]=" << matrixALocal[i][j] << " from proc " << procID << "."<< endl;
      }
      sumLocal = sumLocal + matrixALocal[i][j];
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  //send local submatrix sum to node 0
  long long sum(0), temp(0);
  if( procID > 0 ){
    ierr = MPI_Isend(&sumLocal, 1, MPI_LONG_LONG, 0, 666, MPI_COMM_WORLD,&send_request);
  }
  if( procID == 0 ){
    sum = sum + sumLocal;
    for(int i=1; i<p; i++){
      ierr = MPI_Recv(&temp, 1, MPI_LONG_LONG, i, 666, MPI_COMM_WORLD, &status);
      sum = sum + temp;
    }
    cout << "  All sum is " << sum << endl;
  }
  
  MPI_Finalize ( );
}