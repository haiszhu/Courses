/* ----------------------------------------------------------------------
 * hw4_hszhu.cpp
 *
 * 10/29/2017 Hai Zhu
 *
 * EECS587 Parallel Computing Class, programming assignment
 * compute max of continuous differentiable function
 *
 * g(x) has bounded derivative, and max was found by dividing interval
 * into halves if there is a possibility that max can be updated. Otherwise
 * discard the interval
 * To compile and run this programm on PSC bridges
 *    $ g++ -fopenmp hw4_hszhu.cpp -o hw4_hszhu (with GNU compiler)
 *    $ icpc -qopenmp hw4_hszhu.cpp -o hw4_hszhu (with Intel compiler)
 *    $ icpc -qopenmp -xHost -O2 hw4_hszhu.cpp -o hw4_hszhu (with Intel compiler, and optimization option)
 *    $ export OMP_NUM_THREAD = 28
 *    $ ./hw4_hszhu 1 100 0.000001 12
 * where 1, 100 are left and right end point of the interval, and 0.000001
 * is the tolerance for max, and 12 is the bound on abs value of derivative.
 * Further modification can be made to use less memory maybe using priority queue
 * data structure.
 ------------------------------------------------------------------------
 */

#include <omp.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <float.h>
using namespace std;

#include "g.h"
//double g(double x);

int main(int argc, char *argv[]){
  double temp[4];
  if(argc < 5){
    cout << "You must provide interval end points, tolerance, and bounds on derivative" << endl;
    cout << "For example, ./hw4_hszhu 1 100 0.000001 12 " << endl;
   // exit(0);
  }
  for(int i=1; i<argc; i++){
    temp[i-1] = atof(argv[i]);
  }
  
  int numel = (temp[1] - temp[0])*temp[3]/temp[2];
  cout << "number of double precision requested " << numel << endl;
  double M(-DBL_MAX);
  double *leftNode = new double[numel];
  double *rightNode = new double[numel];
  double *gvalL = new double[numel];
  double *gvalR = new double[numel];
  
  //initialize first interval
  double eps(temp[2]), gprime(temp[3]);
  leftNode[0] = temp[0]; rightNode[0] = temp[1];
  gvalL[0] = g(leftNode[0]); gvalR[0] = g(rightNode[0]);
  
  //local private
  int headIdx(0), tailIdx(1), numOfInterval(1), numIncrement(0);
  double left, right, mid, leftVal, rightVal, midVal;
  bool nonEmpty = true;
  
  int numOfCores = omp_get_num_procs ( );
  
  cout << "Number of procs used is " << numOfCores << endl;
  
  double x(0);
  M = max(M,max(gvalL[0],gvalR[0]));
  //if( M==gvalL[0] ){
  //  x =
  //}
  
  double seconds = omp_get_wtime ( );
  //continually apply the max possible checking procedure until no further refinement required
  while( nonEmpty ){
    
    numIncrement = 0; //num of intervals increased to update tail index
    
    //start parallelize each level of intervals (each interval length gets divided by half compare to last while loop)
    #pragma omp parallel for private(left,right,mid,leftVal,rightVal,midVal) reduction(+:numIncrement)
    for(int i=0; i < numOfInterval; i++){
      int localHead = (headIdx + i) % numel;  //head index for each thread, manually assign job
      left = leftNode[localHead];
      right = rightNode[localHead];
      leftVal = gvalL[localHead];
      rightVal = gvalR[localHead];
      if((leftVal + rightVal + gprime*(right-left))/2 >= M+eps){
        mid = (left + right)/2;
        midVal = g(mid);
        int localTail(0);
        numIncrement += 2;
        #pragma omp critical
        {
          localTail = tailIdx;
          tailIdx = (tailIdx + 2) % numel;
          M = max(M,midVal);
        }
        leftNode[localTail] = left;
        rightNode[localTail] = mid;
        gvalL[localTail] = leftVal;
        gvalR[localTail] = midVal;
        leftNode[localTail+1] = mid;
        rightNode[localTail+1] = right;
        gvalL[localTail+1] = midVal;
        gvalR[localTail+1] = rightVal;
      }
    }
    if(numIncrement == 0){
      nonEmpty = false;
    }
    headIdx = headIdx + numOfInterval;  //update new head index
    numOfInterval = numIncrement;
    //cout << "number of intervals " << numOfInterval << endl;
    //cout << "headIdx is " << headIdx << endl;
    //cout << "tailIdx is " << tailIdx << endl;
    //cout << endl;
  }
  
  cout.precision(17);
  cout << "=================================== " << endl;
  cout << "max value is " << M << endl;
  seconds = omp_get_wtime ( ) - seconds;
  cout << "wall time is " << seconds << endl;
  cout << endl;
  
}


