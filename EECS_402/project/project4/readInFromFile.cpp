/* ----------------------------------------------------------------------
 * readInFromFile.cpp
 *
 * 04/10/2017 Hai Zhu
 *
 * global function to read in simulation parameters in .txt file
 ------------------------------------------------------------------------
 */

#include <iostream>
#include <fstream>
using namespace std;

void readInFromFile(string &fname, int &closeTime, int &rArrivalMean,
                    double &rArrivalStddev, int &cArrivalMin,
                    int &cArrivalMax, int &percentSFP, int &percentFP,
                    int &sfpAdmitted, int &fpAdmitted)
{
  ifstream inFile;
  inFile.open(fname.c_str());
  inFile >> closeTime >> rArrivalMean >> rArrivalStddev 
         >> cArrivalMin >> cArrivalMax
         >> percentSFP >> percentFP
         >> sfpAdmitted >> fpAdmitted;
  inFile.close();
  
}
