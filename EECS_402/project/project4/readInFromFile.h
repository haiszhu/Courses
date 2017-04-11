/* -------------------------------------------------------------------
 * readInFromFile.h
 *
 * 04/10/2017 Hai Zhu
 *
 * header file forglobal function to read in simulation parameters
 ---------------------------------------------------------------------
 */

#ifndef _READINFROMFILE_H_
#define _READINFROMFILE_H_

#include <iostream>
#include <fstream>
using namespace std;

void readInFromFile(string &fname, int &closeTime, int &rArrivalMean,
                    double &rArrivalStddev, int &cArrivalMin,
                    int &cArrivalMax, int &percentSFP, int &percentFP,
                    int &sfpAdmitted, int &fpAdmitted);

#endif 



