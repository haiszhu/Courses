/* ----------------------------------------------------------------------
 * RiderClass.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * Rider Class ctor dtor, and member functions.
 ------------------------------------------------------------------------
 */

#include <iostream>
using namespace std;

#include "RiderClass.h" 
#include "constants.h"

//default ctor
RiderClass::RiderClass()
{
  riderArrivalTime = INI_VALUE;
  riderPassType = INI_VALUE;
  riderWaitTime = INI_VALUE;
}

//value ctor
RiderClass::RiderClass(int arrivalTime, int passType)
                :riderArrivalTime(arrivalTime), 
                 riderPassType(passType)
{
  riderWaitTime = INI_VALUE;
}

//default dtor
RiderClass::~RiderClass()
{
  ;
}

//get rider arrival time
void RiderClass::getRiderArrivalTime(int &arrivalTime)
{
  arrivalTime = riderArrivalTime;
}

//get rider pass type
void RiderClass::getRiderPassType(int &passType)
{
  passType = riderPassType;
}

//get info used for queue printing
void RiderClass::getValue(int &riderInfo)
{
  riderInfo = riderArrivalTime;
}

//set rider class waiting time variable
void RiderClass::setWaitTime(int &currentTime)
{
  riderWaitTime = currentTime - riderArrivalTime;
}

//get rider class waiting time
void RiderClass::getWaitTime(int &waitTime)
{
  waitTime = riderWaitTime;
}
