/* --------------------------------------------------------
 * CarClass.cpp
 *
 * 04/10/2017 Hai Zhu
 *
 * Car Class ctor dtor, and member functions.
 ----------------------------------------------------------
 */

#include <iostream>
using namespace std;

#include "CarClass.h" 
#include "constants.h"

//default ctor
CarClass::CarClass()
{
  carArrivalTime = INI_VALUE;
}

//value ctor
CarClass::CarClass(int arrivalTime)
                :carArrivalTime(arrivalTime)
{
  ;
}

//default dtor
CarClass::~CarClass()
{
  ;
}

//member function to get car arrival time
void CarClass::getCarArrivalTime(int &arrivalTime)
{
  arrivalTime = carArrivalTime;
}


