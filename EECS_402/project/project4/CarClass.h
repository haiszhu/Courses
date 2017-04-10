/* ----------------------------------------------------------------------
 * CarClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * Car Class header file
 ------------------------------------------------------------------------
 */

#ifndef CARCLASS_H
#define CARCLASS_H
#include "CarClass.h" 
#include "constants.h"

class CarClass
{
  public:
    CarClass();                                  //default ctor
    CarClass(int arrivalTime);             //value ctor
    ~CarClass();                                 //dtor
  
    void getCarArrivalTime(int &arrivalTime);
    
  private:
    int carArrivalTime;                           //int results of return value

};
#endif