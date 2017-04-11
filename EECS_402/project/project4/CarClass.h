/* --------------------------------------------------------------
 * CarClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * Car Class header file
 ----------------------------------------------------------------
 */

#ifndef CARCLASS_H
#define CARCLASS_H
#include "CarClass.h" 
#include "constants.h"

class CarClass
{
  public:
    CarClass();                               //default ctor
    CarClass(int arrivalTime);                //value ctor
    ~CarClass();                              //dtor
  
    //private variable related
    void getCarArrivalTime(int &arrivalTime); //ger arrival time
    
  private:
    int carArrivalTime;                       //car arrival time

};
#endif
