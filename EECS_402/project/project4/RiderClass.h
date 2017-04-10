/* ----------------------------------------------------------------------
 * RiderClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * Rider Class header file
 ------------------------------------------------------------------------
 */

#ifndef RIDERCLASS_H
#define RIDERCLASS_H

class RiderClass
{
  public:
    RiderClass();                                  //default ctor
    RiderClass(int arrivalTime, int passType);             //value ctor
    ~RiderClass();                                 //dtor
  
    void getRiderArrivalTime(int &arrivalTime);
    void getRiderPassType(int &passType);
    void getValue(int &riderInfo);
    void setWaitTime(int &currentTime);
    void getWaitTime(int &waitTime);
     
  private:
    int riderArrivalTime;                           //int results of return value 
    int riderWaitTime;
    int riderPassType;                                      //image height

};
#endif
