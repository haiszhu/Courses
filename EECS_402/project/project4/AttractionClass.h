/* ----------------------------------------------------------------------
 * AttractionClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * Attraction Class header file
 ------------------------------------------------------------------------
 */

//SortedListClass< T > &rhs

#ifndef ATTRACTIONCLASS_H
#define ATTRACTIONCLASS_H

#include "FIFOQueueClass.h"
#include "RiderClass.h"
//#include <iostream>
//#include "LinkedNodeClass.h"


class AttractionClass
{
  public:
    AttractionClass();                                 //default ctor
    AttractionClass(string attrName, int numOfSeats,
                    int numOfPriority);                //value ctor
    ~AttractionClass();                                //dtor
  
    //private variable related
    void getAttractionName(string &attrName);          //get attr name
    void getNumberOfSeats(int &numOfSeats);           //get num of seats
    void getNumberOfRider(int &numOfRider,
                          int lineIndex);             //get num of rider
  
    //event operation
    void addRiderToLine(RiderClass &rider);          //add rider to line
    void getRiderToCar(int idealNum[], int realNum[],
                       int averArrivalTime[]);       //get rider on car
    
  
  private:
    string attractionName;                        //string of attr name
    int numberOfSeats;                            //num of seats of attr
    int numberOfPriority;                         //pass type
    int* numberOfRider;                           //ptr to num of riders
    FIFOQueueClass< RiderClass >* priorityLevel; //line for SFP riders
  

};
#endif
