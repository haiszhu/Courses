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
    AttractionClass();                                  //default ctor
    AttractionClass(string attrName, int numOfSeats, int numOfPriority);   //value ctor
    ~AttractionClass();                                 //dtor
  
    void getAttractionName(string &attrName);
    void getNumberOfSeats(int &numOfSeats);
    void getNumberOfRider(int &numOfRider, int lineIndex);
    
    void addRiderToLine(RiderClass &rider);
    void getRiderToCar(int idealNum[], int realNum[], int averArrivalTime[]);
    
    //FIFOQueueClass< RiderClass >* priorityLevel; //line for SFP riders
    //int* numberOfRider;
  private:
    string attractionName;                 //int results of return value
    int numberOfSeats;
    int numberOfPriority;
    int* numberOfRider;
    FIFOQueueClass< RiderClass >* priorityLevel; //line for SFP riders
    
    //FIFOQueueClass< int >* priorityLevel; 
    //LinkedNodeClass< int > *head;
    

};
#endif