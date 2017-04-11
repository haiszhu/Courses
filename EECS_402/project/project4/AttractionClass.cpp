/* ----------------------------------------------------------------------
 * AttractionClass.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * Attraction Class ctor dtor, and member functions.
 ------------------------------------------------------------------------
 */

#include <iostream>
using namespace std;

//incldue classes and header files for park attraction class
#include "AttractionClass.h"
#include "RiderClass.h"
#include "FIFOQueueClass.h"
#include "LinkedNodeClass.h"
#include "constants.h"

//default ctor
AttractionClass::AttractionClass()
{
  attractionName = "Space Mountain";
  numberOfSeats = NUM_OF_SEATS;
  numberOfPriority = NUM_OF_PRIORITY;
  priorityLevel = new FIFOQueueClass< RiderClass >[NUM_OF_PRIORITY];
  numberOfRider = new int[numberOfPriority];
  
  //loop over to initialize number of rider along each line
  for (int i = 0; i < numberOfPriority; i++)
  {
    numberOfRider[i] = INI_VALUE;
  }
}

//value ctor
AttractionClass::AttractionClass(string attrName, int numOfSeats,
                                 int numOfPriority)
                :attractionName(attrName),
                 numberOfSeats(numOfSeats),
                 numberOfPriority(numOfPriority)
{
  priorityLevel = new FIFOQueueClass< RiderClass >[numOfPriority];
  numberOfRider = new int[numOfPriority];
  for (int i = 0; i < numberOfPriority; i++)
  {
    //loop over to initialize number of rider along each line
    numberOfRider[i] = INI_VALUE;
  }
}

//default dtor
AttractionClass::~AttractionClass()
{
  ;
}

//get current attraction name
void AttractionClass::getAttractionName(string &attrName)
{
  attrName = attractionName;
}

//get number of seats per car, probably unused
void AttractionClass::getNumberOfSeats(int &numOfSeats)
{
  numOfSeats = numberOfSeats;
}

//get number of riders along lineIndex line
void AttractionClass::getNumberOfRider(int &numOfRider,
                                       int lineIndex)
{
  numOfRider = numberOfRider[lineIndex];
}

//add rider to line based on its type
void AttractionClass::addRiderToLine(RiderClass &rider)
{
  int passType;
  int tempNumOfRider;   //current number of riders in line
  rider.getRiderPassType(passType);
  for (int i = 0; i < numberOfPriority; i++)
  {
    //match passType with priority line in attraction class
    if (passType == i)
    {
      priorityLevel[i].enqueue(rider);
      numberOfRider[i] = numberOfRider[i] + 1;
    }
    //print info for observation
    tempNumOfRider = priorityLevel[i].getNumElems();
    cout << "      Now number of riders in line " 
         << i << " is " << tempNumOfRider << endl;
  }
}

//get rider to car each time car event generated
void AttractionClass::getRiderToCar(int idealNum[], int realNum[],
                                    int averArrivalTime[])
{
  int tempIdealNum;   //store ideal number
  int tempNumOfRider;   //store current number of rider
  int increNumOfNextPriority(INI_VALUE);  //increment computation
  int tempArrivalTime(INI_VALUE); //rider arrival time info
  
  RiderClass tempRider;
  //loop over each priority line to get rider in car
  for (int i = 0; i < numberOfPriority; i++)
  {
    tempIdealNum = idealNum[i] + increNumOfNextPriority;
    tempNumOfRider = priorityLevel[i].getNumElems();
    averArrivalTime[i] = INI_VALUE;
    
    //if there are more riders than ideal number
    if (tempNumOfRider >= tempIdealNum)
    {
      for (int j = 0; j < tempIdealNum; j++)
      {
        priorityLevel[i].dequeue(tempRider);
        tempRider.getRiderArrivalTime(tempArrivalTime);
        averArrivalTime[i] = averArrivalTime[i] + tempArrivalTime;
        
        numberOfRider[i] = numberOfRider[i] - 1;
      }
      realNum[i] = tempIdealNum;
      cout << "      Now number of riders in line is " 
           << i << " is " << tempNumOfRider - tempIdealNum << endl;
    }
    //if not enough rider in priority line, give opportunity to next
    else
    {
      for (int j = 0; j < tempNumOfRider; j++)
      {
        priorityLevel[i].dequeue(tempRider);
        tempRider.getRiderArrivalTime(tempArrivalTime);
        averArrivalTime[i] = averArrivalTime[i] + tempArrivalTime;
        numberOfRider[i] = numberOfRider[i] - 1;
      }
      increNumOfNextPriority = tempIdealNum - tempNumOfRider;
      realNum[i] = tempNumOfRider;
      cout << "      Now number of riders in line is " 
           << i << " is " << tempNumOfRider - tempNumOfRider
           << endl;
    }
  }
}


