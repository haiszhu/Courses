/* ----------------------------------------------------------------------
 * test_phase3.cpp
 *
 * 04/10/2017 Hai Zhu
 *
 * This is the main function to generate proj4.exe executable.
 * To generate executable, run Makefile.
 *
 * Functionality explain:
 * This executable will run an event simulation at attraction of a park
 * based on parameters provided in .txt file. You will see results 
 * displayed with the following style:
 
 (every time rider event generated)
 ===============================================
 Rider event of type 0 generated!
 Rider arrival time is 394
 
 Now number of riders in line 0 is 2
 Now number of riders in line 1 is 3
 Now number of riders in line 2 is 5
 
 (every time car event generated with detailed info)
 ===============================================
 Car event generated!
 Car arrival time is 438
 Now number of riders in line is 0 is 0
 Now number of riders in line is 1 is 0
 Now number of riders in line is 2 is 0
 
 Detailed info:
 2 riders take car from line 0!
 Average wait time for them is  45
 3 riders take car from line 1!
 Average wait time for them is  63
 5 riders take car from line 2!
 Average wait time for them is  71
 
 (after simulation is done, average waiting time along each line)
 ===============================================
 ===============================================
 Average Waiting Time is as follows
 Average wait time for line 0 is 37
 Average wait time for line 1 is 50
 Average wait time for line 2 is 72

 *
 * You could get simulation result by run the executable:
 * ./proj4.exe simParams.txt
 *
 ------------------------------------------------------------------------
 */

#include <iostream>
#include <fstream>
using namespace std;

//include constants header file
#include "constants.h"

//include class
#include "RiderClass.h"
#include "CarClass.h"
#include "AttractionClass.h"
#include "FIFOQueueClass.h"

//include gloabl functions
#include "random.h"
#include "readInFromFile.h"
#include "detRiderType.h"

int main(int argc, char* argv[])
{
  //variable initialization--------------------------------------------------
  int currentTime(INI_VALUE);
  int closeTime(INI_VALUE);
  
  int rArrivalTime(INI_VALUE);
  int cArrivalTime(INI_VALUE);
  
  int rArrivalMean(INI_VALUE);
  double rArrivalStddev(INI_VALUE_D);
  
  int cArrivalMin(INI_VALUE), cArrivalMax(INI_VALUE);
  
  int percentSFP(INI_VALUE), percentFP(INI_VALUE);
  int sfpAdmitted(INI_VALUE), fpAdmitted(INI_VALUE);
  
  //simulation related
  int numOfIdealNum(NUM_OF_PRIORITY);
  int idealNumSFP(IDEAL_NUM_SFP);
  int idealNumFP(IDEAL_NUM_FP);
  int idealNumSTD(IDEAL_NUM_STD);
  int idealNum[numOfIdealNum] = {idealNumSFP, idealNumFP, idealNumSTD};
  
  int realNum[numOfIdealNum];
  int averArrivalTime[numOfIdealNum];
  int averWaitTime[numOfIdealNum];
  int incrNum[numOfIdealNum];
  int longestLine[numOfIdealNum];
  int tempNumOfRider(INI_VALUE);
  
  for (int i = 0; i < numOfIdealNum; i++)
  {
    realNum[i] = INI_VALUE;
    averArrivalTime[i] = INI_VALUE;
    averWaitTime[i] = INI_VALUE;
    incrNum[i] = INI_VALUE;
    longestLine[i] = INI_VALUE;
  }
  
  //Read in parameters from file---------------------------------------------  
  string fname(argv[INI_FNAME]);
  readInFromFile(fname, closeTime, rArrivalMean,
                 rArrivalStddev, cArrivalMin,
                 cArrivalMax, percentSFP, percentFP,
                 sfpAdmitted, fpAdmitted);
  
  //Start Simulation---------------------------------------------------------
  int seedValue = INI_VALUE;
  int riderType(INI_VALUE);
  int riderTypeID(INI_VALUE);
  RiderClass* riderPtr;
  CarClass* carPtr;
  AttractionClass attraction;
  
  setSeed(seedValue);
  rArrivalTime = getNormal(rArrivalMean, rArrivalStddev);
  cArrivalTime = getUniform(cArrivalMin, cArrivalMax);
  riderTypeID = getUniform(UNI_MIN,UNI_MAX);
  
  
  //initialize the first event and set current time-------------------------
  if (rArrivalTime < cArrivalTime)
  {
    currentTime = rArrivalTime; 
  } 
  else
  {
    currentTime = cArrivalTime;
  }
  
  //this function needs to be modified if more lines created
  riderType = detRiderType(riderTypeID, percentSFP, percentFP);
  
  //keep simulation while it is not closed---------------------------------
  while (currentTime <= closeTime)
  {
    if (rArrivalTime <= cArrivalTime && rArrivalTime <= closeTime)
    {
      //update event and current time
      riderPtr = new RiderClass(rArrivalTime, riderType);
      currentTime = rArrivalTime;
      
      cout << " " << endl;
      cout << "(Rider Event)" << endl;
      cout << "===============================================" << endl;
      cout << "Rider event of type " << riderType << " generated!" << endl;
      cout << "  Rider arrival time is " << currentTime << endl;
      cout << " " << endl;
      attraction.addRiderToLine(*riderPtr);
      
      //update car arrival time and rider arrival time
      rArrivalTime = rArrivalTime + getNormal(rArrivalMean, rArrivalStddev);
      riderTypeID = getUniform(UNI_MIN,UNI_MAX);
      riderType = detRiderType(riderTypeID, percentSFP, percentFP);
      
      
      attraction.getNumberOfRider(tempNumOfRider, riderType);
      if (tempNumOfRider > longestLine[riderType])
      {
        longestLine[riderType] = tempNumOfRider;
      }
      
    }
    else if (cArrivalTime <= rArrivalTime && cArrivalTime <= closeTime)
    {
      
      carPtr = new CarClass(cArrivalTime);
      
      //update current time
      currentTime = cArrivalTime;
      
      cout << " " << endl;
      cout << "(Car Event with detailed info)" << endl;
      cout << "===============================================" << endl;
      cout << "Car event generated!" << endl;
      cout << "  Car arrival time is " << currentTime << endl;
      cout << " " << endl;
      
      //attraction.addRiderToLine(*riderPtr);
      attraction.getRiderToCar(idealNum, realNum, averArrivalTime);
      
      //statistics
      cout << " " << endl;
      cout << "    Detailed info: " << endl;
      for (int i = 0; i < numOfIdealNum; i++)
      {
        if (realNum[i])
        {
          averWaitTime[i] = averWaitTime[i] + currentTime * realNum[i] - averArrivalTime[i];
          incrNum[i] = incrNum[i] + realNum[i];
          
          cout << "      " << realNum[i] << " riders take car from line " 
               << i << "! " << endl;
          cout << "        " << "Average wait time for them is  "
               << (currentTime * realNum[i] - averArrivalTime[i])/realNum[i] 
               << " " << endl;
        }
      }
      
      cArrivalTime = cArrivalTime + getUniform(cArrivalMin, cArrivalMax);
    }
    else
    {
      if (cArrivalTime <= rArrivalTime)
      {
        currentTime = rArrivalTime;
      }
      else
      {
        currentTime = cArrivalTime;
      }
    }
    
  }
  
  //keep simulation when there are still riders waiting in line
  int statusCheck(INI_VALUE);
  int numOfRider(INI_VALUE);
  for (int i = 0; i < numOfIdealNum; i++)
  {
    attraction.getNumberOfRider(numOfRider, i);
    statusCheck = statusCheck + numOfRider;
  }
  
  while (statusCheck)
  {
    statusCheck = INI_VALUE;
    carPtr = new CarClass(cArrivalTime);
    
    //update current time
    currentTime = cArrivalTime;
      
    //update car arrival time and rider arrival time
    cArrivalTime = cArrivalTime + getUniform(cArrivalMin, cArrivalMax);
    
    cout << " " << endl;
    cout << "(Car Event with detailed info)" << endl;  
    cout << "===============================================" << endl;
    cout << "Car event generated!" << endl;
    cout << "Car arrival time is " << currentTime << endl;
    
    //attraction.addRiderToLine(*riderPtr);
    attraction.getRiderToCar(idealNum, realNum, averArrivalTime);
    
    //statistics
    cout << " " << endl;
    cout << "    Detailed info: " << endl;
    for (int i = 0; i < numOfIdealNum; i++)
    {
      if (realNum[i])
      {
        averWaitTime[i] = averWaitTime[i] + currentTime * realNum[i] - averArrivalTime[i];
        incrNum[i] = incrNum[i] + realNum[i];
        
        cout << "    " << realNum[i] << " riders take car from line " 
             << i << "! " << endl;
        cout << "      " << "Average wait time for them is  "
             << (currentTime * realNum[i] - averArrivalTime[i])/realNum[i] 
             << " " << endl;
      }
      attraction.getNumberOfRider(numOfRider, i);
      statusCheck = statusCheck + numOfRider;
    }
  }
  
  //Final statistic info print----------------------------------------------
  cout << " " << endl;
  cout << "(Simulation is done, results on average waiting time, etc)" 
       << endl;
  cout << "===============================================" << endl;
  cout << "===============================================" << endl;
  cout << "Average Waiting Time is as follows:" << endl;
  for (int i = 0; i < numOfIdealNum; i++)
  {
    cout << "  Average wait time for line " << i
         << " is " << averWaitTime[i]/incrNum[i] << "!" << endl;
  }
  cout << "Longest Waiting line is as follows:" << endl;
  for (int i = 0; i < numOfIdealNum; i++)
  {
    cout << "  Longest line for pass type " << i
         << " has " << longestLine[i] << " riders! " << endl;
  }
  
  
}
