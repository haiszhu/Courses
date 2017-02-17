//histogram implementation
//02/17/2017 Hai Zhu
#include <iostream>
using namespace std;

//declare max number of samples, bins, input exit value, and uninitialize value
#define MAX_SAMPLES 100
#define MAX_BINS 100
#define EXIT_VALUE -99999
#define UNINI_VALUE -1
#define MAX_INPUT 1000

//user interface prompt function
int promptUserAndGetChoice();

//sample class describing input data
class SamplingClass
  {
    public:
      bool readFromKeyboard();
      bool printToScreen();
      int getSampleValues(int n);
      int getNumOfValues();
    
      SamplingClass();
      SamplingClass(const char inIdChar, const int inNumSamples,
                  const int inputSamples[]);

    private:
      char charID;
      int numOfValues;
      int sampleValues[MAX_SAMPLES];
  };

//histogram class recording bins
class HistogramClass
  {
    public:
      bool setupHistogram();
      bool addDataToHistogram(SamplingClass &sampling);
      bool printHistogramCounts();
      bool displayHistogram();
    
      HistogramClass();

    private:
      int minBinValue;
      int maxBinValue;
      int numbBins;
      int sumBinNum = UNINI_VALUE; //sum of bin numbers
      int histoBinCounts[MAX_BINS + 2];
  };


//****************************************************************************
#ifdef ANDREW_TEST
#include "andrewTest.h"
#else
int main()
  {
    int userChoice;
    SamplingClass samplingSet;
    HistogramClass histogramSet;

    do
      {
        promptUserAndGetChoice();
        cout << "Your Choice: ";
        cin >> userChoice;

        if (userChoice == 1)  //enter data value
          {
            if (samplingSet.readFromKeyboard())
              cout << "Last Operation Successful: YES \n\n";
            else
              cout << "Last Operation Successful: NO \n\n";
          }
        else if (userChoice == 2) //print current sample set
          {
            if (samplingSet.printToScreen())
              cout << "Last Operation Successful: YES \n\n";
            else
              cout << "Last Operation Successful: NO \n\n";
          }
        else if (userChoice == 3) //set up histogram
          {
            if (histogramSet.setupHistogram())
              cout << "Last Operation Successful: YES \n\n";
            else
              cout << "Last Operation Successful: NO \n\n";
          }
        else if (userChoice == 4) //add current sample to histogram
          {
            if (histogramSet.addDataToHistogram(samplingSet))
              cout << "Last Operation Successful: YES \n\n";
            else
              cout << "Last Operation Successful: NO \n\n";
          }
        else if (userChoice == 5) //print bin counts
          {
            if (histogramSet.printHistogramCounts())
              cout << "Last Operation Successful: YES \n\n";
            else
              cout << "Last Operation Successful: NO \n\n";
          }
        else if (userChoice == 6) //view histogram in graphical form
          {
            if (histogramSet.displayHistogram())
              cout << "Last Operation Successful: YES \n\n";
            else
              cout << "Last Operation Successful: NO \n\n";
          }
        else if (userChoice != 0) //invalid input
          {
            cout << "Sorry that is an invalid menu choice - try again!\n\n";
          }

      } while (userChoice != 0);
    cout << "Thanks for using the histogram generator!\n "
       << "Last Operation Successful: YES" << endl;
  
    return 0;
  }
#endif




//************************************************************************
//global user option prompt
int promptUserAndGetChoice()
  {
    cout << "1. Enter a sample set of data values\n";
    cout << "2. Print the contents of the current sample set\n";
    cout << "3. Reset / Provide values for setting up a histogram\n";
    cout << "4. Add the contents of current sample set to histogram\n";
    cout << "5. Print bin counts contained in histogram\n";
    cout << "6. View the histogram in graphical form\n";
    cout << "0: Exit the program\n\n";
    return 0;
  }

//************************************************************************

//sampling class constructor and member functions
//constructor to initialize charID
SamplingClass::SamplingClass() //initializing charID
  {
    charID = '0';
  }

SamplingClass::SamplingClass(const char inIdChar,
                             const int inNumSamples,
                             const int inputSamples[])
  {
    int i = 0;
  
    if (numOfValues > MAX_SAMPLES)
      {
        cout << "ERROR: Number of input sample is larger than " << MAX_SAMPLES << "\n";
        charID = '0';
      }
    else
      {
        charID = inIdChar;
        numOfValues = inNumSamples;
        for (i = 0; i < inNumSamples; i++)
          {
            sampleValues[i] = inputSamples[i];
          }
      }
  }

//read keyboard input samples
bool SamplingClass::readFromKeyboard()
  {
    int i = 0;
    cout << "Enter character identifier for this sample: ";
    cin >> charID;
    cout << "Enter all samples, then enter " << EXIT_VALUE << " to end:\n";
    cin >> sampleValues[i];
    if (sampleValues[i] == EXIT_VALUE)  //if the first input is asking for exit
      {
        cout << "ERROR: Exit with no sample input!\n";
        numOfValues = 0;
        return false;
      }
    else
      {
        i++;
  
        while ((i < MAX_INPUT) && (sampleValues[i-1] != EXIT_VALUE))
          //read input within max num of input, and as long as not exiting
          {
            cin >> sampleValues[i];
            i++;
          }
        if (i <= MAX_SAMPLES + 1)
          {
            numOfValues = i - 1;
            return true;
          }
        else
          {
            cout << "ERROR: Can not take more than " << MAX_SAMPLES
                 << " samples!\n";
            numOfValues = 0;
            return false;
          }
      }
  }

//print samples to screen if samples are nonempty
bool SamplingClass::printToScreen()
  {
    if (numOfValues == 0) //not enough or too many input
      {
        cout << "ERROR: Can not print uninitialized sampling!\n";
        return false;
      }
    else  //perform accordingly, print valid samples
      {
        cout << "Data stored for sampling with identifier " << charID << ":\n";
        cout << "Total samples:" << numOfValues << "\n";
        cout << "Samples (5 samples per line):\n";
        int i;
    
        for(i=0; i<numOfValues;i++)
          {
            cout << sampleValues[i] << " ";
            if (((i+1) % 5) == 0)
              cout << endl;
          }
        cout << endl;
        return true;
      }
  }

//member function to get nth value in sample
int SamplingClass::getSampleValues(int n)
  {
    return sampleValues[n];
  }
//member function to get num of samples
int SamplingClass::getNumOfValues()
  {
    return numOfValues;
  }


//histogram class constructor and member functions
//default histogram constructor
HistogramClass::HistogramClass()
  {;}
//allow user to perform histogram bin setup
bool HistogramClass::setupHistogram()
  {
    int i;
    int histoBinLen;
    int diff; //measure whether evenly divided
    sumBinNum = 0;
    cout << "Enter minimum value: ";
    cin >> minBinValue;
    cout << "Enter maximum value: ";
    cin >> maxBinValue;
  
    if (minBinValue < maxBinValue)  //perform setup if input makes sense
      {
        cout << "Enter number of bins: ";
        cin >> numbBins;
  
        histoBinLen = (maxBinValue - minBinValue + 1)/numbBins;
        diff = histoBinLen*numbBins - (maxBinValue - minBinValue + 1);
  
        if (diff == 0)  //perform setup if evenly divided
          {
            if (numbBins <= MAX_BINS) //perform setup if num of bins within limit
              {
                for (i = 0; i < numbBins; i++)
                  {
                    histoBinCounts[i] = 0;
                  }
                return true;
              }
            else  //user asks too many bins
              {
                cout << "Sorry, the maximum amount of bins allowed is "
                     << MAX_BINS << ". Try again!\n";
                sumBinNum = -1;
                return false;
              }
          }
        else  //not evenly divided
          {
            cout << "ERROR: Num bins must evenly divide specified range.\n";
            sumBinNum = -1;
            return false;
          }
      }
    else  //min > max, not make sense
      {
        cout << "ERROR: max value must be greater than min value!\n";
        sumBinNum = -1;
        return false;
      }
  }
//allow user to add current sample to histogram bin counts
bool HistogramClass::addDataToHistogram(SamplingClass &sampling)
  {
    int histoBinLen;  //to compute which bin should sample been put into
    int sampleValue;  //current sample value
    int sampleNum;    //num of samples
    int i;
    int idx;          //bin index
  
    sampleNum = sampling.getNumOfValues();
  
    histoBinLen = (maxBinValue - minBinValue + 1)/numbBins;
  
    if (sumBinNum == -1) //check if histogram were initialized
      {
        cout << "ERROR: Can not add samples to uninitialized histogram!\n";
        return false;
      }
    else
      {
        if (sampleNum != 0) //perform if indeed there are samples
          {
            for (i = 0; i < sampleNum; i++) //loop over all samples
              {
                sampleValue = sampling.getSampleValues(i);
                if (sampleValue < minBinValue)  //add to first bin
                  {
                    histoBinCounts[0]++;
                  }
                else if (sampleValue > maxBinValue) //add to last bin
                  {
                    histoBinCounts[numbBins+1]++;
                  }
                else
                  {
                    idx = (sampleValue-minBinValue)/histoBinLen + 1;
                    histoBinCounts[idx]++;
                  }
              }
            sumBinNum = sumBinNum + sampling.getNumOfValues();
            return true;
          }
        else
          return false;
      }
  }

//display histogram
bool HistogramClass::displayHistogram()
  {
    int i, j; //i for bin loop, and j for loop of '=' print
    int numBar;
  
    if (sumBinNum == -1) //check if histogram were initialized
      {
        cout << "ERROR: Can not display uninitialized histogram!\n";
        return false;
      }
    else
      {
        i = 0;
        numBar = 100*histoBinCounts[i]/(2*sumBinNum); //num of '='
        cout << "Bin <min: "; //grapical view for min bin
        for (j = 0; j < numBar; j++)
          {
            cout << "=" ;
          }
        cout << "\n";
  
        for (i = 1; i <= numbBins; i++)
          {
            numBar = 100*histoBinCounts[i]/(2*sumBinNum);
            cout << "Bin    " << i << ": ";
            for (j = 0; j < numBar; j++)
              {
                cout << "=" ;
              }
            cout << "\n";
          }
  
        numBar = 100*histoBinCounts[i]/(2*sumBinNum);
        cout << "Bin >max: "; //graphical view for max bin
        for (j = 0; j < numBar; j++)
          {
            cout << "=" ;
          }
        cout << "\n";
        return true;
      }
  }

//print histogram
bool HistogramClass::printHistogramCounts()
  {
    int i;
    double percent;

    if (sumBinNum == -1) //check if histogram were initialized
      {
        cout << "ERROR: Can not display uninitialized histogram!\n";
        return false;
      }
    else
      {
        i = 0;
        if (sumBinNum == 0) //deal with initialized but no data situation
          percent = 0;
        else
          percent = histoBinCounts[i]/static_cast<double>(sumBinNum)*100;
        cout << "Bin <min: " << histoBinCounts[i] << "  (" << percent << "%)\n";
  
        for (i = 1; i <= numbBins; i++)
          {
            if (sumBinNum == 0)
              percent = 0;
            else
              percent = histoBinCounts[i]/static_cast<double>(sumBinNum)*100;
            cout << "Bin    " << i << ": " << histoBinCounts[i] << "  ("
                 << percent << "%)\n";
          }
        if (sumBinNum == 0)
          percent = 0;
        else
          percent = histoBinCounts[i]/static_cast<double>(sumBinNum)*100;
        cout << "Bin >max: " << histoBinCounts[i] << "  (" << percent
             << "%)\n";
        
        return true;
      }
  }

