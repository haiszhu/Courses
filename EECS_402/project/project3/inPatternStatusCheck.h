/* ----------------------------------------------------------------------
 * inPatternStatusCheck.h
 *
 * 03/21/2017 Hai Zhu
 *
 * check status of read in pattern header file
 ------------------------------------------------------------------------
 */
#ifndef _INPATTERNSTATUSCHECK_H_
#define _INPATTERNSTATUSCHECK_H_

#include "constants.h"

void inPatternStatusCheck(string &patternFname, int &row, int &column)
{
  bool validInputFound = false;
  while (!validInputFound)
  {
    cout << "Enter string for file name containing pattern: ";
    cin >> patternFname;
    ifstream inPatternFile;
    inPatternFile.open(patternFname.c_str());
    if (inPatternFile.fail())   //if in fail state
    {
      cout << "Unable to open input file, or file doesn't exist!" << endl;
    }
    else
    {
      inPatternFile >> column >> row ;
      //if no pattern included
      if ((column != MEMPTY)&&(row != MEMPTY))
      {
        validInputFound = true;      
      }
      else
      {
        cout << "Empty pattern file, please try again: " << endl;  
      }
    }
    inPatternFile.close();
  } 
}

#endif
