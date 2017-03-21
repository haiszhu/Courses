/* ----------------------------------------------------------------------
 * inFileStatusCheck.h
 *
 * 03/21/2017 Hai Zhu
 *
 * check status of read in file name
 ------------------------------------------------------------------------
 */
#ifndef _INFILESTATUSCHECK_H_
#define _INFILESTATUSCHECK_H_

#include "constants.h"

void inFileStatusCheck(string &fname,string &imageType,int &xsize, 
                       int &ysize,int &maxrgb,int flag)
{
  bool validInputFound = false;
  while (!validInputFound)
  {
    if (flag == ORIGINAL)   //original file to be modified
    {
      cout << "Enter string for PPM image file name to load: ";
    }
    else                    //ppm file to be inserted to original
    {
      cout << "Enter string for file name of PPM image to insert: ";
    }
    cin >> fname;
    ifstream inFile;
    inFile.open(fname.c_str());
    if (inFile.fail())      //if in fail state
    {
      cout << "Unable to open input file, or file doesn't exist! " 
           << endl;
    }
    else
    {
      inFile >> imageType >> xsize >> ysize >> maxrgb;
      if (!imageType.compare(MTYPE))    //if right image type
      {
        //if image size and color scales are all right
        if ((xsize != MEMPTY)&&(ysize != MEMPTY)&&(maxrgb == MAX_RGB))
        {
          validInputFound = true;      
        }
        else
        {
          cout << "Empty image or color scale is nor right" << endl;  
        }
      }
      else
      {
        cout << "Image type is expected to be PPM, try again! " 
             << endl;      
      }
    }
    inFile.close();
  } 
}

#endif