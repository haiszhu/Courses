/* ----------------------------------------------------------------------
 * main.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * This is the main function to generate proj3.exe executable.
 * To generate executable, run Makefile.
 * 
 * Functionality explain:
 * This executable will read, change and write a specify type image 
 * file .ppm, with the following possible options:
 *
 * 1. Annotate image with rectangle
 * 2. Annotate image with pattern from file
 * 3. Insert another image
 * 4. Write out current image
 * 5. Exit the program
 *
 * You will get these instructions along using this program.
 ------------------------------------------------------------------------
 */
#include <iostream>
#include <fstream>
using namespace std;

//include constants header file
#include "constants.h"

//include class header files
#include "ColorImageClass.h"
#include "ColorClass.h"
#include "PixelLocationClass.h"
#include "RectangleClass.h"

//include global function header files
#include "printMainMenu.h"
#include "inFileStatusCheck.h"
#include "inOptionStatusCheck.h"
#include "inPatternStatusCheck.h"
#include "rectangleSetUp.h"

int main()
{
  //initializtion-------------------------------------------------------
  //integers
  int usrOption(UNI_OPT);                    //main menu option
  int recType(UNI_OPT);                      //rectangle type
  int xsize(UNI_OPT);                        //ppm image size
  int ysize(UNI_OPT);
  int maxrgb(UNI_OPT);                       //ppm image color scale
  int r(UNI_OPT), g(UNI_OPT), b(UNI_OPT);    //color related
  int xstart, ystart, width, height;         //pixel related
  int rgb;                                   //usr color option
  int fillOpt;                               //usr fill rectangle option
  int column, row;                           //pattern file size
  int flag(ORIGINAL);                        //original file or added one
  int* pattern;                              //pattern recording
    
  //strings
  string fname;                              //original ppm file name                 
  string patternFname;                       //pattern file name
  string anotherFname;                       //added file name
  string outFname;                           //output file name
  string imageType;                          //image type 

    
  //image file name read in----------------------------------------------
  //status check for original image file
  inFileStatusCheck(fname, imageType, xsize, ysize, maxrgb, flag);
  
  //class ininitialization
  ColorImageClass p(xsize,ysize);
  ColorClass pixelColor(r,g,b), transColor(r,g,b);
  PixelLocationClass pixelLocation(UNI_OPT,UNI_OPT);
  RectangleClass rectangle;
  
  //image file read in
  p.readColorImage(fname,imageType,xsize,ysize,maxrgb);
  
    
  //user interface-------------------------------------------------------
  printMainMenu();                          //print main menu
  inOptionStatusCheck(usrOption);           //status check
    
  while (usrOption != EXIT)
  {
      
    //Annotate image with rectangle--------------------------------------
    if (usrOption == ANN_REC)
    {
      rectangle.printRectangleMenu();       //print rectangle option
      inOptionStatusCheck(recType);         //status check
      rectangleSetUp(recType,rectangle);    //setup rectangle class
            
      //get rectangle dimensions and color
      rectangle.getRectangle(xstart,ystart,width,height);
      pixelColor.printColorMenu();
      cout << "Enter int for rectangle color: ";
      inOptionStatusCheck(rgb);             //status check
      pixelColor.setColor(rgb);
          
      //get user required fillin option
      fillOpt = UNI_OPT;                    
      cout << REC_NFIL << ". No" << endl;
      cout << REC_FIL << ". Yes" << endl;
      cout << "Enter int for rectangle fill option: ";
      inOptionStatusCheck(fillOpt);         //status check
      if (fillOpt == REC_NFIL)              //not fill in
      {
        for (int i=xstart; i<xstart+width; i++)
        {
          pixelLocation.setPixelLocation(i,ystart);
          p.setPixel(pixelLocation,pixelColor);
          pixelLocation.setPixelLocation(i,ystart+height);
          p.setPixel(pixelLocation,pixelColor);
        }

        for (int j=ystart; j<ystart+height; j++)
        {
          pixelLocation.setPixelLocation(xstart,j);
          p.setPixel(pixelLocation,pixelColor);
          pixelLocation.setPixelLocation(xstart+width,j);
          p.setPixel(pixelLocation,pixelColor);
        }
      }  
      else if (fillOpt == REC_FIL)          //fill in
      {
        for (int i=xstart; i<xstart+width; i++)
        {
          for (int j=ystart; j<ystart+height; j++)
          {
            pixelLocation.setPixelLocation(i,j);
            p.setPixel(pixelLocation,pixelColor);
          }
        }  
      }
    }
            
    //Annotate image with pattern from file------------------------------        
	else if (usrOption == ANN_PAT)
    {
      //get user typed in pattern file name
      inPatternStatusCheck(patternFname, row, column);      
      cout << "Enter upper left corner of pattern row and column: ";
      inOptionStatusCheck(ystart);           //status check
      inOptionStatusCheck(xstart);           //status check
        
      //get user required pattern color    
      pixelColor.printColorMenu();
      cout << "Enter int for pattern color: ";
      inOptionStatusCheck(rgb);              //status check
      pixelColor.setColor(rgb);
      
      //read in pattern file and record
      ifstream inPatternFile;
      inPatternFile.open(patternFname.c_str());
      inPatternFile >> column >> row;
      pattern = new int[row*column];
      for (int j=0; j<row; j++)
      {
        for (int i=0; i<column; i++)
        {
          inPatternFile >> pattern[j*column+i]; 
        }
      }
    
      //modify original ppm file according to pattern
      for (int i=0; i<column; i++)
      {
        for (int j=0; j<row; j++)
        {
          if (pattern[j*column+i] == PATTERN_ID)
          {
            pixelLocation.setPixelLocation(xstart+i,ystart+j);
            p.setPixel(pixelLocation,pixelColor);        
          }
        }
      }
      inPatternFile.close();                //close file  
    }
      
    //Insert another ppm picture ----------------------------------------   
    else if (usrOption == INS_IMA)
    {
      //get user typed in image name to be inserted
      flag = ADDED;
      inFileStatusCheck(anotherFname, imageType, xsize, 
                        ysize, maxrgb, flag);
    
      //get image dimension and transparency color
      cout << "Enter upper left corner of pattern row and column: ";
      inOptionStatusCheck(ystart);          //status check
      inOptionStatusCheck(xstart);          //status check
      pixelColor.printColorMenu();
      cout << "Enter int for transparecy color: ";
      inOptionStatusCheck(rgb);
      transColor.setColor(rgb);
        
      //read in image file to be inserted
      ColorImageClass pAdd(xsize,ysize);
      pAdd.readColorImage(anotherFname,imageType,xsize,ysize,maxrgb);
    
      //insert to original image     
      for (int j=0; j<ysize; j++)
      {
        for (int i=0; i<xsize; i++)
        {
          pixelLocation.setPixelLocation(i,j);
          pAdd.getPixel(pixelLocation,pixelColor);
          pixelColor.getColor(r,g,b);
          if (!transColor.compareColor(r,g,b))
          {
            pixelLocation.setPixelLocation(xstart+i,ystart+j);
            p.setPixel(pixelLocation,pixelColor);        
          }
        }
      } 
    }
      
    //Write out current image ------------------------------------------- 
    else if (usrOption == WRI_OUT)
    {
      cout << "Enter string for PPM file name to output: ";
      cin >> outFname;
      p.saveP3Format(outFname);
    }
    
    //Invalid option provided -------------------------------------------
    else
    {
      cout << "Not a valid option please try again according to the menu!"
           << endl;
    }
    
    //Ask user for next operation
    printMainMenu();
    cin >> usrOption;
  }
    
  //End of program ------------------------------------------------------
  cout << "Thank you for using this program" << endl;
  return 0;
}
