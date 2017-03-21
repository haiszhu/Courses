/* ----------------------------------------------------------------------
 * constants.h
 *
 * 03/21/2017 Hai Zhu
 *
 * global constants header file
 ------------------------------------------------------------------------
 */
#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

//global constants used in image processing project
const int INI_LOCATION = 0;

//user menu option 
const int UNI_OPT = 0;  //uninitialized user option
const int ANN_REC = 1;  //annotate image with rectangle
const int ANN_PAT = 2;  //annotate image with pattern
const int INS_IMA = 3;  //insert another image
const int WRI_OUT = 4;  //write out current image
const int EXIT = 5;     //exit program

//rectangle type
const int REC_ULLR = 1; //specify upper left and lower right corners
const int REC_ULD = 2;  //specify upper left and dimensions of rectangle
const int REC_CRL = 3;  //specify extent from center of rectangle

//rectangle fill in option
const int REC_NFIL = 1;
const int REC_FIL = 2;

//color option
const int RED = 1;  // color option for rectangle and pattern
const int GREEN = 2;
const int BLUE = 3;
const int BLACK = 4;
const int WHITE = 5;

//image type etc.
const string MTYPE = "P3";
const int MEMPTY = 0;
const int MAX_RGB = 255;
const int MIN_RGB = 0;

//flag to distingush orginal ppm file and added one
const int ORIGINAL = 1;
const int ADDED = 0;

//default image size
const int WIDTH = 640;
const int HEIGHT = 480;

//bad status ignoer num
const int SKIP = 200;

//patter identifier
const int PATTERN_ID = 1;

#endif
