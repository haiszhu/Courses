/* ----------------------------------------------------------------------
 * RectangleClass.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * rectangle class ctor dtor, and member functions.
 ------------------------------------------------------------------------
 */
#include <iostream>
using namespace std;
#include "constants.h"
#include "RectangleClass.h"

//default ctor
RectangleClass::RectangleClass()
               :xstart(INI_LOCATION),ystart(INI_LOCATION)
{
  width = INI_LOCATION;
  height = INI_LOCATION;    
}

//dtor
RectangleClass::~RectangleClass()
{
  ;
}

//print rectangle type option
void RectangleClass::printRectangleMenu()
{
  cout << REC_ULLR <<
    ". Specify upper left and lower right corners of rectangle" << endl;
  cout << REC_ULD << 
    ". Specify upper left corner and dimensions of rectangle" << endl;
  cout << REC_CRL <<
    ". Specify extent from center of rectangle" << endl;
  cout << "Enter int for rectangle specification method: ";
}

//set rectangle class attributes
void RectangleClass::setRectangle(int x,int y,int w,int h)
{
  xstart = x;
  ystart = y;
  width = w;
  height = h;
}

//get rectangle class attributes
void RectangleClass::getRectangle(int &x,int &y,int &w,int &h)
{
  x = xstart;
  y = ystart;
  w = width;
  h = height;
}