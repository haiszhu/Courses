/* ----------------------------------------------------------------------
 * PixelLocationClass.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * Pixel Location Class ctor dtor, and member functions.
 ------------------------------------------------------------------------
 */
#include <iostream>
using namespace std;
#include "constants.h"
#include "PixelLocationClass.h"

//default constructor
PixelLocationClass::PixelLocationClass():x(INI_LOCATION),y(INI_LOCATION)
{
  ;
}

//value constructor
PixelLocationClass::PixelLocationClass(int i,int j):x(i),y(j)
{
  ;
}

//default destructor
PixelLocationClass::~PixelLocationClass()
{
  ;
}

//get pixel location attributes
void PixelLocationClass::getPixelLocation(int &idx_x,int &idx_y)
{
  idx_x = x;
  idx_y = y;
}

//set pixel location attributes
void PixelLocationClass::setPixelLocation(int idx_x,int idx_y)
{
  x = idx_x;
  y = idx_y;
}

