/* ----------------------------------------------------------------------
 * ColorClass.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * Color Class ctor dtor, and member functions.
 ------------------------------------------------------------------------
 */
#include <iostream>
using namespace std;
#include "constants.h"
#include "ColorClass.h"

//default constructor
ColorClass::ColorClass():r(MAX_RGB),g(MAX_RGB),b(MAX_RGB)
{ 
  ;
}

//value constructor
ColorClass::ColorClass(int red,int green,int blue):
                       r(red),g(green),b(blue)
{ 
  ;
}

//destructor
ColorClass::~ColorClass()
{ 
  ;
}

//color menu option
void ColorClass::printColorMenu()
{
  cout << RED << ". Red" << endl;
  cout << GREEN << ". Green" << endl;
  cout << BLUE << ". Blue" << endl;
  cout << BLACK << ". Black" << endl;
  cout << WHITE << ". White" << endl;
}

//get color class attributes
void ColorClass::getColor(int &red,int &green,int &blue)
{
  red = r;
  green = g;
  blue = b;
}

//set color class attributes
void ColorClass::setColor(int red,int green,int blue)
{
  r = red;
  g = green;
  b = blue;
}

//set color class attributes according to color menu option
void ColorClass::setColor(int rgb)
{
  if (rgb == RED)
  {
    r = MAX_RGB;
    g = MIN_RGB;
    b = MIN_RGB;
  }
  else if (rgb == GREEN )
  {
    r = MIN_RGB;
    g = MAX_RGB;
    b = MIN_RGB;
  }
  else if (rgb == BLUE)
  {
    r = MIN_RGB;
    g = MIN_RGB;
    b = MAX_RGB;
  }
  else if (rgb == BLACK)
  {
    r = MIN_RGB;
    g = MIN_RGB;
    b = MIN_RGB;
  }
  else if (rgb == WHITE)
  {
    r = MAX_RGB;
    g = MAX_RGB;
    b = MAX_RGB;
  }
  else
  {
    cout << "Not a valid color option, please try again with" <<
        " red, green, blue, black, and white." << endl;
  }    
}

//compare color class with given r g b data
bool ColorClass::compareColor(int red,int green,int blue)
{
  if ((r == red)&&(g == green)&&(b == blue))
  {
    return true;
  }
  else
  {
    return false;
  }
}
