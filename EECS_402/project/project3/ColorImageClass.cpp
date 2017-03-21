/* ----------------------------------------------------------------------
 * ColorImageClass.cpp
 *
 * 03/21/2017 Hai Zhu
 *
 * Color Image Class ctor dtor, and member functions.
 ------------------------------------------------------------------------
 */
#include <iostream>
#include <fstream>
using namespace std;
#include "ColorImageClass.h"
#include "constants.h"

//default constructor
ColorImageClass::ColorImageClass():m_width(WIDTH),m_height(HEIGHT)
{
  color_image = new ColorClass[m_width * m_height];
}

//value constructor
ColorImageClass::ColorImageClass(int width, int height)
                :m_width(width), m_height(height)
{ 
  color_image = new ColorClass[m_width * m_height];
}

//destructor
ColorImageClass::~ColorImageClass()
{
  delete [] color_image;
}

//get pixel color at given location
void ColorImageClass::getPixel(PixelLocationClass pixelLocation, 
                               ColorClass &pixelColor)
{
  int r,g,b;
  int x,y;
  pixelLocation.getPixelLocation(x,y);
  color_image[m_width * y + x].getColor(r,g,b);
  pixelColor.setColor(r,g,b);
}

//set pixel color at given location
void ColorImageClass::setPixel(PixelLocationClass pixelLocation,
                               ColorClass pixelColor)
{
  int r,g,b;
  int x,y;
  pixelLocation.getPixelLocation(x,y);
  pixelColor.getColor(r,g,b);
  color_image[m_width * y + x].setColor(r,g,b);  
}

//save color image class variable to ppm file
void ColorImageClass::saveP3Format(std::string path)
{
  int red,green,blue;
  std::fstream f;
  f.open(path.c_str(), std::ios_base::out);
    
  f << "P3" << endl;
  f << m_width << " " << m_height << endl;
  f << "255" << std::endl;
  
  for (int i = 0; i < m_height; i++)
  {
    for (int j = 0; j < m_width; j++)
    {
      color_image[m_width * i + j].getColor(red,green,blue);
      f << red << " " << green << " " << blue << " "; 
    }
    f << endl;
  }
  f.close();
    
}

//read ppm file to color image class variable
bool ColorImageClass::readColorImage(string fname, string &word, 
                              int &xsize, int &ysize, int &maxrgb)
{
  int rr,gg,bb;      
  ifstream inFile;
  inFile.open(fname.c_str());
    
  inFile >> word;
    
  inFile >> xsize;
  inFile >> ysize;
  inFile >> maxrgb;
    
  for (int i = 0; i < m_width * m_height; i++)
  {

    inFile >> rr >> gg >> bb;
    color_image[i].setColor(rr,gg,bb);
  }
  inFile.close();
      
}

