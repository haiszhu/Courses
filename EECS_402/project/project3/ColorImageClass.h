/* ----------------------------------------------------------------------
 * ColorImageClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * Color Image Class header file
 ------------------------------------------------------------------------
 */
#ifndef _COLORIMAGECLASS_H_
#define _COLORIMAGECLASS_H_

#include "ColorClass.h"
#include "PixelLocationClass.h"

class ColorImageClass
{
  public:
    ColorImageClass();                                  //default ctor
    ColorImageClass(int width, int height);             //value ctor
    ~ColorImageClass();                                 //dtor
    
    void getPixel(PixelLocationClass pixelLocation, 
                  ColorClass &pixelColor);              //get pixel color
    void setPixel(PixelLocationClass pixelLocation,
                  ColorClass pixelColor);               //set pixel color
    void saveP3Format(std::string path);                //save ppm
    bool readColorImage(string fname,string &word, 
                        int &xsize,int &ysize,int &maxrgb);//read ppm
    
  private:
    int m_width;                                        //image width
    int m_height;                                       //image height
    ColorClass* color_image;                            //image object
};

#endif
