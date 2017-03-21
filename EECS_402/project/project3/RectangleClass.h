/* ----------------------------------------------------------------------
 * RectangleClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * rectangle class header file
 ------------------------------------------------------------------------
 */
#ifndef _RECTANGLECLASS_H_
#define _RECTANGLECLASS_H_

class RectangleClass
{
  public:
    RectangleClass();                                   //default ctor
    ~RectangleClass();                                  //dtor
    
    void printRectangleMenu();                          //print type 
    void setRectangle(int x,int y,int w,int h);         //set attributes
    void getRectangle(int &x,int &y,int &w,int &h);  //get attributes

  private:
    int xstart;                                         //upper left x
    int ystart;                                         //upper left y                                
    int width;                                          //rec dimension
    int height;
};

#endif