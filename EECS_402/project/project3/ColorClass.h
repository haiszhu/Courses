/* ----------------------------------------------------------------------
 * ColorClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * Color Class header file
 ------------------------------------------------------------------------
 */
#ifndef _COLORCLASS_H_
#define _COLORCLASS_H_

class ColorClass
{
  public:
    ColorClass();                                   //default ctor
    ColorClass(int red,int green,int blue);         //value ctor
    ~ColorClass();                                  //dtor
    
    void printColorMenu();                          //print color option    
    void getColor(int &red,int &green,int &blue);   //get rgb
    void setColor(int red,int green,int blue);      //set rgb
    void setColor(int rgb);                         //according to option    
    bool compareColor(int red,int green,int blue);  //compare color
    
  private:
    int r;                                          //red
    int g;                                          //green
    int b;                                          //blue
};

#endif
