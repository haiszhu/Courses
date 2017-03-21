/* ----------------------------------------------------------------------
 * PixelLocationClass.h
 *
 * 03/21/2017 Hai Zhu
 *
 * pixel location class header file
 ------------------------------------------------------------------------
 */
#ifndef _PIXELLOCATIONCLASS_H_
#define _PIXELLOCATIONCLASS_H_

class PixelLocationClass
{
  public:
    PixelLocationClass();                           //default ctor
    PixelLocationClass(int i,int j);                //value ctor
    ~PixelLocationClass();                          //dtor
    
    void getPixelLocation(int &idx_x,int &idx_y);   //get attributes
    void setPixelLocation(int idx_x,int idx_y);     //set attributes

  private:
    int x;                                          //horizontal
    int y;                                          //vertical 
};

#endif
