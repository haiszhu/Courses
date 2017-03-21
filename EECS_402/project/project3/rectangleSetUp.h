/* ----------------------------------------------------------------------
 * rectangleSetUp.h
 *
 * 03/21/2017 Hai Zhu
 *
 * rectangleSetUp function header file
 * it sets up uniform attributes needed to determine a rectangle
 * it has issues when run on my personal computer if using state check
 ------------------------------------------------------------------------
 */
#ifndef _RECTANGLESETUP_H_
#define _RECTANGLESETUP_H_

#include "constants.h"

void rectangleSetUp(int recType,RectangleClass &rectangle)
{
  int width(UNI_OPT), height(UNI_OPT);
  if (recType == REC_ULLR)
  {
      int upperLeftRow,upperLeftColumn;
      int lowerRightRow,lowerRightColumn;
      cout << "Enter upper left corner row and column: ";
      cin >> upperLeftRow >> upperLeftColumn;
      //inOptionStatusCheck(upperLeftRow);
      //inOptionStatusCheck(lowerRightColumn);
      //could run into segmentation fault, and couldn't fix it
              
      cout << "Enter lower right corner row and column: ";
      cin >> lowerRightRow >> lowerRightColumn;
      //inOptionStatusCheck(lowerRightRow);
      //inOptionStatusCheck(lowerRightColumn);
      width = lowerRightColumn - upperLeftColumn;
      height = lowerRightRow - upperLeftRow;
      rectangle.setRectangle(upperLeftColumn,upperLeftRow,
                width, height);
  }
  else if (recType == REC_ULD)
  {
    int upperLeftRow,upperLeftColumn;
    cout << "Enter upper left corner row and column: ";
    cin >> upperLeftRow >> upperLeftColumn;
    //inOptionStatusCheck(upperLeftRow);
    //inOptionStatusCheck(upperLeftColumn);
    cout << "Enter int for number of rows: ";
    cin >> height;
    //inOptionStatusCheck(height);
    cout << "Enter int for number of columns: ";
    cin >> width;
    //inOptionStatusCheck(width);
    rectangle.setRectangle(upperLeftColumn,upperLeftRow,
                width, height);      
  }
  else if (recType == REC_CRL)
  {
    int upperLeftRow,upperLeftColumn;
    int centerRow,centerColumn;
    int halfWidth,halfHeight;
    cout << "Enter rectangle center row and column: ";
    cin >> centerRow >> centerColumn;
    //inOptionStatusCheck(centerRow);
    //inOptionStatusCheck(centerColumn);
    cout << "Enter int for half number of rows: ";
    cin >> halfHeight;
    //inOptionStatusCheck(halfHeight);
    cout << "Enter int for half number of columns: ";
    cin >> halfWidth;
    //inOptionStatusCheck(halfWidth);
    upperLeftColumn = centerColumn - halfWidth;
    upperLeftRow = centerRow - halfHeight;
    width = halfWidth + halfWidth;
    height = halfHeight + halfHeight;
    rectangle.setRectangle(upperLeftColumn,upperLeftRow,
                width, height);
  }
}

#endif