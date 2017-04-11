/* -----------------------------------------------------------------
 * constants.h
 *
 * 04/09/2017 Hai Zhu
 *
 * global constants header file
 -------------------------------------------------------------------
 */
#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

//global constants used in initialization
const int INI_VALUE = 0;
const int INI_FNAME = 1;
const double INI_VALUE_D = 0.0;

//attraction related constants
const int NUM_OF_PRIORITY = 3;
const int NUM_OF_SEATS = 12;
const int IDEAL_NUM_SFP = 6;
const int IDEAL_NUM_FP = 3;
const int IDEAL_NUM_STD = 3;


//distribution min and max for rider type
const int UNI_MIN = 0;
const int UNI_MAX = 100;

//rider type
const int RIDER_SFP = 0;
const int RIDER_FP = 1;
const int RIDER_STD = 2;

#endif
